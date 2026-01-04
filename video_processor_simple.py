import cv2
import os
import numpy as np
from pathlib import Path
from predict import PigPersonAnnotator


class VideoProcessorSimple:
    """简化的视频处理器 - 生成图片序列而不是视频（更可靠）"""

    def __init__(self, annotator):
        self.annotator = annotator

    def process_video_to_images(self, video_path, output_dir,
                               frame_skip=5,
                               conf_threshold=0.25,
                               enable_deduplication=True,
                               progress_callback=None):
        """
        处理视频，生成标注后的图片序列

        Args:
            video_path: 输入视频路径
            output_dir: 输出图片目录
            frame_skip: 帧采样间隔（1=每帧，5=每5帧）
            conf_threshold: 置信度阈值
            enable_deduplication: 是否启用跨帧去重
            progress_callback: 进度回调函数

        Returns:
            dict: 处理结果统计
        """
        # 更新 annotator 参数
        self.annotator.conf = conf_threshold
        self.annotator.enable_deduplication = enable_deduplication

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or not fps:
            fps = 25.0

        print(f"视频信息: {fps} fps, {total_frames} 帧")
        print(f"处理设置: 采样间隔={frame_skip}, 置信度={conf_threshold}, 去重={enable_deduplication}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 统计信息
        stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'total_pigs': 0,
            'total_persons': 0,
            'frame_results': [],
            'output_dir': output_dir,
            'fps': fps,
            'frame_skip': frame_skip,
            'dedup_enabled': enable_deduplication
        }

        # 跨帧去重：记录每只猪的中心点和边界框
        tracked_pigs = []
        TRACKING_IOU_THRESHOLD = 0.3  # IoU 阈值
        TRACKING_DISTANCE_THRESHOLD = 50  # 距离阈值（像素）- 更保守
        MAX_TRACKING_AGE = 30 * frame_skip  # 最大跟踪帧数（避免误判）

        # 创建临时目录存储帧
        temp_dir = os.path.join(output_dir, 'temp_frames')
        os.makedirs(temp_dir, exist_ok=True)

        # IoU 计算函数
        def calculate_iou(box1, box2):
            """计算两个边界框的 IoU"""
            x1_min, y1_min, w1, h1 = box1
            x2_min, y2_min, w2, h2 = box2

            x1_max = x1_min + w1
            y1_max = y1_min + h1
            x2_max = x2_min + w2
            y2_max = y2_min + h2

            # 计算交集
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
                return 0.0

            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

            # 计算并集
            box1_area = w1 * h1
            box2_area = w2 * h2
            union_area = box1_area + box2_area - inter_area

            if union_area == 0:
                return 0.0

            return inter_area / union_area

        try:
            frame_count = 0
            processed_count = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # 帧采样：只处理每隔 N 帧
                if frame_count % frame_skip != 0:
                    continue

                # 保存当前帧为临时图片
                temp_frame_path = os.path.join(temp_dir, f'frame_{frame_count:06d}.jpg')
                cv2.imwrite(temp_frame_path, frame)

                # 使用 annotator 进行检测和可视化
                try:
                    annotated_frame, coco_result = self.annotator.visualize(
                        temp_frame_path,
                        output_path=None
                    )

                    # 统计当前帧的检测结果
                    annotations = coco_result.get('annotations', [])

                    # 跨帧去重逻辑
                    unique_pig_count = 0
                    unique_person_count = 0

                    for ann in annotations:
                        category_name = ann.get('category_name', '')
                        bbox = ann.get('bbox', [])  # [x, y, width, height]

                        if category_name == 'pig' and len(bbox) == 4:
                            # 计算中心点
                            center_x = bbox[0] + bbox[2] / 2
                            center_y = bbox[1] + bbox[3] / 2

                            if enable_deduplication:
                                # 检查是否与已有的猪匹配
                                matched = False

                                for tracked in tracked_pigs:
                                    # 方法1：计算 IoU（更准确）
                                    iou = calculate_iou(bbox, tracked['bbox'])

                                    # 方法2：计算中心点距离
                                    tracked_center = tracked['center']
                                    distance = np.sqrt((center_x - tracked_center[0])**2 +
                                                     (center_y - tracked_center[1])**2)

                                    # 综合判断：IoU 高 或 距离近 = 同一只猪
                                    if iou > TRACKING_IOU_THRESHOLD or distance < TRACKING_DISTANCE_THRESHOLD:
                                        # 更新跟踪记录（使用新的位置）
                                        tracked['bbox'] = bbox
                                        tracked['center'] = (center_x, center_y)
                                        tracked['frame'] = frame_count
                                        matched = True
                                        break  # 找到匹配就停止

                                if matched:
                                    # 匹配到了已有的猪，说明这是同一只猪在后续帧中被再次检测到
                                    # 关键：不重复计数！只记录一次
                                    pass
                                else:
                                    # 没有匹配到任何已有的猪，是新猪
                                    tracked_pigs.append({
                                        'bbox': bbox,
                                        'center': (center_x, center_y),
                                        'frame': frame_count
                                    })
                                    unique_pig_count += 1  # 只有新猪才计数
                            else:
                                unique_pig_count += 1

                        elif category_name == 'person':
                            unique_person_count += 1

                    # 清理过期的跟踪记录（避免误判）
                    if enable_deduplication and processed_count % 10 == 0:
                        current_frame = frame_count
                        tracked_pigs = [
                            t for t in tracked_pigs
                            if current_frame - t['frame'] < MAX_TRACKING_AGE
                        ]

                    stats['total_pigs'] += unique_pig_count
                    stats['total_persons'] += unique_person_count

                    # 在帧上添加统计信息
                    dedup_note = " (去重)" if enable_deduplication else ""
                    info_text = f"Frame: {frame_count}/{total_frames} | Pigs: {unique_pig_count}{dedup_note} | Persons: {unique_person_count}"
                    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    cv2.putText(
                        annotated_frame_bgr,
                        info_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    # 保存标注后的图片
                    output_frame_path = os.path.join(output_dir, f'annotated_frame_{processed_count:06d}.jpg')
                    cv2.imwrite(output_frame_path, annotated_frame_bgr)

                    # 记录帧结果
                    stats['frame_results'].append({
                        'frame_number': frame_count,
                        'pig_count': unique_pig_count,
                        'person_count': unique_person_count,
                        'total_objects': unique_pig_count + unique_person_count,
                        'image_path': f'annotated_frame_{processed_count:06d}.jpg'
                    })

                    # 更新进度
                    stats['processed_frames'] = processed_count
                    processed_count += 1

                    if progress_callback:
                        progress_callback(frame_count, total_frames, temp_frame_path)

                    if frame_count % 50 == 0:
                        print(f"已处理 {frame_count}/{total_frames} 帧，生成 {processed_count} 张图片")

                except Exception as e:
                    print(f"处理帧 {frame_count} 时出错: {str(e)}")

                # 清理临时帧文件
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)

        finally:
            # 清理
            cap.release()

            # 删除临时目录
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)

        # 计算统计信息
        if len(stats['frame_results']) > 0:
            avg_pigs = stats['total_pigs'] / len(stats['frame_results'])
            avg_persons = stats['total_persons'] / len(stats['frame_results'])

            print(f"\n{'='*50}")
            print(f"{'视频处理完成':^40}")
            print(f"{'='*50}")
            print(f"总帧数: {total_frames}")
            print(f"采样间隔: 每 {frame_skip} 帧")
            print(f"生成图片: {len(stats['frame_results'])} 张")
            print(f"-" * 50)
            print(f"{'统计结果（已启用跨帧去重）':^40}")
            print(f"-" * 50)
            print(f"检测到的猪总数（去重后）: {stats['total_pigs']} 只")
            print(f"平均每张图片猪数量: {avg_pigs:.2f} 只")
            print(f"检测到的人总数: {stats['total_persons']} 个")
            print(f"平均每张图片人数量: {avg_persons:.2f} 个")
            print(f"-" * 50)
            print(f"\n💡 结果解读:")
            print(f"   - 猪总数代表视频中检测到的所有不同的猪")
            print(f"   - 如果数量仍然偏高，建议：")
            print(f"     1. 减小采样间隔（改为每 5-10 帧）")
            print(f"     2. 提高置信度阈值（改为高 0.40）")
            print(f"     3. 查看生成的图片进行人工核对")
            print(f"{'='*50}\n")

        return stats
