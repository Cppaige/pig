import cv2
import os
import numpy as np
from pathlib import Path
from predict import PigPersonAnnotator


class VideoProcessor:
    def __init__(self, annotator):
        """
        视频处理器

        Args:
            annotator: PigPersonAnnotator 实例
        """
        self.annotator = annotator

    def process_video(self, video_path, output_path, progress_callback=None):
        """
        处理视频文件，逐帧检测并生成带标注的视频

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            progress_callback: 进度回调函数 callback(current, total, current_frame_path)

        Returns:
            dict: 处理结果统计
        """
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or not fps:
            fps = 25.0  # 默认帧率
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"视频信息: {width}x{height}, {fps} fps, {total_frames} 帧")

        # 创建视频写入器 - 使用更兼容的编码器
        # 尝试使用 H.264 编码器 (如果可用)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 编码器
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print("avc1 编码器不可用，尝试 mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print("mp4v 也不行，尝试 XVID")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise ValueError("无法创建视频写入器，请检查 OpenCV 安装")

        print(f"使用编码器: {fourcc}")

        # 统计信息
        stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'total_pigs': 0,
            'total_persons': 0,
            'frame_results': []
        }

        # 创建临时目录存储帧
        temp_dir = os.path.join(os.path.dirname(output_path), 'temp_frames')
        os.makedirs(temp_dir, exist_ok=True)

        try:
            frame_count = 0
            success_count = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # 每 N 帧处理一次（提高速度）
                # 如果需要每帧都处理，设置 skip_frames = 1
                skip_frames = 1
                if frame_count % skip_frames != 0:
                    continue

                # 保存当前帧为临时图片
                temp_frame_path = os.path.join(temp_dir, f'frame_{frame_count:06d}.jpg')
                cv2.imwrite(temp_frame_path, frame)

                # 使用 annotator 进行检测和可视化
                try:
                    annotated_frame, coco_result = self.annotator.visualize(
                        temp_frame_path,
                        output_path=None  # 不保存文件
                    )

                    # 统计当前帧的检测结果
                    annotations = coco_result.get('annotations', [])
                    pig_count = sum(1 for ann in annotations if ann.get('category_name') == 'pig')
                    person_count = sum(1 for ann in annotations if ann.get('category_name') == 'person')

                    stats['total_pigs'] += pig_count
                    stats['total_persons'] += person_count

                    # 在帧上添加统计信息
                    info_text = f"Frame: {frame_count}/{total_frames} | Pigs: {pig_count} | Persons: {person_count}"
                    cv2.putText(
                        annotated_frame,
                        info_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    # 记录帧结果
                    stats['frame_results'].append({
                        'frame_number': frame_count,
                        'pig_count': pig_count,
                        'person_count': person_count,
                        'total_objects': len(annotations)
                    })

                    # 将 BGR 格式转回（visualize 返回的是 RGB）
                    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

                    # 确保尺寸一致
                    if annotated_frame_bgr.shape[:2] != (height, width):
                        annotated_frame_bgr = cv2.resize(annotated_frame_bgr, (width, height))

                    # 写入输出视频
                    out.write(annotated_frame_bgr)
                    success_count += 1

                    # 更新进度
                    stats['processed_frames'] = frame_count

                    if progress_callback:
                        progress_callback(frame_count, total_frames, temp_frame_path)

                    if frame_count % 10 == 0:
                        print(f"已处理 {frame_count}/{total_frames} 帧")

                except Exception as e:
                    print(f"处理帧 {frame_count} 时出错: {str(e)}")
                    # 出错时使用原始帧
                    out.write(frame)

                # 清理临时帧文件
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)

            print(f"成功写入 {success_count} 帧")

            # 添加最终统计帧（显示在视频最后 3 秒）
            self._add_summary_frame(out, width, height, stats, fps)

        finally:
            # 清理
            cap.release()
            out.release()

            # 删除临时目录
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)

        # 验证输出文件
        if not os.path.exists(output_path):
            raise ValueError("输出视频文件未生成")

        file_size = os.path.getsize(output_path)
        if file_size == 0:
            raise ValueError("输出视频文件大小为 0")

        print(f"输出视频大小: {file_size / 1024 / 1024:.2f} MB")

        return stats

    def _add_summary_frame(self, video_writer, width, height, stats, fps, duration_sec=3):
        """
        在视频末尾添加统计信息帧

        Args:
            video_writer: 视频写入器
            width: 视频宽度
            height: 视频高度
            stats: 统计信息
            fps: 帧率
            duration_sec: 显示时长（秒）
        """
        summary_frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 绘制半透明背景
        overlay = summary_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (240, 240, 240), -1)
        summary_frame = cv2.addWeighted(overlay, 0.3, summary_frame, 0.7, 0)

        # 添加统计文本
        texts = [
            "视频检测统计报告",
            f"=" * 30,
            f"总帧数: {stats['total_frames']}",
            f"检测到的猪总数: {stats['total_pigs']}",
            f"检测到的人总数: {stats['total_persons']}",
            f"平均每帧猪数量: {stats['total_pigs'] / max(stats['processed_frames'], 1):.2f}",
            f"平均每帧人数量: {stats['total_persons'] / max(stats['processed_frames'], 1):.2f}",
        ]

        y_offset = 80
        for i, text in enumerate(texts):
            y = y_offset + i * 50
            if i == 0:
                # 标题
                cv2.putText(summary_frame, text, (width//2 - 200, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 120, 255), 3)
            else:
                # 内容
                cv2.putText(summary_frame, text, (100, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # 写入多帧（显示 3 秒）
        num_frames = fps * duration_sec
        for _ in range(num_frames):
            video_writer.write(summary_frame)

    def extract_frames(self, video_path, output_dir, max_frames=100):
        """
        从视频中提取关键帧（用于预览）

        Args:
            video_path: 视频路径
            output_dir: 输出目录
            max_frames: 最多提取的帧数

        Returns:
            list: 提取的帧路径列表
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

        os.makedirs(output_dir, exist_ok=True)
        frame_paths = []

        for idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame_path = os.path.join(output_dir, f'preview_{idx:03d}.jpg')
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)

        cap.release()

        return frame_paths
