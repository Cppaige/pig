from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
import os
from werkzeug.utils import secure_filename
from predict import PigPersonAnnotator
from video_processor import VideoProcessor
from video_processor_simple import VideoProcessorSimple
import json
from datetime import datetime
import sqlite3
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['VIDEO_FOLDER'] = 'static/videos'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 最大 500MB 文件（支持视频）
app.config['DATABASE'] = 'annotations_history.db'
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'flv'}

# 确保文件夹存在
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER'], app.config['VIDEO_FOLDER']]:
    os.makedirs(folder, exist_ok=True)


# 数据库初始化
def init_db():
    """初始化数据库表"""
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            original_path TEXT NOT NULL,
            result_path TEXT NOT NULL,
            json_path TEXT NOT NULL,
            pig_count INTEGER DEFAULT 0,
            person_count INTEGER DEFAULT 0,
            total_objects INTEGER DEFAULT 0,
            total_area REAL DEFAULT 0.0,
            avg_area REAL DEFAULT 0.0
        )
    ''')
    conn.commit()
    conn.close()


# 启动时初始化数据库
init_db()


def save_annotation_to_db(filename, original_path, result_path, json_path, stats):
    """保存标注记录到数据库"""
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO annotations
        (filename, original_path, result_path, json_path, pig_count, person_count, total_objects, total_area, avg_area)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        filename,
        original_path,
        result_path,
        json_path,
        stats.get('pigs', 0),
        stats.get('persons', 0),
        stats.get('total_objects', 0),
        stats.get('total_area', 0.0),
        stats.get('avg_area', 0.0)
    ))
    conn.commit()
    conn.close()

# 加载模型
model_path = os.getenv('MODEL_PATH', './runs/segment/pig-person-seg/weights/best.pt')
conf_threshold = float(os.getenv('CONF_THRESHOLD', '0.25'))
iou_threshold = float(os.getenv('IOU_THRESHOLD', '0.7'))
model_device = os.getenv('MODEL_DEVICE') or None

annotator = PigPersonAnnotator(
    model_path,
    conf=conf_threshold,
    iou=iou_threshold,
    device=model_device
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}


def allowed_file(filename):
    """检查是否为允许的图片格式"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_video_file(filename):
    """检查是否为允许的视频格式"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_VIDEO_EXTENSIONS']


def _count_by_name(annotations, name):
    name = name.lower()
    return len([a for a in annotations if a.get('category_name', '').lower() == name])


def _class_counts(annotations):
    counts = {}
    for ann in annotations:
        cname = ann.get('category_name', 'unknown')
        counts[cname] = counts.get(cname, 0) + 1
    return counts


def _mask_stats(annotations):
    mask_count = 0
    total_area = 0.0
    for ann in annotations:
        segmentation = ann.get('segmentation', [])
        if segmentation:
            mask_count += 1
        total_area += float(ann.get('area', 0.0))
    avg_area = total_area / len(annotations) if annotations else 0.0
    return mask_count, total_area, avg_area


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '未找到上传文件'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    if file and allowed_file(file.filename):
        # 保存上传文件
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"

        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(upload_path)

        # 生成结果文件名
        result_filename = f"annotated_{unique_filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

        try:
            # 推理并可视化
            annotated_image, coco_result = annotator.visualize(upload_path, result_path)

            # 保存 COCO 风格结果
            annotation_filename = f"annotations_{unique_filename.rsplit('.', 1)[0]}.json"
            annotation_path = os.path.join(app.config['RESULT_FOLDER'], annotation_filename)

            with open(annotation_path, 'w') as f:
                json.dump(coco_result, f, indent=2)

            annotations = coco_result.get('annotations', [])
            mask_count, total_area, avg_area = _mask_stats(annotations)

            stats = {
                'total_objects': len(annotations),
                'mask_objects': mask_count,
                'total_area': round(total_area, 2),
                'avg_area': round(avg_area, 2),
                'pigs': _count_by_name(annotations, 'pig'),
                'persons': _count_by_name(annotations, 'person'),
                'class_counts': _class_counts(annotations),
                'image_size': f"{coco_result['image_info']['width']}x{coco_result['image_info']['height']}"
            }

            # 保存到数据库
            save_annotation_to_db(
                filename,
                f'/static/uploads/{unique_filename}',
                f'/static/results/{result_filename}',
                f'/static/results/{annotation_filename}',
                stats
            )

            return jsonify({
                'success': True,
                'original_image': f'/static/uploads/{unique_filename}',
                'annotated_image': f'/static/results/{result_filename}',
                'annotations': annotations,
                'stats': stats,
                'annotation_file': f'/static/results/{annotation_filename}'
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': '文件类型不支持'}), 400


@app.route('/batch_upload', methods=['POST'])
def batch_upload():
    if 'files[]' not in request.files:
        return jsonify({'error': '未找到文件列表'}), 400

    files = request.files.getlist('files[]')
    results = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{filename}"

            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(upload_path)

            result_filename = f"annotated_{unique_filename}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

            try:
                annotated_image, coco_result = annotator.visualize(upload_path, result_path)

                annotation_filename = f"annotations_{unique_filename.rsplit('.', 1)[0]}.json"
                annotation_path = os.path.join(app.config['RESULT_FOLDER'], annotation_filename)

                with open(annotation_path, 'w') as f:
                    json.dump(coco_result, f, indent=2)

                annotations = coco_result.get('annotations', [])
                mask_count, total_area, avg_area = _mask_stats(annotations)

                stats = {
                    'total_objects': len(annotations),
                    'mask_objects': mask_count,
                    'total_area': round(total_area, 2),
                    'avg_area': round(avg_area, 2),
                    'pigs': _count_by_name(annotations, 'pig'),
                    'persons': _count_by_name(annotations, 'person'),
                    'class_counts': _class_counts(annotations)
                }

                # 保存到数据库
                save_annotation_to_db(
                    filename,
                    f'/static/uploads/{unique_filename}',
                    f'/static/results/{result_filename}',
                    f'/static/results/{annotation_filename}',
                    stats
                )

                results.append({
                    'filename': filename,
                    'original': f'/static/uploads/{unique_filename}',
                    'annotated': f'/static/results/{result_filename}',
                    'annotation_file': f'/static/results/{annotation_filename}',
                    'objects': len(annotations),
                    'stats': {
                        'mask_objects': mask_count,
                        'total_area': round(total_area, 2),
                        'avg_area': round(avg_area, 2),
                        'class_counts': _class_counts(annotations)
                    }
                })

            except Exception as e:
                results.append({
                    'filename': filename,
                    'error': str(e)
                })

    return jsonify({
        'success': True,
        'results': results,
        'total_files': len(files),
        'processed_files': len([r for r in results if 'error' not in r])
    })


@app.route('/download_annotations/<filename>')
def download_annotations(filename):
    annotation_path = os.path.join(app.config['RESULT_FOLDER'], filename)

    if os.path.exists(annotation_path):
        return send_file(annotation_path, as_attachment=True)

    return jsonify({'error': '文件不存在'}), 404


@app.route('/batch_upload_stream', methods=['POST'])
def batch_upload_stream():
    """支持实时进度反馈的批量上传接口（使用SSE）"""

    if 'files[]' not in request.files:
        return jsonify({'error': '未找到文件列表'}), 400

    files = request.files.getlist('files[]')

    # 第一步：在流式传输前，先将所有文件保存到临时位置
    temp_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            unique_filename = f"{timestamp}_{filename}"

            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

            # 保存文件（此时文件流还未关闭）
            file.save(upload_path)
            temp_files.append({
                'filename': filename,
                'unique_filename': unique_filename,
                'upload_path': upload_path
            })

    def generate():
        """SSE生成器函数"""
        total = len(temp_files)
        results = []

        for idx, file_info in enumerate(temp_files):
            filename = file_info['filename']
            unique_filename = file_info['unique_filename']
            upload_path = file_info['upload_path']

            # 发送进度更新
            progress = {
                'type': 'progress',
                'current': idx + 1,
                'total': total,
                'percentage': round((idx + 1) / total * 100, 1),
                'filename': filename
            }
            yield f"data: {json.dumps(progress)}\n\n"

            result_filename = f"annotated_{unique_filename}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

            try:
                annotated_image, coco_result = annotator.visualize(upload_path, result_path)

                annotation_filename = f"annotations_{unique_filename.rsplit('.', 1)[0]}.json"
                annotation_path = os.path.join(app.config['RESULT_FOLDER'], annotation_filename)

                with open(annotation_path, 'w') as f:
                    json.dump(coco_result, f, indent=2)

                annotations = coco_result.get('annotations', [])
                mask_count, total_area, avg_area = _mask_stats(annotations)

                stats = {
                    'total_objects': len(annotations),
                    'mask_objects': mask_count,
                    'total_area': round(total_area, 2),
                    'avg_area': round(avg_area, 2),
                    'pigs': _count_by_name(annotations, 'pig'),
                    'persons': _count_by_name(annotations, 'person'),
                    'class_counts': _class_counts(annotations)
                }

                # 保存到数据库
                save_annotation_to_db(
                    filename,
                    f'/static/uploads/{unique_filename}',
                    f'/static/results/{result_filename}',
                    f'/static/results/{annotation_filename}',
                    stats
                )

                results.append({
                    'filename': filename,
                    'original': f'/static/uploads/{unique_filename}',
                    'annotated': f'/static/results/{result_filename}',
                    'annotation_file': f'/static/results/{annotation_filename}',
                    'objects': len(annotations),
                    'stats': {
                        'mask_objects': mask_count,
                        'total_area': round(total_area, 2),
                        'avg_area': round(avg_area, 2),
                        'class_counts': _class_counts(annotations)
                    }
                })

            except Exception as e:
                results.append({
                    'filename': filename,
                    'error': str(e)
                })

        # 发送完成信号
        complete = {
            'type': 'complete',
            'results': results,
            'total_files': total,
            'processed_files': len([r for r in results if 'error' not in r])
        }
        yield f"data: {json.dumps(complete)}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/history')
def history():
    """历史记录页面"""
    return render_template('history.html')


@app.route('/api/history')
def get_history():
    """获取历史记录API"""
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 获取查询参数
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    offset = (page - 1) * per_page

    # 获取总数
    cursor.execute('SELECT COUNT(*) as total FROM annotations')
    total = cursor.fetchone()['total']

    # 获取记录（按时间倒序）
    cursor.execute('''
        SELECT * FROM annotations
        ORDER BY upload_time DESC
        LIMIT ? OFFSET ?
    ''', (per_page, offset))

    records = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify({
        'records': records,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total,
            'pages': (total + per_page - 1) // per_page
        }
    })


@app.route('/api/history/<int:record_id>', methods=['DELETE'])
def delete_history_record(record_id):
    """删除历史记录"""
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()

    # 先获取记录信息
    cursor.execute('SELECT * FROM annotations WHERE id = ?', (record_id,))
    record = cursor.fetchone()

    if not record:
        conn.close()
        return jsonify({'error': '记录不存在'}), 404

    # 删除数据库记录
    cursor.execute('DELETE FROM annotations WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()

    return jsonify({'success': True, 'message': '删除成功'})


@app.route('/api/history/clear', methods=['DELETE'])
def clear_history():
    """清空所有历史记录"""
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('DELETE FROM annotations')
    conn.commit()
    deleted = cursor.rowcount
    conn.close()

    return jsonify({'success': True, 'deleted': deleted})


# ==================== 视频处理路由 ====================

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """视频上传和处理"""
    if 'file' not in request.files:
        return jsonify({'error': '未找到上传文件'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    if file and allowed_video_file(file.filename):
        # 保存上传文件
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"

        upload_path = os.path.join(app.config['VIDEO_FOLDER'], unique_filename)
        file.save(upload_path)

        # 生成结果文件名
        result_filename = f"annotated_{unique_filename}"
        result_path = os.path.join(app.config['VIDEO_FOLDER'], result_filename)

        try:
            # 创建视频处理器
            processor = VideoProcessor(annotator)

            # 处理视频（无进度回调）
            stats = processor.process_video(upload_path, result_path)

            return jsonify({
                'success': True,
                'original_video': f'/static/videos/{unique_filename}',
                'annotated_video': f'/static/videos/{result_filename}',
                'stats': stats
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': '视频格式不支持'}), 400


@app.route('/upload_video_stream', methods=['POST'])
def upload_video_stream():
    """视频上传和处理（支持实时进度）"""

    if 'file' not in request.files:
        return jsonify({'error': '未找到上传文件'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    if not file or not allowed_video_file(file.filename):
        return jsonify({'error': '视频格式不支持'}), 400

    # 保存上传文件
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"

    upload_path = os.path.join(app.config['VIDEO_FOLDER'], unique_filename)
    file.save(upload_path)

    # 生成结果文件名
    result_filename = f"annotated_{unique_filename}"
    result_path = os.path.join(app.config['VIDEO_FOLDER'], result_filename)

    def generate():
        """SSE生成器函数"""
        try:
            # 创建视频处理器
            processor = VideoProcessor(annotator)

            # 自定义进度回调
            def progress_callback(current, total, frame_path):
                progress_data = {
                    'type': 'progress',
                    'current': current,
                    'total': total,
                    'percentage': round(current / total * 100, 1) if total > 0 else 0,
                    'status': f'处理帧 {current}/{total}'
                }
                yield f"data: {json.dumps(progress_data)}\n\n"

            # 处理视频
            stats = processor.process_video(
                upload_path,
                result_path,
                progress_callback=progress_callback
            )

            # 发送完成信号
            complete = {
                'type': 'complete',
                'original_video': f'/static/videos/{unique_filename}',
                'annotated_video': f'/static/videos/{result_filename}',
                'stats': stats
            }
            yield f"data: {json.dumps(complete)}\n\n"

        except Exception as e:
            # 发送错误信号
            error = {
                'type': 'error',
                'message': str(e)
            }
            yield f"data: {json.dumps(error)}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/upload_video_to_images', methods=['POST'])
def upload_video_to_images():
    """视频上传并生成图片序列（备用方案）"""

    if 'file' not in request.files:
        return jsonify({'error': '未找到上传文件'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    if not file or not allowed_video_file(file.filename):
        return jsonify({'error': '视频格式不支持'}), 400

    # 获取前端传来的参数
    frame_skip = int(request.form.get('frame_skip', 5))
    conf_threshold = float(request.form.get('conf_threshold', 0.25))
    enable_dedup = request.form.get('enable_dedup', 'true').lower() == 'true'

    print(f"📥 接收到参数: frame_skip={frame_skip}, conf_threshold={conf_threshold}, enable_dedup={enable_dedup}")

    # 保存上传文件
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"

    upload_path = os.path.join(app.config['VIDEO_FOLDER'], unique_filename)
    file.save(upload_path)

    # 生成输出目录
    output_dir = os.path.join(app.config['VIDEO_FOLDER'], f'frames_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    def generate():
        """SSE生成器函数"""
        try:
            # 创建简化版视频处理器
            processor = VideoProcessorSimple(annotator)

            # 自定义进度回调
            def progress_callback(current, total, frame_path):
                progress_data = {
                    'type': 'progress',
                    'current': current,
                    'total': total,
                    'percentage': round(current / total * 100, 1) if total > 0 else 0,
                    'status': f'处理帧 {current}/{total}'
                }
                yield f"data: {json.dumps(progress_data)}\n\n"

            # 处理视频生成图片序列（传递参数）
            stats = processor.process_video_to_images(
                upload_path,
                output_dir,
                frame_skip=frame_skip,
                conf_threshold=conf_threshold,
                enable_deduplication=enable_dedup,
                progress_callback=progress_callback
            )

            # 获取生成的图片���表
            image_files = sorted([f for f in os.listdir(output_dir) if f.startswith('annotated_')])

            # 发送完成信号
            complete = {
                'type': 'complete',
                'original_video': f'/static/videos/{unique_filename}',
                'output_dir': f'/static/videos/frames_{timestamp}',
                'images': image_files[:50],  # 只返回前50张预览
                'total_images': len(image_files),
                'stats': stats,
                'is_image_sequence': True,  # 标记为图片序列
                'settings': {
                    'frame_skip': frame_skip,
                    'conf_threshold': conf_threshold,
                    'enable_dedup': enable_dedup
                }
            }
            yield f"data: {json.dumps(complete)}\n\n"

        except Exception as e:
            # 发送错误信号
            error = {
                'type': 'error',
                'message': str(e)
            }
            yield f"data: {json.dumps(error)}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


if __name__ == '__main__':
    if not os.path.exists(model_path):
        print(f"未找到模型：{model_path}")
        print("请先训练模型或设置 MODEL_PATH 指向已有的 YOLOv11 分割权重。")

    app.run(debug=True, port=5000, threaded=True)
