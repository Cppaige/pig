# 猪与人实例分割（YOLOv11）

本项目提供一个简易 Web 界面，用于对图片中的猪和人进行实例分割。
基于 YOLOv11 分割模型，支持 COCO segmentation 标注数据，并将 mask 叠加到图片上进行可视化。

## 功能
- YOLOv11 实例分割（mask + box）
- COCO segmentation 数据集支持（train/val/test）
- Web 上传批量推理与结果展示
- 输出 COCO 风格 JSON 标注结果

## 目录结构
- `app.py`：Web 应用入口
- `predict.py`：YOLOv11 推理与 mask 可视化
- `train.py`：训练入口（读取 `config.yaml`）
- `config.yaml`：训练配置（数据集路径与训练参数）
- `utils/coco_to_yolo_seg.py`：COCO segmentation 转 YOLO 分割标签
- `templates/index.html`：Web 页面
- `static/uploads`：上传图片
- `static/results`：推理结果

## 环境依赖
- 建议 Python 3.9
- 安装依赖：

```powershell
pip install -r requirements.txt
```

## 数据集格式（COCO segmentation）
数据集根目录需包含 train/valid/test 等子目录，每个子目录内有 `_annotations.coco.json`：

```
dataset_root/
  train/
    _annotations.coco.json
    *.jpg
  valid/  (或 val/)
    _annotations.coco.json
    *.jpg
  test/   (可选)
    _annotations.coco.json
    *.jpg
```

`train.py` 会在每个子目录下生成 YOLO 分割标签 `labels/`，并在数据集根目录写入 `data.yaml`。

## 训练（读取 config.yaml）
训练入口在 `train.py`，参数统一在 `config.yaml` 中配置。

请先修改 `config.yaml`：

```yaml
data_root: H:/path/to/dataset_root
model: yolo11n-seg.pt
epochs: 100
imgsz: 640
batch: 8
device: null
project: runs/segment
name: pig-person-seg
```

然后直接运行：

```powershell
python train.py
```

训练完成后，权重默认保存在：
`runs\segment\pig-person-seg\weights\best.pt`

## 运行 Web（直接使用已有权重）
设置环境变量 `MODEL_PATH` 指向你的分割权重 `.pt`，再启动：

```powershell
set MODEL_PATH=H:\path\to\best.pt
set CONF_THRESHOLD=0.25
set IOU_THRESHOLD=0.7
set MODEL_DEVICE=0
python app.py
```

浏览器访问：`http://127.0.0.1:5000/`

## 环境变量说明（app.py）
- `MODEL_PATH`：分割权重路径（`.pt`）
- `CONF_THRESHOLD`：置信度阈值（默认 0.25）
- `IOU_THRESHOLD`：NMS 阈值（默认 0.7）
- `MODEL_DEVICE`：设备（`0`/`cpu`/空）
