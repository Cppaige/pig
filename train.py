import os
import yaml
from ultralytics import YOLO
from utils.coco_to_yolo_seg import convert_coco_to_yolo_seg, find_splits, write_data_yaml


CONFIG_PATH = 'config.yaml'


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def prepare_dataset(dataset_root, data_yaml_path):
    splits = find_splits(dataset_root)
    if 'train' not in splits or 'val' not in splits:
        raise ValueError('数据集需包含 train 和 val 目录，并且包含 _annotations.coco.json。')

    names = None
    for split_name, info in splits.items():
        labels_dir = os.path.join(info['dir'], 'labels')
        names = convert_coco_to_yolo_seg(info['dir'], info['annotation'], labels_dir)

    if names is None:
        raise ValueError('无法从 COCO 标注中读取类别名称。')

    write_data_yaml(data_yaml_path, dataset_root, splits, names)
    return data_yaml_path


def main():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f'未找到配置文件：{CONFIG_PATH}')

    config = load_config(CONFIG_PATH)

    dataset_root = os.path.abspath(config.get('data_root', ''))
    if not dataset_root:
        raise ValueError('config.yaml 中 data_root 不能为空。')

    data_yaml = os.path.join(dataset_root, 'data.yaml')
    prepare_dataset(dataset_root, data_yaml)

    model = YOLO(config.get('model', 'yolo11n-seg.pt'))
    model.train(
        data=data_yaml,
        task='segment',
        epochs=int(config.get('epochs', 100)),
        imgsz=int(config.get('imgsz', 640)),
        batch=int(config.get('batch', 8)),
        patience=int(config.get('patience', 0)),
        device=config.get('device'),
        project=config.get('project', 'runs/segment'),
        name=config.get('name', 'pig-person-seg')
    )


if __name__ == '__main__':
    main()
