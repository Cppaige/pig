import json
import os
from collections import defaultdict

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def _polygon_area(points):
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _choose_largest_polygon(polygons):
    best = None
    best_area = 0.0
    for poly in polygons:
        if len(poly) < 3:
            continue
        area = _polygon_area(poly)
        if area > best_area:
            best_area = area
            best = poly
    return best


def _segmentation_to_polygon(segmentation, image_size):
    if isinstance(segmentation, list) and segmentation:
        polygons = []
        for seg in segmentation:
            if not isinstance(seg, list) or len(seg) < 6:
                continue
            coords = list(zip(seg[0::2], seg[1::2]))
            polygons.append(coords)
        return _choose_largest_polygon(polygons)

    if isinstance(segmentation, dict):
        height, width = image_size[1], image_size[0]
        rle = segmentation
        if isinstance(rle.get('counts'), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        mask = mask_utils.decode(rle)
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contour = contours[0].reshape(-1, 2)
        return [(float(x), float(y)) for x, y in contour]

    return None


def convert_coco_to_yolo_seg(split_dir, annotation_file, labels_dir):
    with open(annotation_file, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    categories = coco.get('categories', [])
    if not categories:
        raise ValueError('COCO 标注中未找到类别信息。')

    id_to_index = {cat['id']: idx for idx, cat in enumerate(categories)}
    names = [cat['name'] for cat in categories]

    images = {img['id']: img for img in coco.get('images', [])}
    ann_by_image = defaultdict(list)
    for ann in coco.get('annotations', []):
        ann_by_image[ann['image_id']].append(ann)

    os.makedirs(labels_dir, exist_ok=True)

    for image_id, image_info in images.items():
        file_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']

        label_lines = []
        for ann in ann_by_image.get(image_id, []):
            if ann.get('iscrowd', 0) == 1:
                continue
            class_id = id_to_index.get(ann['category_id'])
            if class_id is None:
                continue

            polygon = _segmentation_to_polygon(ann.get('segmentation'), (width, height))
            if polygon is None or len(polygon) < 3:
                continue

            normalized = []
            for x, y in polygon:
                normalized.append(x / width)
                normalized.append(y / height)

            if len(normalized) < 6:
                continue

            line = f"{class_id} " + " ".join(f"{v:.6f}" for v in normalized)
            label_lines.append(line)

        label_name = os.path.splitext(file_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(label_lines))

    return names


def find_splits(dataset_root):
    splits = {}
    for entry in os.listdir(dataset_root):
        split_dir = os.path.join(dataset_root, entry)
        if not os.path.isdir(split_dir):
            continue
        annotation_file = os.path.join(split_dir, '_annotations.coco.json')
        if os.path.exists(annotation_file):
            split_key = entry
            if entry.lower() in ['valid', 'validation']:
                split_key = 'val'
            splits[split_key] = {
                'dir': split_dir,
                'annotation': annotation_file
            }
    return splits


def write_data_yaml(output_path, dataset_root, splits, names):
    lines = [
        f"path: {dataset_root}",
        f"train: {os.path.basename(splits['train']['dir'])}",
        f"val: {os.path.basename(splits['val']['dir'])}",
    ]

    if 'test' in splits:
        lines.append(f"test: {os.path.basename(splits['test']['dir'])}")

    lines.append(f"names: {names}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    return output_path
