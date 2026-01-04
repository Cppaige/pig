import os
import cv2
import numpy as np
from ultralytics import YOLO


class PigPersonAnnotator:
    def __init__(self, model_path, conf=0.25, iou=0.7, device=None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.names = self.model.names

        # 添加去重参数
        self.enable_deduplication = True  # 是否启用跨帧去重
        self.dedup_iou_threshold = 0.5    # 去重 IoU 阈值
        self.dedup_distance_threshold = 50  # 去重距离阈值（像素）

    def _color_for_class(self, class_id):
        palette = [
            (255, 99, 71),
            (54, 162, 235),
            (255, 206, 86),
            (75, 192, 192),
            (153, 102, 255),
            (255, 159, 64),
        ]
        return palette[class_id % len(palette)]

    def _build_coco(self, image_path, image_size, boxes, labels, scores, masks_xy, mask_data):
        annotations = []
        image_info = {
            'file_name': os.path.basename(image_path),
            'width': image_size[0],
            'height': image_size[1]
        }

        for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            segmentation = []
            area = float(width * height)

            if masks_xy is not None and idx < len(masks_xy):
                polygon = masks_xy[idx]
                if polygon is not None and len(polygon) >= 3:
                    flat = []
                    for x, y in polygon:
                        flat.extend([float(x), float(y)])
                    if len(flat) >= 6:
                        segmentation = [flat]
                        if mask_data is not None and idx < len(mask_data):
                            area = float(mask_data[idx].sum())

            annotation = {
                'id': idx + 1,
                'image_id': 1,
                'category_id': int(label),
                'category_name': self.names.get(int(label), str(int(label))),
                'bbox': [float(x1), float(y1), float(width), float(height)],
                'area': area,
                'score': float(score),
                'segmentation': segmentation,
                'iscrowd': 0
            }
            annotations.append(annotation)

        return {
            'image_info': image_info,
            'annotations': annotations
        }

    def predict(self, image_path):
        results = self.model.predict(
            image_path,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )
        result = results[0]

        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
        labels = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.array([])

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image_size = (image.shape[1], image.shape[0])

        masks_xy = result.masks.xy if result.masks is not None else None
        mask_data = result.masks.data.cpu().numpy().astype(bool) if result.masks is not None else None
        if mask_data is not None:
            mask_h, mask_w = mask_data.shape[1], mask_data.shape[2]
            if (mask_w, mask_h) != image_size:
                resized = []
                for mask in mask_data:
                    resized_mask = cv2.resize(
                        mask.astype(np.uint8),
                        image_size,
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    resized.append(resized_mask)
                if resized:
                    mask_data = np.stack(resized, axis=0)

        coco_result = self._build_coco(
            image_path,
            image_size,
            boxes,
            labels,
            scores,
            masks_xy,
            mask_data
        )

        return coco_result, boxes, labels, scores, masks_xy, mask_data

    def visualize(self, image_path, output_path=None):
        coco_result, boxes, labels, scores, masks_xy, mask_data = self.predict(image_path)

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        overlay = image.copy()
        alpha = 0.5

        if mask_data is not None:
            for idx, mask in enumerate(mask_data):
                color = self._color_for_class(int(labels[idx]))
                overlay[mask] = (
                    overlay[mask].astype(np.float32) * (1 - alpha) +
                    np.array(color, dtype=np.float32) * alpha
                ).astype(np.uint8)

        result_image = overlay

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            color = self._color_for_class(int(labels[idx]))
            label_name = self.names.get(int(labels[idx]), str(int(labels[idx])))
            score = scores[idx] if idx < len(scores) else 0.0

            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            text = f"{label_name} {score:.2f}"
            cv2.putText(result_image, text, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if masks_xy is not None and idx < len(masks_xy):
                polygon = masks_xy[idx]
                if polygon is not None and len(polygon) >= 3:
                    pts = np.array(polygon, dtype=np.int32)
                    cv2.polylines(result_image, [pts], isClosed=True, color=color, thickness=2)

        if output_path:
            cv2.imwrite(output_path, result_image)

        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        return result_image_rgb, coco_result

    def batch_predict(self, image_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_files = [f for f in os.listdir(image_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        all_results = []

        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            output_path = os.path.join(output_dir, f'annotated_{image_file}')

            _, coco_result = self.visualize(image_path, output_path)
            all_results.append(coco_result)

        return all_results
