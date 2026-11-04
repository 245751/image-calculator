"""COCOeval adapter shared by training and class/font evaluation."""

from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco(records, class_ids, iou_thresholds, image_indices=None):
    """Evaluate existing predictions without changing model inference.

    Records contain image_index, gt_boxes/gt_labels, and
    pred_boxes/pred_labels/pred_scores as CPU tensors in xyxy coordinates.
    Undefined AP values are returned as None.
    """
    label_names = {label: str(label) for label in class_ids}
    with redirect_stdout(StringIO()):
        return _evaluate_coco(
            records, label_names, tuple(iou_thresholds), image_indices
        )


def _evaluate_coco(records, label_names, iou_thresholds, image_indices=None):
    """正解・予測から、全体mAPとクラス別・IoU別APを返す。"""
    records = list(records)
    available_ids = [record["image_index"] for record in records]
    if len(available_ids) != len(set(available_ids)):
        raise ValueError("image_indexが重複しています")

    if image_indices is not None:
        selected_ids = set(image_indices)
        if not selected_ids <= set(available_ids):
            raise ValueError("存在しないimage_indexが指定されています")
        records = [
            record for record in records
            if record["image_index"] in selected_ids
        ]
    if not records:
        return {
            "map": None,
            "class_aps": {
                label: {threshold: None for threshold in iou_thresholds}
                for label in label_names
            },
        }
    if not label_names or any(label <= 0 for label in label_names):
        raise ValueError("背景を除くクラスIDを指定してください")

    def to_coco_box(box):
        xmin, ymin, xmax, ymax = map(float, box)
        width, height = xmax - xmin, ymax - ymin
        if not np.isfinite([xmin, ymin, xmax, ymax]).all():
            raise ValueError("検出枠に非有限値があります")
        if width < 0 or height < 0:
            raise ValueError("検出枠の幅・高さを負にはできません")
        return [xmin, ymin, width, height]

    images = []
    annotations = []
    detections = []

    for record in records:
        image_id = int(record["image_index"])
        images.append({"id": image_id})

        for box, label in zip(
            record["gt_boxes"].tolist(),
            record["gt_labels"].tolist(),
            strict=True,
        ):
            if label not in label_names:
                raise ValueError(f"未知の正解クラスID: {label}")
            bbox = to_coco_box(box)
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": int(label),
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            })

        for box, label, score in zip(
            record["pred_boxes"].tolist(),
            record["pred_labels"].tolist(),
            record["pred_scores"].tolist(),
            strict=True,
        ):
            if label not in label_names:
                raise ValueError(f"未知の予測クラスID: {label}")
            if not np.isfinite(score):
                raise ValueError("信頼度に非有限値があります")
            detections.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": to_coco_box(box),
                "score": float(score),
            })

    categories = [
        {"id": label, "name": name}
        for label, name in sorted(label_names.items())
    ]
    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }
    coco_gt.createIndex()

    if detections:
        coco_dt = coco_gt.loadRes(detections)
    else:
        # loadRes([])は使えないため、空の予測データを直接作る。
        coco_dt = COCO()
        coco_dt.dataset = {
            "images": images,
            "categories": categories,
            "annotations": [],
        }
        coco_dt.createIndex()

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = [image["id"] for image in images]
    evaluator.params.catIds = sorted(label_names)
    evaluator.params.iouThrs = np.asarray(iou_thresholds, dtype=float)
    evaluator.params.recThrs = np.linspace(0.0, 1.0, 101)
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.params.areaRng = [evaluator.params.areaRng[0]]
    evaluator.params.areaRngLbl = ["all"]
    evaluator.evaluate()
    evaluator.accumulate()

    # 配列の軸: IoU、Recall、クラス、面積区分、最大検出数
    max_det_index = evaluator.params.maxDets.index(100)
    precision = evaluator.eval["precision"][:, :, :, 0, max_det_index]

    def mean_valid(values):
        valid = values[values >= 0]  # -1は未定義なので除外する。
        return float(valid.mean()) if valid.size else None

    class_aps = {
        label: {
            threshold: mean_valid(precision[t, :, k])
            for t, threshold in enumerate(iou_thresholds)
        }
        for k, label in enumerate(evaluator.params.catIds)
    }
    return {
        "map": mean_valid(precision),
        "class_aps": class_aps,
    }

