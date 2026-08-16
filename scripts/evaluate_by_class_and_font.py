import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection._utils import retrieve_out_channels
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.ops import box_iou
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "outputs" / "checkpoints" / "ssd_calculator_state_dict.pth"
)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "val"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "logs"

LABEL_MAP = {
    "0": 1,
    "1": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    "5": 6,
    "6": 7,
    "7": 8,
    "8": 9,
    "9": 10,
    "+": 11,
    "-": 12,
    "x": 13,
    "=": 14,
}
LABEL_NAMES = {label: name for name, label in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP) + 1
IOU_THRESHOLDS = tuple(i / 100 for i in range(50, 100, 5))


class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set="train"):
        self.root = Path(root)
        self.image_dir = self.root / "JPEGImages"
        self.annotation_dir = self.root / "Annotations"
        split_file = self.root / "ImageSets" / "Main" / f"{image_set}.txt"
        self.image_ids = [
            line.strip()
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = Image.open(self.image_dir / f"{image_id}.jpg").convert("RGB")
        annotation = ET.parse(
            self.annotation_dir / f"{image_id}.xml"
        ).getroot()

        font_name = annotation.findtext("font")
        if not font_name:
            raise ValueError(
                f"{image_id}.xml に <font> がありません。"
                "generate_dataset.pyでvalデータを再生成してください。"
            )

        boxes = []
        labels = []
        for obj in annotation.findall("object"):
            class_name = obj.findtext("name")
            bbox = obj.find("bndbox")
            boxes.append(
                [
                    float(bbox.findtext("xmin")),
                    float(bbox.findtext("ymin")),
                    float(bbox.findtext("xmax")),
                    float(bbox.findtext("ymax")),
                ]
            )
            labels.append(LABEL_MAP[class_name])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return self.transform(image), target, font_name, image_id


def collate_batch(batch):
    return tuple(zip(*batch))


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model(model_path, device):
    model = ssd300_vgg16(weights=None, weights_backbone=None)
    model.head.classification_head = SSDClassificationHead(
        retrieve_out_channels(model.backbone, (300, 300)),
        model.anchor_generator.num_anchors_per_location(),
        NUM_CLASSES,
    )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {
            key.removeprefix("module."): value for key, value in checkpoint.items()
        }

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def run_inference(model, data_loader, device):
    records = []
    image_index = 0

    with torch.inference_mode():
        for images, targets, font_names, image_ids in tqdm(
            data_loader, desc="Evaluation inference"
        ):
            predictions = model([image.to(device) for image in images])

            for prediction, target, font_name, image_id in zip(
                predictions, targets, font_names, image_ids
            ):
                records.append(
                    {
                        "image_index": image_index,
                        "image_id": image_id,
                        "font": font_name,
                        "pred_boxes": prediction["boxes"].detach().cpu(),
                        "pred_labels": prediction["labels"].detach().cpu(),
                        "pred_scores": prediction["scores"].detach().cpu(),
                        "gt_boxes": target["boxes"].cpu(),
                        "gt_labels": target["labels"].cpu(),
                    }
                )
                image_index += 1

    return records


def prepare_class_data(records):
    predictions_by_class = {label: [] for label in LABEL_NAMES}
    ground_truths_by_class = {label: {} for label in LABEL_NAMES}

    for record in records:
        image_index = record["image_index"]
        for label in LABEL_NAMES:
            gt_boxes = record["gt_boxes"][record["gt_labels"] == label]
            pred_mask = record["pred_labels"] == label
            pred_boxes = record["pred_boxes"][pred_mask]
            pred_scores = record["pred_scores"][pred_mask]
            ground_truths_by_class[label][image_index] = gt_boxes

            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                iou_matrix = box_iou(pred_boxes, gt_boxes)
            else:
                iou_matrix = torch.empty((len(pred_boxes), len(gt_boxes)))

            for score, ious in zip(pred_scores.tolist(), iou_matrix):
                predictions_by_class[label].append((score, image_index, ious))

    for predictions in predictions_by_class.values():
        predictions.sort(key=lambda item: item[0], reverse=True)

    return predictions_by_class, ground_truths_by_class


def calculate_average_precision(predictions, ground_truths, iou_threshold):
    num_ground_truths = sum(len(boxes) for boxes in ground_truths.values())
    if num_ground_truths == 0:
        return None

    matched = {
        image_index: torch.zeros(len(boxes), dtype=torch.bool)
        for image_index, boxes in ground_truths.items()
    }
    true_positives = []
    false_positives = []

    for _, image_index, ious in predictions:
        unmatched_indices = torch.where(~matched[image_index])[0]
        if len(unmatched_indices) == 0:
            true_positives.append(0.0)
            false_positives.append(1.0)
            continue

        unmatched_ious = ious[unmatched_indices]
        best_iou, best_position = unmatched_ious.max(dim=0)
        if best_iou.item() >= iou_threshold:
            matched_index = unmatched_indices[best_position]
            matched[image_index][matched_index] = True
            true_positives.append(1.0)
            false_positives.append(0.0)
        else:
            true_positives.append(0.0)
            false_positives.append(1.0)

    if not predictions:
        return 0.0

    true_positives = torch.tensor(true_positives).cumsum(dim=0)
    false_positives = torch.tensor(false_positives).cumsum(dim=0)
    recalls = true_positives / num_ground_truths
    precisions = true_positives / (true_positives + false_positives)

    interpolated_precisions = []
    for recall_level in torch.linspace(0, 1, 101):
        candidates = precisions[recalls >= recall_level]
        interpolated_precisions.append(
            candidates.max().item() if len(candidates) > 0 else 0.0
        )
    return sum(interpolated_precisions) / len(interpolated_precisions)


def calculate_class_aps(
    predictions_by_class,
    ground_truths_by_class,
    image_indices=None,
):
    if image_indices is not None:
        image_indices = set(image_indices)

    results = {}
    for label in LABEL_NAMES:
        predictions = predictions_by_class[label]
        ground_truths = ground_truths_by_class[label]
        if image_indices is not None:
            predictions = [
                prediction
                for prediction in predictions
                if prediction[1] in image_indices
            ]
            ground_truths = {
                image_index: boxes
                for image_index, boxes in ground_truths.items()
                if image_index in image_indices
            }

        results[label] = {
            threshold: calculate_average_precision(
                predictions, ground_truths, threshold
            )
            for threshold in IOU_THRESHOLDS
        }

    return results


def mean_present_class_ap(class_aps, threshold):
    values = [
        threshold_aps[threshold]
        for threshold_aps in class_aps.values()
        if threshold_aps[threshold] is not None
    ]
    return sum(values) / len(values) if values else None


def write_character_csv(class_aps, output_path):
    threshold_columns = [f"ap_{threshold:.2f}" for threshold in IOU_THRESHOLDS]
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["class", *threshold_columns, "ap_0.50_0.95"])
        for label, class_name in LABEL_NAMES.items():
            values = [class_aps[label][threshold] for threshold in IOU_THRESHOLDS]
            mean_ap = (
                sum(value for value in values if value is not None)
                / sum(value is not None for value in values)
                if any(value is not None for value in values)
                else None
            )
            writer.writerow(
                [
                    class_name,
                    *["" if value is None else value for value in values],
                    "" if mean_ap is None else mean_ap,
                ]
            )


def write_font_csv(records, predictions_by_class, ground_truths_by_class, output_path):
    threshold_columns = [f"map_{threshold:.2f}" for threshold in IOU_THRESHOLDS]
    fonts = sorted({record["font"] for record in records})

    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["font", "num_images", *threshold_columns, "map_0.50_0.95"])

        for font_name in fonts:
            font_image_indices = {
                record["image_index"]
                for record in records
                if record["font"] == font_name
            }
            class_aps = calculate_class_aps(
                predictions_by_class,
                ground_truths_by_class,
                font_image_indices,
            )
            map_values = [
                mean_present_class_ap(class_aps, threshold)
                for threshold in IOU_THRESHOLDS
            ]
            valid_values = [value for value in map_values if value is not None]
            mean_map = (
                sum(valid_values) / len(valid_values) if valid_values else None
            )
            writer.writerow(
                [
                    font_name,
                    len(font_image_indices),
                    *["" if value is None else value for value in map_values],
                    "" if mean_map is None else mean_map,
                ]
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="検証データの文字クラス別APとフォント別mAPを計算します。"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-suffix",
        default="",
        help="出力CSV名の末尾に付ける識別子（例: seed42）",
    )
    parser.add_argument("--image-set", default="train")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if "/" in args.output_suffix or "\\" in args.output_suffix:
        raise ValueError("--output-suffixにはファイル名に使える識別子を指定してください")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    dataset = EvaluationDataset(args.data_dir, image_set=args.image_set)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    model = create_model(args.model, device)

    print(
        f"device={device}, model={args.model}, "
        f"validation images={len(dataset)}"
    )
    records = run_inference(model, data_loader, device)
    predictions_by_class, ground_truths_by_class = prepare_class_data(records)
    class_aps = calculate_class_aps(
        predictions_by_class, ground_truths_by_class
    )

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    character_csv = args.output_dir / f"ap_by_character{suffix}.csv"
    font_csv = args.output_dir / f"map_by_font{suffix}.csv"
    write_character_csv(class_aps, character_csv)
    write_font_csv(
        records,
        predictions_by_class,
        ground_truths_by_class,
        font_csv,
    )

    print(f"文字クラス別AP: {character_csv}")
    print(f"フォント別mAP: {font_csv}")


if __name__ == "__main__":
    main()
