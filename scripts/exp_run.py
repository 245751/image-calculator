import argparse
from contextlib import nullcontext
import math
import os
import random

import numpy as np
import torch
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import csv
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection._utils import retrieve_out_channels
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.ops import box_iou
from tqdm.auto import tqdm

NUM_CLASSES = 16  # 背景(0) + 数字(10) + 記号(5: +, -, ×, ÷, =)
RANDOM_SEEDS = (42, 43, 44)
NUM_EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 4e-3
WARMUP_EPOCHS = 4
MIN_LEARNING_RATE = 1e-5
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 1e-4
MAP_IOU_THRESHOLDS = tuple(i / 100 for i in range(50, 100, 5))
OUTPUT_BASE_DIR = os.path.join("outputs", "logs")


def set_random_seed(seed):
    """学習のランダム性を再現可能にする。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="指定したseedでSSDを学習します。"
    )
    parser.add_argument(
        "--output-dir-name",
        required=True,
        help="outputs/logsの下に作成する学習結果ディレクトリ名",
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=RANDOM_SEEDS,
        help="学習に使用するseed",
    )
    args = parser.parse_args()

    directory_name = args.output_dir_name
    if (
        not directory_name
        or directory_name in {".", ".."}
        or "/" in directory_name
        or "\\" in directory_name
    ):
        parser.error("--output-dir-nameにはディレクトリ名だけを指定してください")
    return args


class CustomVOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set='train', transforms=None, label_map=None):
        self.root = root
        self.image_dir = os.path.join(root, "JPEGImages")
        self.annotation_dir = os.path.join(root, "Annotations")
        self.transforms = transforms
        self.label_map = label_map or {
            '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6,
            '6': 7, '7': 8, '8': 9, '9': 10,
            '+': 11, '-': 12, '×': 13, '÷': 14, '=': 15
        }

        # image_set (train.txtなど) を読み込む
        split_file = os.path.join(root, "ImageSets", "Main", f"{image_set}.txt")
        with open(split_file) as f:
            self.image_ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        xml_path = os.path.join(self.annotation_dir, f"{image_id}.xml")

        img = Image.open(img_path).convert("RGB")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            name = obj.find("name").text
            label = self.label_map[name]
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target ,image_id

transform = transforms.ToTensor()
train_dataset = CustomVOCDataset(
    root="data/processed/train",
    image_set="train",
    transforms=transform,
)
# val/ImageSets/Main 内の分割ファイル名は train.txt
val_dataset = CustomVOCDataset(
    root="data/processed/val",
    image_set="train",
    transforms=transform,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=lambda batch: tuple(zip(*batch)),
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def amp_autocast():
    """CUDAではfloat16の自動混合精度を有効にする。"""
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def calculate_average_precision(predictions, ground_truths, iou_threshold):
    """1クラス分のAPを101点補間で計算する。"""
    num_ground_truths = sum(len(boxes) for boxes in ground_truths.values())
    if num_ground_truths == 0:
        return None

    matched = {
        image_index: torch.zeros(len(boxes), dtype=torch.bool)
        for image_index, boxes in ground_truths.items()
    }
    true_positives = []
    false_positives = []

    for score, image_index, pred_box in sorted(
        predictions, key=lambda item: item[0], reverse=True
    ):
        gt_boxes = ground_truths.get(image_index)
        if gt_boxes is None or len(gt_boxes) == 0:
            true_positives.append(0.0)
            false_positives.append(1.0)
            continue

        unmatched_indices = torch.where(~matched[image_index])[0]
        if len(unmatched_indices) == 0:
            true_positives.append(0.0)
            false_positives.append(1.0)
            continue

        ious = box_iou(
            pred_box.unsqueeze(0), gt_boxes[unmatched_indices]
        ).squeeze(0)
        best_iou, best_position = ious.max(dim=0)

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

    # COCO形式と同じく、recall 0.00〜1.00の101点で補間する。
    interpolated_precisions = []
    for recall_level in torch.linspace(0, 1, 101):
        candidates = precisions[recalls >= recall_level]
        interpolated_precisions.append(
            candidates.max().item() if len(candidates) > 0 else 0.0
        )

    return sum(interpolated_precisions) / len(interpolated_precisions)


def evaluate_map(model, data_loader):
    """IoU 0.50:0.95におけるクラス平均mAPを計算する。"""
    model.eval()
    predictions_by_class = {label: [] for label in range(1, NUM_CLASSES)}
    ground_truths_by_class = {label: {} for label in range(1, NUM_CLASSES)}
    image_index = 0

    with torch.inference_mode():
        for images, targets, _ in tqdm(data_loader, desc="Validation mAP", leave=False):
            with amp_autocast():
                predictions = model([image.to(device) for image in images])

            for prediction, target in zip(predictions, targets):
                pred_boxes = prediction["boxes"].detach().cpu()
                pred_labels = prediction["labels"].detach().cpu()
                pred_scores = prediction["scores"].detach().cpu()
                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()

                for label in range(1, NUM_CLASSES):
                    ground_truths_by_class[label][image_index] = gt_boxes[
                        gt_labels == label
                    ]

                for box, label, score in zip(
                    pred_boxes, pred_labels.tolist(), pred_scores.tolist()
                ):
                    predictions_by_class[label].append(
                        (score, image_index, box)
                    )

                image_index += 1

    average_precisions = []
    for iou_threshold in MAP_IOU_THRESHOLDS:
        for label in range(1, NUM_CLASSES):
            average_precision = calculate_average_precision(
                predictions_by_class[label],
                ground_truths_by_class[label],
                iou_threshold,
            )
            if average_precision is not None:
                average_precisions.append(average_precision)

    return sum(average_precisions) / max(len(average_precisions), 1)


def evaluate_loss(model, data_loader):
    """検証データの平均lossを、重みを更新せずに計算する。"""
    was_training = model.training

    # Torchvisionの物体検出モデルはtrainモードのときだけlossを返す。
    # BatchNormの移動平均は検証データで更新しないよう、個別にevalへ戻す。
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    running_loss = 0.0
    num_batches = 0

    try:
        with torch.no_grad():
            for images, targets, _ in tqdm(
                data_loader, desc="Validation loss", leave=False
            ):
                images = [image.to(device) for image in images]
                targets = [
                    {key: value.to(device) for key, value in target.items()}
                    for target in targets
                ]

                with amp_autocast():
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
                running_loss += loss.item()
                num_batches += 1
    finally:
        model.train(was_training)

    return running_loss / max(num_batches, 1)

def create_train_loader(seed):
    return DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )


def create_model():
    """COCO事前学習済みSSDを16クラス用に作り直す。"""
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    model.head.classification_head = SSDClassificationHead(
        retrieve_out_channels(model.backbone, (300, 300)),
        model.anchor_generator.num_anchors_per_location(),
        NUM_CLASSES,
    )
    return model.to(device)


def train_for_seed(seed, model_output_dir, csv_output_dir):
    """指定seedで学習し、そのseed内で最高mAPのモデルを保存する。"""
    set_random_seed(seed)
    train_loader = create_train_loader(seed)
    model = create_model()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=5e-4,
    )

    def learning_rate_multiplier(epoch):
        if epoch < WARMUP_EPOCHS:
            warmup_progress = epoch / WARMUP_EPOCHS
            return 0.1 + 0.9 * warmup_progress

        cosine_epochs = NUM_EPOCHS - WARMUP_EPOCHS
        cosine_progress = (epoch - WARMUP_EPOCHS) / max(cosine_epochs - 1, 1)
        minimum_multiplier = MIN_LEARNING_RATE / LEARNING_RATE
        return minimum_multiplier + (1.0 - minimum_multiplier) * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=learning_rate_multiplier,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model_path = os.path.join(
        model_output_dir, f"ssd_calculator_seed{seed}_state_dict.pth"
    )
    csv_path = os.path.join(
        csv_output_dir, f"training_results_seed{seed}.csv"
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    print(
        f"\n===== seed {seed} の学習を開始 =====\n"
        f"device: {device} / train images: {len(train_dataset)} "
        f"/ val images: {len(val_dataset)} "
        f"/ AMP: {'enabled' if use_amp else 'disabled'}"
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["seed", "epoch", "learning_rate", "train_loss", "val_loss", "map"]
        )

    best_map = float("-inf")
    best_epoch = 0
    early_stopping_best_map = float("-inf")
    epochs_without_map_improvement = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        current_learning_rate = optimizer.param_groups[0]["lr"]

        progress = tqdm(
            train_loader,
            desc=f"Seed {seed} | Epoch {epoch + 1}/{NUM_EPOCHS}",
        )
        for batch_index, (images, targets, image_ids) in enumerate(progress):
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]

            optimizer.zero_grad(set_to_none=True)

            with amp_autocast():
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                loss = torch.tensor(float("inf"))
                if not(torch.isfinite(loss)):
                    print(
                        f"NaN detected:epoch={epoch+1}"
                        f"batch={batch_index} image_index={image_ids}"
                          )
                    raise RuntimeError("Non-finite training loss detected")           

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = running_loss / len(train_loader)
        val_loss = evaluate_loss(model, val_loader)
        mean_average_precision = evaluate_map(model, val_loader)
        print(
            f"Seed {seed} / Epoch {epoch + 1}: "
            f"lr={current_learning_rate:.6g}, train_loss={mean_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"mAP@0.50:0.95={mean_average_precision:.4f}"
        )

        with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    seed,
                    epoch + 1,
                    current_learning_rate,
                    mean_loss,
                    val_loss,
                    mean_average_precision,
                ]
            )

        if mean_average_precision > best_map:
            best_map = mean_average_precision
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)
            print(
                f"最高mAPを更新したためモデルを保存しました: "
                f"seed={seed}, epoch={best_epoch}, "
                f"mAP={best_map:.4f}, path={model_path}"
            )

        if (
            mean_average_precision
            > early_stopping_best_map + EARLY_STOPPING_MIN_DELTA
        ):
            early_stopping_best_map = mean_average_precision
            epochs_without_map_improvement = 0
        else:
            epochs_without_map_improvement += 1

        if epochs_without_map_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping: mAPが{EARLY_STOPPING_PATIENCE} epoch連続で"
                f"{EARLY_STOPPING_MIN_DELTA:g}以上改善しなかったため、"
                f"epoch {epoch + 1}で学習を終了します。"
            )
            break

        scheduler.step()

    print(
        f"seed {seed} の最高mAPモデル: epoch={best_epoch}, "
        f"mAP@0.50:0.95={best_map:.4f}, path={model_path}"
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_map": best_map,
        "model_path": model_path,
        "csv_path": csv_path,
    }


def write_csv_atomically(path, rows):
    """並列ジョブから途中まで書かれたCSVが見えないように保存する。"""
    temporary_path = f"{path}.{os.getpid()}.tmp"
    try:
        with open(temporary_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def write_training_summaries(result, csv_output_dir):
    """seed別サマリーを書き、全seed完了時には統合版も生成する。"""
    header = ["seed", "best_epoch", "best_map", "model_path", "csv_path"]
    result_row = [
        result["seed"],
        result["best_epoch"],
        result["best_map"],
        result["model_path"],
        result["csv_path"],
    ]
    seed_summary_path = os.path.join(
        csv_output_dir, f"training_seed_summary_seed{result['seed']}.csv"
    )
    write_csv_atomically(seed_summary_path, [header, result_row])

    seed_summary_paths = [
        os.path.join(csv_output_dir, f"training_seed_summary_seed{seed}.csv")
        for seed in RANDOM_SEEDS
    ]
    if not all(os.path.exists(path) for path in seed_summary_paths):
        return seed_summary_path, None

    combined_rows = [header]
    for path in seed_summary_paths:
        with open(path, newline="", encoding="utf-8") as csv_file:
            rows = list(csv.reader(csv_file))
        if len(rows) != 2 or rows[0] != header:
            raise ValueError(f"不正なseed別サマリーです: {path}")
        combined_rows.append(rows[1])

    combined_summary_path = os.path.join(
        csv_output_dir, "training_seed_summary.csv"
    )
    write_csv_atomically(combined_summary_path, combined_rows)
    return seed_summary_path, combined_summary_path


if __name__ == "__main__":
    args = parse_args()
    run_output_dir = os.path.join(OUTPUT_BASE_DIR, args.output_dir_name)
    model_output_dir = os.path.join(run_output_dir, "model")
    csv_output_dir = os.path.join(run_output_dir, "csv")
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(csv_output_dir, exist_ok=True)

    print(f"学習結果の保存先: {run_output_dir}")
    result = train_for_seed(args.seed, model_output_dir, csv_output_dir)
    seed_summary_path, combined_summary_path = write_training_summaries(
        result, csv_output_dir
    )

    print(
        f"\n学習完了: seed={result['seed']}, "
        f"best_epoch={result['best_epoch']}, "
        f"best_map={result['best_map']:.4f}"
    )
    print(f"seed別学習サマリー: {seed_summary_path}")
    if combined_summary_path is not None:
        print(f"全seed学習サマリー: {combined_summary_path}")
