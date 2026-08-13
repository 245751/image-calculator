import os
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

NUM_CLASSES = 16  # 背景(0) + 数字(10) + 記号(5)
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAP_IOU_THRESHOLDS = tuple(i / 100 for i in range(50, 100, 5))
MODEL_PATH = "outputs/checkpoints/ssd_calculator_state_dict.pth"
CSV_PATH = "outputs/logs/training_results.csv"

class CustomVOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set='train', transforms=None, label_map=None):
        self.root = root
        self.image_dir = os.path.join(root, "JPEGImages")
        self.annotation_dir = os.path.join(root, "Annotations")
        self.transforms = transforms
        self.label_map = label_map or {
            '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6,
            '6': 7, '7': 8, '8': 9, '9': 10,
            '+': 11, '-': 12, '*': 13, '/': 14, '=': 15
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

        return img, target

transform = transforms.ToTensor()
train_dataset = CustomVOCDataset(
    root="data/processed/train",
    image_set="train",
    transforms=transform,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=lambda batch: tuple(zip(*batch)),
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

# COCOで事前学習済みのSSD300-VGG16を使い、分類ヘッドだけ16クラス用に交換する
model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
model.head.classification_head = SSDClassificationHead(
    retrieve_out_channels(model.backbone, (300, 300)),
    model.anchor_generator.num_anchors_per_location(),
    NUM_CLASSES,
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=5e-4,
)

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
        for images, targets in tqdm(data_loader, desc="Validation mAP", leave=False):
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
            for images, targets in tqdm(
                data_loader, desc="Validation loss", leave=False
            ):
                images = [image.to(device) for image in images]
                targets = [
                    {key: value.to(device) for key, value in target.items()}
                    for target in targets
                ]

                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                running_loss += loss.item()
                num_batches += 1
    finally:
        model.train(was_training)

    return running_loss / max(num_batches, 1)

print(
    f"device: {device} / train images: {len(train_dataset)} "
    f"/ val images: {len(val_dataset)}"
)

# 学習開始時にCSVを新規作成し、各エポック終了時に結果を追記する
with open(CSV_PATH, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["epoch", "train_loss", "val_loss", "map"])

best_map = float("-inf")
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    for images, targets in progress:
        images = [image.to(device) for image in images]
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    mean_loss = running_loss / len(train_loader)
    val_loss = evaluate_loss(model, val_loader)
    mean_average_precision = evaluate_map(model, val_loader)
    print(
        f"Epoch {epoch + 1}: train_loss={mean_loss:.4f}, "
        f"val_loss={val_loss:.4f}, "
        f"mAP@0.50:0.95={mean_average_precision:.4f}"
    )

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch + 1, mean_loss, val_loss, mean_average_precision])

    if mean_average_precision > best_map:
        best_map = mean_average_precision
        best_epoch = epoch + 1
        torch.save(model.state_dict(), MODEL_PATH)
        print(
            f"最高mAPを更新したためモデルを保存しました: "
            f"epoch={best_epoch}, mAP={best_map:.4f}, path={MODEL_PATH}"
        )

print(
    f"最高mAPモデル: epoch={best_epoch}, "
    f"mAP@0.50:0.95={best_map:.4f}, path={MODEL_PATH}"
)
print(f"学習結果を保存しました: {CSV_PATH}")
