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
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
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

def evaluate_detector(model, data_loader):
    model.eval()
    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.inference_mode():
        for images, targets in tqdm(data_loader, desc="Validation", leave=False):
            predictions = model([image.to(device) for image in images])

            for prediction, target in zip(predictions, targets):
                keep = prediction["scores"].detach().cpu() >= SCORE_THRESHOLD
                pred_boxes = prediction["boxes"].detach().cpu()[keep]
                pred_labels = prediction["labels"].detach().cpu()[keep]
                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()
                matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)

                for pred_box, pred_label in zip(pred_boxes, pred_labels):
                    candidate_indices = torch.where(
                        (gt_labels == pred_label) & (~matched_gt)
                    )[0]
                    if len(candidate_indices) == 0:
                        false_positive += 1
                        continue

                    ious = box_iou(
                        pred_box.unsqueeze(0), gt_boxes[candidate_indices]
                    ).squeeze(0)
                    best_iou, best_position = ious.max(dim=0)
                    if best_iou.item() >= IOU_THRESHOLD:
                        true_positive += 1
                        matched_gt[candidate_indices[best_position]] = True
                    else:
                        false_positive += 1

                false_negative += (~matched_gt).sum().item()

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1

print(
    f"device: {device} / train images: {len(train_dataset)} "
    f"/ val images: {len(val_dataset)}"
)

# 学習開始時にCSVを新規作成し、各エポック終了時に結果を追記する
with open(CSV_PATH, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["epoch", "train_loss", "precision", "recall", "f1"])

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
    precision, recall, f1 = evaluate_detector(model, val_loader)
    print(
        f"Epoch {epoch + 1}: loss={mean_loss:.4f}, "
        f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}"
    )

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch + 1, mean_loss, precision, recall, f1])

torch.save(model.state_dict(), MODEL_PATH)
print(f"学習済みモデルを保存しました: {MODEL_PATH}")
print(f"学習結果を保存しました: {CSV_PATH}")