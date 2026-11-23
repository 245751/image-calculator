"""フォント別に成功・失敗の例を抽出し、正解・予測をPNG/CSV/JSONに保存する。

例:
    uv run python scripts/visualize_predictions.py --characters 1 7 \
        --font Caveat-Bold.ttf --result all --iou-threshold 0.85

--num-imagesは各フォント・各結果の最大枚数。allはsuccess/failureの両方を保存する。
--charactersは注目する正解クラス（指定なしなら全クラス）。描画は常に全クラス。
文字単位の判定なので、同じ画像をsuccess/failureの両方に保存することがある。
これは目視分析用の判定であり、COCO APのマッチング・評価とは別の処理。
"""

import argparse
import csv
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm
from torchvision.ops import box_iou

from evaluate_by_class_and_font import (
    DEFAULT_DATA_DIR,
    LABEL_MAP,
    LABEL_NAMES,
    PROJECT_ROOT,
    EvaluationDataset,
    create_model,
    select_device,
)


GT_COLOR = "#15803d"
PRED_COLOR = "#1d4ed8"
HEADER_HEIGHT = 80
FOOTER_HEIGHT = 40


def select_indices(dataset, font_name, characters, image_ids):
    """画像を移動せず、XMLのフォントと正解クラスで抽出する。"""
    requested_ids = set(image_ids or [])
    unknown_ids = requested_ids - set(dataset.image_ids)
    if unknown_ids:
        raise ValueError(f"splitに存在しない画像ID: {sorted(unknown_ids)}")

    indices = []
    fonts = {}
    for index, image_id in enumerate(dataset.image_ids):
        if requested_ids and image_id not in requested_ids:
            continue
        annotation = ET.parse(dataset.annotation_dir / f"{image_id}.xml").getroot()
        if font_name and annotation.findtext("font") != font_name:
            continue
        names = {obj.findtext("name") for obj in annotation.findall("object")}
        if characters and not names.intersection(characters):
            continue
        indices.append(index)
        fonts[index] = annotation.findtext("font")

    if not indices:
        raise ValueError("条件に合う画像がありません。フォント名や数字を確認してください。")
    return indices, fonts


def draw_tag(draw, text, x, y, color, font, canvas_size):
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0] + 8
    height = bounds[3] - bounds[1] + 6
    x = max(0, min(x, canvas_size[0] - width))
    y = max(0, min(y, canvas_size[1] - height))
    draw.rectangle((x, y, x + width, y + height), fill=color)
    draw.text((x + 4 - bounds[0], y + 3 - bounds[1]), text, fill="white", font=font)


def analyze_predictions(target, prediction, score_threshold, iou_threshold, characters=None):
    """同クラスの成功対応を先に確保し、残りをクラス非依存のIoU順で対応付ける。

    1. score降順で、同クラスかつIoU >= 閾値の未対応GTに割り当てる。
    2. 残りの正解・予測をIoU降順で1対1対応（重なりがある場合のみ）。
    3. 未対応GTはmissed、未対応予測はduplicate/extra_predictionとして記録。

    低スコアを除いたモデル出力に対する目視分析用の分類で、APではない。
    --characters指定時、余分な予測は予測クラスが対象か、対象GTと閾値以上
    重なる場合のみ抽出対象とする。他クラスの誤検出も行としては保存する。
    """
    gt_labels = target["labels"].tolist()
    pred_labels = prediction["labels"].tolist()
    scores = prediction["scores"].tolist()
    gt_boxes = target["boxes"].tolist()
    pred_boxes = prediction["boxes"].tolist()
    ious = box_iou(target["boxes"].float(), prediction["boxes"].float()).tolist()
    # 比較はモデルと同じdtypeで行い、float32の0.3が丸めで除外されるのを防ぐ。
    eligible = torch.where(prediction["scores"] >= score_threshold)[0].tolist()
    available_gt = set(range(len(gt_labels)))
    available_pred = set(eligible)
    assignments = {}
    target_labels = {LABEL_MAP[name] for name in characters} if characters else None

    for p in sorted(eligible, key=lambda p: (-scores[p], p)):
        candidates = [
            g for g in available_gt
            if gt_labels[g] == pred_labels[p] and ious[g][p] >= iou_threshold
        ]
        if candidates:
            g = max(candidates, key=lambda g: (ious[g][p], -g))
            assignments[g] = p
            available_gt.remove(g)
            available_pred.remove(p)

    pairs = sorted(
        ((ious[g][p], scores[p], g, p) for g in available_gt for p in available_pred
         if ious[g][p] > 0),
        key=lambda item: (-item[0], -item[1], item[2], item[3]),
    )
    for _, _, g, p in pairs:
        if g in available_gt and p in available_pred:
            assignments[g] = p
            available_gt.remove(g)
            available_pred.remove(p)

    rows = []
    for g, label in enumerate(gt_labels):
        p = assignments.get(g)
        overlap = ious[g][p] if p is not None else None
        if p is None:
            reason = "missed"
        elif label == pred_labels[p]:
            reason = "success" if overlap >= iou_threshold else "localization_error"
        else:
            reason = "class_error" if overlap >= iou_threshold else "class_and_localization_error"
        rows.append({
            "kind": "ground_truth", "gt_index": g, "pred_index": p,
            "gt_class": LABEL_NAMES[label],
            "pred_class": LABEL_NAMES[pred_labels[p]] if p is not None else None,
            "score": scores[p] if p is not None else None, "iou": overlap,
            "gt_box": gt_boxes[g], "pred_box": pred_boxes[p] if p is not None else None,
            "result": "success" if reason == "success" else "failure", "reason": reason,
            "is_target": target_labels is None or label in target_labels,
        })

    for p in sorted(available_pred):
        nearest = max(range(len(gt_labels)), key=lambda g: ious[g][p]) if gt_labels else None
        overlap = ious[nearest][p] if nearest is not None else None
        duplicate = (
            nearest is not None and overlap >= iou_threshold
            and gt_labels[nearest] == pred_labels[p]
            and nearest in assignments
            and pred_labels[assignments[nearest]] == gt_labels[nearest]
        )
        is_target = (
            target_labels is None or pred_labels[p] in target_labels
            or any(gt_labels[g] in target_labels and ious[g][p] >= iou_threshold
                   for g in range(len(gt_labels)))
        )
        rows.append({
            "kind": "unmatched_prediction", "gt_index": None, "pred_index": p,
            "gt_class": None, "pred_class": LABEL_NAMES[pred_labels[p]],
            "score": scores[p], "iou": overlap,
            "gt_box": None, "pred_box": pred_boxes[p],
            "result": "failure", "reason": "duplicate" if duplicate else "extra_prediction",
            "is_target": is_target,
            "nearest_gt_index": nearest,
        })
    return rows


def draw_overlay(image, target, prediction, image_id, font_name, score_threshold,
                 iou_threshold, result, analysis):
    focus = [row for row in analysis if row["is_target"] and row["result"] == result]
    focus_gt = {row["gt_index"] for row in focus if row["gt_index"] is not None}
    focus_pred = {row["pred_index"] for row in focus if row["pred_index"] is not None}
    # モデルの枠は元画像の座標。300x300への追加変換は不要。
    footer_height = FOOTER_HEIGHT + 22 * len(focus)
    canvas = Image.new("RGB", (image.width, image.height + HEADER_HEIGHT + footer_height), "white")
    canvas.paste(image, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    draw.text((8, 5), f"{image_id} | {font_name} | {result}", fill="black", font=font)
    draw.text((8, 27), "GT: green (thick); Pred: blue (thin); *: selected object", fill=GT_COLOR, font=font)
    draw.text((8, 49), f"score >= {score_threshold:g}; success = same class + IoU >= {iou_threshold:g}",
              fill=PRED_COLOR, font=font)

    for g, (box, label) in enumerate(zip(target["boxes"].tolist(), target["labels"].tolist())):
        x1, y1, x2, y2 = box
        y1 += HEADER_HEIGHT
        y2 += HEADER_HEIGHT
        draw.rectangle((x1, y1, x2, y2), outline=GT_COLOR, width=4)
        mark = "*" if g in focus_gt else ""
        draw_tag(draw, f"G{g}{mark}: {LABEL_NAMES[label]}", x1, y1 - 24, GT_COLOR, font, canvas.size)

    visible_count = 0
    visible = torch.where(prediction["scores"] >= score_threshold)[0].tolist()
    for p in visible:
        box = prediction["boxes"][p].tolist()
        label = prediction["labels"][p].item()
        score = prediction["scores"][p].item()
        visible_count += 1
        x1, y1, x2, y2 = box
        y1 += HEADER_HEIGHT
        y2 += HEADER_HEIGHT
        draw.rectangle((x1, y1, x2, y2), outline=PRED_COLOR, width=2)
        mark = "*" if p in focus_pred else ""
        draw_tag(draw, f"P{p}{mark}: {LABEL_NAMES[label]} {score:.3f}", x1, y2 + 2,
                 PRED_COLOR, font, canvas.size)

    footer_top = HEADER_HEIGHT + image.height + 12
    draw.text((8, footer_top), f"GT: {len(target['labels'])} | displayed predictions: {visible_count}",
              fill="black", font=font)
    for n, row in enumerate(focus):
        gt = f"G{row['gt_index']}={row['gt_class']}" if row["gt_index"] is not None else "extra"
        pred = f"P{row['pred_index']}={row['pred_class']}" if row["pred_index"] is not None else "none"
        score = f"{row['score']:.3f}" if row["score"] is not None else "-"
        overlap = f"{row['iou']:.3f}" if row["iou"] is not None else "-"
        note = "nearest IoU" if row["kind"] == "unmatched_prediction" else "IoU"
        draw.text((8, footer_top + 22 * (n + 1)),
                  f"* {gt} -> {pred} | s={score} {note}={overlap} | {row['reason']}",
                  fill="#9a3412", font=font)
    return canvas


def safe_component(value):
    return re.sub(r"[^\w.=-]+", "_", value).strip(".") or "unknown"


def model_tag(model_path):
    match = re.search(r"(?:^|[_-])seed(\d+)(?=[_.-]|$)", model_path.stem)
    return f"seed{match.group(1)}" if match else safe_component(model_path.stem)


def write_results(output_dir, metadata, records, result):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.json").write_text(
        json.dumps({**metadata, "result": result, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fields = ["image_id", "font", "image_path", "kind", "gt_index", "pred_index",
              "gt_class", "pred_class", "score", "iou", "result", "reason",
              "is_target", "selected", "nearest_gt_index", "gt_box", "pred_box"]
    with (output_dir / "detections.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for row in record["analysis"]:
                writer.writerow({
                    "image_id": record["image_id"], "font": record["font"],
                    "image_path": record["image_path"], **row,
                    "selected": row["is_target"] and row["result"] == result,
                    "gt_box": json.dumps(row["gt_box"]), "pred_box": json.dumps(row["pred_box"]),
                })


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path,
        default=PROJECT_ROOT / "outputs/logs/baseline/model/ssd_calculator_seed42_state_dict.pth",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--image-set", default="train")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/logs/baseline/plots")
    parser.add_argument("--font", help="XMLのフォント名と完全一致（例: Caveat-Bold.ttf）")
    parser.add_argument("--characters", nargs="+", choices=tuple(LABEL_MAP), help="注目する正解クラス。描画は全クラス")
    parser.add_argument("--image-ids", nargs="+", help="拡張子なしの画像IDを指定")
    parser.add_argument("--num-images", type=int, default=30, help="各フォント・各結果の最大保存枚数")
    parser.add_argument("--result", choices=("all", "success", "failure"), default="all")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="成功とするIoUの下限")
    parser.add_argument("--seed", type=int, default=42, help="表示画像の抽出seed（学習seedとは独立）")
    parser.add_argument("--score-threshold", type=float, default=0.3, help="判定・描画する予測のスコア下限")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    if args.num_images < 1:
        parser.error("--num-imagesは1以上にしてください")
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-thresholdは0以上1以下にしてください")
    if not 0 < args.iou_threshold <= 1:
        parser.error("--iou-thresholdは0より大きく1以下にしてください")
    return args


def main():
    args = parse_args()
    dataset = EvaluationDataset(args.data_dir, image_set=args.image_set)
    indices, fonts_by_index = select_indices(dataset, args.font, args.characters, args.image_ids)
    # ランダム順で調べ、条件に合ったものを必要枚数まで採用する。
    # 推論前に枚数を制限しないので、失敗例が後半にあっても抽出できる。
    random.Random(args.seed).shuffle(indices)
    fonts = sorted(set(fonts_by_index.values()))
    results = ("success", "failure") if args.result == "all" else (args.result,)
    groups = {(font, result): [] for font in fonts for result in results}
    counts = {font: 0 for font in fonts}
    condition = f"iou{args.iou_threshold}_score{args.score_threshold}"
    run_root = args.output_dir / model_tag(args.model)
    group_dirs = {
        (font, result): run_root / safe_component(Path(font).stem) / condition / result
        for font, result in groups
    }
    metadata = {
        "model": str(args.model.resolve()), "data_dir": str(args.data_dir.resolve()),
        "image_set": args.image_set, "font": args.font, "characters": args.characters,
        "sampling_seed": args.seed, "num_images_per_font_and_result": args.num_images,
        "score_threshold": args.score_threshold, "iou_threshold": args.iou_threshold,
        "label_names": LABEL_NAMES,
        "matching": "score-ordered same-class matches at IoU threshold, then remaining positive-IoU pairs in descending IoU order",
        "selection": "images containing at least one target object with the requested result; target extra predictions also count as failures",
        "scope": "selected examples only; not full-validation metrics; raw predictions are after model postprocessing",
    }
    device = select_device() if args.device == "auto" else torch.device(args.device)
    model = create_model(args.model, device)
    print(f"device={device}, candidates={len(indices)}, output={run_root}")

    with torch.inference_mode():
        for index in tqdm(indices, desc="Finding success/failure examples"):
            font_name = fonts_by_index[index]
            if all(len(groups[font_name, result]) >= args.num_images for result in results):
                continue
            tensor, target, font_name, image_id = dataset[index]
            prediction = {
                key: value.detach().cpu()
                for key, value in model([tensor.to(device)])[0].items()
            }
            counts[font_name] += 1
            analysis = analyze_predictions(
                target, prediction, args.score_threshold, args.iou_threshold, args.characters
            )
            qualifying = {row["result"] for row in analysis if row["is_target"]}
            save_results = [result for result in results if result in qualifying
                            and len(groups[font_name, result]) < args.num_images]
            if not save_results:
                continue
            with Image.open(dataset.image_dir / f"{image_id}.jpg") as source:
                image = source.convert("RGB")
            record = {
                "image_index": index, "image_id": image_id, "font": font_name,
                "image_size": list(image.size),
                "gt_boxes": target["boxes"].tolist(), "gt_labels": target["labels"].tolist(),
                "pred_boxes": prediction["boxes"].tolist(),
                "pred_labels": prediction["labels"].tolist(),
                "pred_scores": prediction["scores"].tolist(), "analysis": analysis,
            }
            for result in save_results:
                output_dir = group_dirs[font_name, result]
                output_dir.mkdir(parents=True, exist_ok=True)
                image_path = output_dir / f"{image_id}.png"
                draw_overlay(image, target, prediction, image_id, font_name,
                             args.score_threshold, args.iou_threshold, result, analysis).save(image_path)
                groups[font_name, result].append({**record, "image_path": str(image_path.resolve())})
            if all(len(records) >= args.num_images for records in groups.values()):
                break

    for (font_name, result), records in groups.items():
        output_dir = group_dirs[font_name, result]
        write_results(output_dir, {**metadata, "font": font_name,
                                  "num_inferred_images": counts[font_name]}, records, result)
        print(f"{font_name} / {result}: {len(records)}/{args.num_images} images -> {output_dir}")
    print("保存枚数は各フォント・各結果の上限です。同一画像が両方に入る場合があります。")
    print("CSV/JSONは保存した画像のみの記録で、validation全体の誤り率ではありません。")


if __name__ == "__main__":
    main()
