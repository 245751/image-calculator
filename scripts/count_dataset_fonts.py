"""Annotations/*.xmlからフォント別画像数・数記号別出現数をCSVに保存する。"""

from collections import Counter
import csv
from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 調べたいデータセットのルート（train、val、testがある場所）を指定する。
# 別の場所を調べる場合: DATASET_PATH = Path("/path/to/dataset")
DATASET_PATH = PROJECT_ROOT / "data" / "generated" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "report" / "baseline" / "analysis"
SPLITS = ("train", "val", "test")


def count_dataset(dataset_path: Path) -> tuple[dict[str, Counter], dict[str, Counter]]:
    """フォントはXML単位、数記号はobject単位で数える。Noneは未設定。"""
    font_counts = {}
    symbol_counts = {}
    for split in SPLITS:
        annotation_dir = dataset_path / split / "Annotations"
        if not annotation_dir.is_dir():
            raise FileNotFoundError(f"Annotationsが見つかりません: {annotation_dir}")

        split_counts = Counter()
        split_symbols = Counter()
        for xml_path in sorted(annotation_dir.glob("*.xml")):
            try:
                root = ET.parse(xml_path).getroot()
            except ET.ParseError as exc:
                raise ValueError(f"XMLを解析できません: {xml_path}: {exc}") from exc
            font = (root.findtext("font") or "").strip()
            split_counts[font or None] += 1
            for obj in root.findall("object"):
                symbol = (obj.findtext("name") or "").strip()
                split_symbols[symbol or None] += 1
        font_counts[split] = split_counts
        symbol_counts[split] = split_symbols
    return font_counts, symbol_counts


def write_counts_csv(path: Path, counts: dict[str, Counter], kind: str) -> None:
    """列はtrain・val・test。フォント行は最初に使われるsplitの順に並べる。"""
    labels = {label for counter in counts.values() for label in counter if label is not None}
    if kind == "font":
        labels = sorted(labels, key=lambda label: (
            next(i for i, split in enumerate(SPLITS) if counts[split][label]), label
        ))
    else:
        labels = sorted(labels)

    # Excelでも日本語が文字化けしにくいUTF-8 BOM付きで保存する。
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([kind, *SPLITS])
        for label in labels:
            display_label = label.removesuffix(".ttf") if kind == "font" else label
            writer.writerow([display_label, *(counts[split][label] for split in SPLITS)])
        if any(counts[split][None] for split in SPLITS):
            writer.writerow([f"({kind}未設定)", *(counts[split][None] for split in SPLITS)])
        total_label = "総画像数" if kind == "font" else "総数記号数"
        writer.writerow([total_label, *(sum(counts[split].values()) for split in SPLITS)])
        if kind == "font":
            writer.writerow(["フォント種類数（未設定を除く）", *(
                sum(label is not None for label in counts[split]) for split in SPLITS
            )])


def export_dataset_counts(dataset_path: Path, output_dir: Path) -> None:
    """対象XMLの集計後に、2つのCSVを作成する。同名ファイルは上書きする。"""
    font_counts, symbol_counts = count_dataset(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for kind, counts in (("font", font_counts), ("symbol", symbol_counts)):
        path = output_dir / f"{kind}_counts.csv"
        write_counts_csv(path, counts, kind)
        print(f"保存しました: {path.resolve()}")


if __name__ == "__main__":
    export_dataset_counts(DATASET_PATH, OUTPUT_DIR)
