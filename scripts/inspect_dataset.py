import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def visualize_voc_sample(dataset_root, image_id):
    """
    VOC形式データセットの1つのサンプルを可視化する
    
    Args:
        dataset_root: データセットのルートディレクトリ
        image_id: 画像ID（拡張子なし）
    """
    # パスを構築
    img_path = os.path.join(dataset_root, "JPEGImages", f"{image_id}.jpg")
    xml_path = os.path.join(dataset_root, "Annotations", f"{image_id}.xml")
    
    # ファイルの存在確認
    if not os.path.exists(img_path):
        print(f"❌ 画像ファイルが見つかりません: {img_path}")
        return
    
    if not os.path.exists(xml_path):
        print(f"❌ アノテーションファイルが見つかりません: {xml_path}")
        return
    
    # 画像を読み込み
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # XMLファイルを解析
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # フォントを設定
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    print(f"📷 画像: {image_id}.jpg")
    print(f"📏 画像サイズ: {image.size}")
    print(f"📦 検出オブジェクト:")
    
    # 各オブジェクトの境界ボックスを描画
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    
    for i, obj in enumerate(root.findall("object")):
        # ラベル名を取得
        name = obj.find("name").text
        
        # 境界ボックスを取得
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        
        # 色を選択
        color = colors[i % len(colors)]
        
        # 境界ボックスを描画
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        
        # ラベルを描画
        text = f"{name}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # テキスト背景
        draw.rectangle(
            [xmin, ymin - text_height - 5, xmin + text_width + 10, ymin], 
            fill=color
        )
        draw.text((xmin + 5, ymin - text_height - 2), text, fill="white", font=font)
        
        print(f"  - {name}: ({xmin}, {ymin}) → ({xmax}, {ymax}) [width: {xmax-xmin}, height: {ymax-ymin}]")
    
    # 結果を表示
    plt.figure(figsize=(12, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"VOC Dataset Visualization: {image_id}")
    plt.tight_layout()
    plt.show()
    
    return image

def browse_voc_dataset(dataset_root, max_samples=10):
    """
    VOC形式データセットの複数サンプルを一覧表示
    
    Args:
        dataset_root: データセットのルートディレクトリ
        max_samples: 表示する最大サンプル数
    """
    # train.txtから画像IDリストを取得
    train_file = os.path.join(dataset_root, "ImageSets", "Main", "train.txt")
    
    if not os.path.exists(train_file):
        print(f"❌ train.txtが見つかりません: {train_file}")
        return
    
    with open(train_file) as f:
        image_ids = [line.strip() for line in f.readlines()]
    
    print(f"📁 データセット: {dataset_root}")
    print(f"📊 総サンプル数: {len(image_ids)}")
    print(f"🔍 表示サンプル数: {min(max_samples, len(image_ids))}")
    print("-" * 50)
    
    # 指定された数のサンプルを表示
    for i, image_id in enumerate(image_ids[:max_samples]):
        print(f"\n[{i+1}/{min(max_samples, len(image_ids))}]")
        visualize_voc_sample(dataset_root, image_id)

def get_dataset_statistics(dataset_root):
    """
    データセットの統計情報を表示
    
    Args:
        dataset_root: データセットのルートディレクトリ
    """
    train_file = os.path.join(dataset_root, "ImageSets", "Main", "train.txt")
    
    if not os.path.exists(train_file):
        print(f"❌ train.txtが見つかりません: {train_file}")
        return
    
    with open(train_file) as f:
        image_ids = [line.strip() for line in f.readlines()]
    
    label_count = {}
    total_objects = 0
    
    for image_id in image_ids:
        xml_path = os.path.join(dataset_root, "Annotations", f"{image_id}.xml")
        
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall("object"):
                name = obj.find("name").text
                label_count[name] = label_count.get(name, 0) + 1
                total_objects += 1
    
    print(f"📊 データセット統計: {dataset_root}")
    print(f"📷 総画像数: {len(image_ids)}")
    print(f"🏷️  総オブジェクト数: {total_objects}")
    print(f"📋 ラベル別統計:")
    
    for label, count in sorted(label_count.items()):
        percentage = (count / total_objects) * 100
        print(f"  - {label}: {count}個 ({percentage:.1f}%)")

# 使用例
if __name__ == "__main__":
    dataset_path = PROJECT_ROOT / "data" / "processed" / "val"
    
    # # 統計情報を表示
    # print("=" * 60)
    # print("データセット統計情報")
    # print("=" * 60)
    # get_dataset_statistics(dataset_path)
    
    # print("\n" + "=" * 60)
    # print("サンプル可視化")
    # print("=" * 60)
    
    # 複数サンプルを表示
    # browse_voc_dataset(dataset_path, max_samples=5)
    
    # 特定のサンプルを表示
    visualize_voc_sample(dataset_path, "image_005")
