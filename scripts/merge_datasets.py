import shutil
import glob
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def merge_voc_datasets(source_dirs, output_dir, rename_pattern="merged_{:05d}"):
    """
    複数のVOCデータセットを一つにまとめる関数
    
    Args:
        source_dirs: マージ元のデータセットディレクトリのリスト
        output_dir: 出力先ディレクトリ
        rename_pattern: 新しいファイル名のパターン（例: "merged_{:05d}"）
    """
    
    # 出力ディレクトリの準備
    output_img_dir = os.path.join(output_dir, "JPEGImages")
    output_ann_dir = os.path.join(output_dir, "Annotations")
    output_sets_dir = os.path.join(output_dir, "ImageSets", "Main")
    
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_ann_dir, exist_ok=True)
    os.makedirs(output_sets_dir, exist_ok=True)
    
    merged_image_ids = []
    file_counter = 0
    
    # 各ソースディレクトリからファイルをコピー
    for i, source_dir in enumerate(source_dirs):
        print(f"📁 処理中: {source_dir}")
        
        source_img_dir = os.path.join(source_dir, "JPEGImages")
        source_ann_dir = os.path.join(source_dir, "Annotations")
        
        if not os.path.exists(source_img_dir) or not os.path.exists(source_ann_dir):
            print(f"⚠️ スキップ: {source_dir} (必要なディレクトリが見つかりません)")
            continue
        
        # 画像ファイルとアノテーションファイルを取得
        img_files = glob.glob(os.path.join(source_img_dir, "*.jpg"))
        
        for img_file in img_files:
            # ファイル名から拡張子を除いた部分を取得
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            ann_file = os.path.join(source_ann_dir, f"{base_name}.xml")
            
            if not os.path.exists(ann_file):
                print(f"⚠️ スキップ: {base_name} (アノテーションファイルが見つかりません)")
                continue
            
            # 新しいファイル名を生成
            new_name = rename_pattern.format(file_counter)
            new_img_path = os.path.join(output_img_dir, f"{new_name}.jpg")
            new_ann_path = os.path.join(output_ann_dir, f"{new_name}.xml")
            
            # ファイルをコピー
            shutil.copy2(img_file, new_img_path)
            
            # アノテーションファイルをコピーして内容を更新
            update_annotation_file(ann_file, new_ann_path, new_name)
            
            merged_image_ids.append(new_name)
            file_counter += 1
    
    # train.txtファイルを作成
    train_file = os.path.join(output_sets_dir, "train.txt")
    with open(train_file, "w") as f:
        for image_id in merged_image_ids:
            f.write(image_id + "\n")
    
    print(f"✅ マージ完了!")
    print(f"📊 総ファイル数: {len(merged_image_ids)}")
    print(f"💾 出力先: {output_dir}")
    
    # 各ソースの統計を表示
    print("\n📈 ソース別統計:")
    current_count = 0
    for i, source_dir in enumerate(source_dirs):
        source_img_dir = os.path.join(source_dir, "JPEGImages")
        if os.path.exists(source_img_dir):
            count = len(glob.glob(os.path.join(source_img_dir, "*.jpg")))
            print(f"  {source_dir}: {count}ファイル")
            current_count += count

def update_annotation_file(source_ann_path, dest_ann_path, new_image_id):
    """アノテーションファイルの画像IDを更新してコピー"""
    tree = ET.parse(source_ann_path)
    root = tree.getroot()
    
    # filenameを更新
    filename_elem = root.find("filename")
    if filename_elem is not None:
        filename_elem.text = f"{new_image_id}.jpg"
    
    # 保存
    tree.write(dest_ann_path, encoding="utf-8", xml_declaration=True)

def merge_datasets_advanced(dataset_configs, output_dir):
    """
    高度なデータセットマージ機能（比率指定可能）
    
    Args:
        dataset_configs: [{"path": "dataset1", "ratio": 0.5}, {"path": "dataset2", "ratio": 0.3}] 形式
        output_dir: 出力先ディレクトリ
    """
    
    # 各データセットのファイル数を調査
    total_files = 0
    dataset_info = []
    
    for config in dataset_configs:
        dataset_path = config["path"]
        img_dir = os.path.join(dataset_path, "JPEGImages")
        
        if os.path.exists(img_dir):
            file_count = len(glob.glob(os.path.join(img_dir, "*.jpg")))
            dataset_info.append({
                "path": dataset_path,
                "count": file_count,
                "ratio": config.get("ratio", 1.0)
            })
            total_files += file_count
    
    print(f"📊 データセット情報:")
    for info in dataset_info:
        print(f"  {info['path']}: {info['count']}ファイル (比率: {info['ratio']})")
    
    # 出力ディレクトリの準備
    output_img_dir = os.path.join(output_dir, "JPEGImages")
    output_ann_dir = os.path.join(output_dir, "Annotations")
    output_sets_dir = os.path.join(output_dir, "ImageSets", "Main")
    
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_ann_dir, exist_ok=True)
    os.makedirs(output_sets_dir, exist_ok=True)
    
    merged_image_ids = []
    file_counter = 0
    
    # 各データセットから指定比率でファイルを選択
    for info in dataset_info:
        dataset_path = info["path"]
        target_count = int(info["count"] * info["ratio"])
        
        print(f"📁 処理中: {dataset_path} ({target_count}/{info['count']}ファイル)")
        
        img_dir = os.path.join(dataset_path, "JPEGImages")
        ann_dir = os.path.join(dataset_path, "Annotations")
        
        # ファイルリストを取得してシャッフル
        img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
        random.shuffle(img_files)
        
        # 指定数だけ処理
        for img_file in img_files[:target_count]:
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            ann_file = os.path.join(ann_dir, f"{base_name}.xml")
            
            if not os.path.exists(ann_file):
                continue
            
            # 新しいファイル名を生成
            new_name = f"merged_{file_counter:05d}"
            new_img_path = os.path.join(output_img_dir, f"{new_name}.jpg")
            new_ann_path = os.path.join(output_ann_dir, f"{new_name}.xml")
            
            # ファイルをコピー
            shutil.copy2(img_file, new_img_path)
            update_annotation_file(ann_file, new_ann_path, new_name)
            
            merged_image_ids.append(new_name)
            file_counter += 1
    
    # train.txtファイルを作成
    train_file = os.path.join(output_sets_dir, "train.txt")
    with open(train_file, "w") as f:
        for image_id in merged_image_ids:
            f.write(image_id + "\n")
    
    print(f"✅ 高度マージ完了!")
    print(f"📊 総ファイル数: {len(merged_image_ids)}")
    print(f"💾 出力先: {output_dir}")

# 使用例1: 基本的なマージ
def example_basic_merge():
    """基本的なマージの例"""
    source_datasets = [
        "dataset1",
        "dataset2", 
        "dataset3"
    ]
    
    merge_voc_datasets(
        source_dirs=source_datasets,
        output_dir="merged_dataset",
        rename_pattern="combined_{:05d}"
    )

# 使用例2: 比率指定マージ
def example_ratio_merge():
    """比率指定マージの例"""
    dataset_configs = [
        {"path": "dataset_large_font", "ratio": 0.7},    # 70%使用
        {"path": "dataset_small_font", "ratio": 0.5},    # 50%使用
        {"path": "dataset_random_layout", "ratio": 1.0}  # 100%使用
    ]
    
    merge_datasets_advanced(
        dataset_configs=dataset_configs,
        output_dir="balanced_dataset"
    )

# 使用例3: あなたの現在のデータセットをマージ
def merge_your_datasets():
    """あなたのデータセットをマージする例"""
    
    # 複数のデータセットをリストアップ
    datasets_to_merge = [
        "a",  # 現在のデータセット
        # 他のデータセットディレクトリがあれば追加
    ]
    
    # 存在するディレクトリのみをフィルタ
    existing_datasets = [d for d in datasets_to_merge if os.path.exists(d)]
    
    if len(existing_datasets) > 1:
        merge_voc_datasets(
            source_dirs=existing_datasets,
            output_dir="final_merged_dataset",
            rename_pattern="final_{:05d}"
        )
        print("🎉 データセットのマージが完了しました！")
    else:
        print("⚠️ マージ可能なデータセットが1つ以下です")

#使用例
#基本的なマージ
# source_datasets = ["dataset1", "dataset2", "dataset3"]
# merge_voc_datasets(source_datasets, "merged_dataset")

# 比率指定マージ
# configs = [
#     {"path": "large_dataset", "ratio": 0.5},  # 50%使用
#     {"path": "small_dataset", "ratio": 1.0}   # 100%使用
# ]
# merge_datasets_advanced(configs, "balanced_dataset")

# 実行
if __name__ == "__main__":
    mini_dir = PROJECT_ROOT / "data" / "generated" / "mini"
    source_datasets = [
        mini_dir / "size_layout_random_mini",
        mini_dir / "size_random_mini",
    ]
    merge_voc_datasets(
        source_dirs=source_datasets,
        output_dir=PROJECT_ROOT / "data" / "generated" / "merged_mini",
    )
