from PIL import Image
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 画像を読み込む
image = Image.open(
    PROJECT_ROOT / "data" / "processed" / "val" / "JPEGImages" / "image_001.jpg"
)

# 新しいサイズを指定 (幅, 高さ)
new_width = 100
new_height = 25

# 画像をリサイズ
resized_image = image.resize((new_width, new_height))

# リサイズした画像を表示
resized_image.show()

# リサイズした画像を保存する場合
# resized_image.save("output.jpg")
