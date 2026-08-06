from pathlib import Path

from image_calculator import ImageCalc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
calculator = ImageCalc()
result = calculator.reading(
    PROJECT_ROOT / "data" / "processed" / "val" / "JPEGImages" / "image_001.jpg"
)
print("計算結果:", result)
