import torch
import torchvision
from torchvision import transforms
from PIL import Image
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ssd_calculator_model.pth"

class ImageCalc:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.device = torch.device('mps')
        torch.serialization.add_safe_globals([torchvision.models.detection.ssd.SSD])
        self.loaded_model = torch.load(model_path, map_location='cpu', weights_only=False)
        self.loaded_model.to(self.device)
        self.loaded_model.eval()
    def calculation(self,equation):
        # "="がリストにあれば取り除く
        if equation[-1] == '=':
            equation = equation[:-1]
        
        # リストを連結して文字列に
        expression = ''.join(equation)
        
        try:
            # evalで式を評価
            result = eval(expression)
            return result
        except Exception as e:
            print("エラー:", e)
            return None
    def reading(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
        ])

        image_path = image_path
        orig_image = Image.open(image_path).convert("RGB")
        input_tensor = transform(orig_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.loaded_model(input_tensor)

        output=outputs[0]

        label_map = {
            1: '0', 2: '1', 3: '2', 4: '3', 5: '4',
            6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
            11: '+', 12: '-', 13: '*', 14: '/', 15: '='
        }

        # スコアしきい値（信頼度）を設定
        score_threshold = 0.8

        boxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]
        orig_w, orig_h = orig_image.size
        x_scale = orig_w / 300
        y_scale = orig_h / 300
        equation=[]
        for box, label, score in zip(boxes, labels, scores):
            if score >= score_threshold:
                x1, y1, x2, y2 = box.tolist()
                x1 *= x_scale
                x2 *= x_scale
                y1 *= y_scale
                y2 *= y_scale
                label_name = label_map[int(label)]
                text = f"{label_name} {score:.2f}"
                equation.append([x1,label_name])
        sorted_data = sorted(equation, key=lambda x: x[0])
        equation=[]
        for item in sorted_data:
            equation.append(item[1])

        return self.calculation(equation)
