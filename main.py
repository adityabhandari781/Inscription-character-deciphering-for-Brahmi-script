import json
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image, ImageDraw
import regex as re

image_path = "data/imgs/engr/1.jpg"
coords_xy = [310, 56, 336, 98] # missing char: 𑀦𑁆



# CODE TO PUT A WHITE BOX ON A CHARACTER (FOR TESTING TRANSFORMER)
def mask_character(image_path, coords_xy):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(coords_xy, fill="green")
    masked_path = "outputs/masked.jpg"
    image.save(masked_path)
    return masked_path



# DETECTION

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_det = YOLO("models/yolo_det_new.pt")

def get_boxes(image_path):
    output_json = "outputs/detection_results.json"
    results = model_det(image_path, conf=0.25)
    detections = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()

        for x1, y1, x2, y2 in boxes:
            detections.append({
                "bbox": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                ]
            })
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2)

    # make image with bounding boxes on it
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    img.save("outputs/detections_visualized.jpg")



# CLASSIFICATION

ckpt = torch.load("models/mobilenet_brahmi_classifier_new.pt", map_location=DEVICE)
classes = ckpt["classes"]
model_clf = mobilenet_v2(pretrained=False)
model_clf.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model_clf.last_channel, len(classes))
)
model_clf.load_state_dict(ckpt["model_state"])
model_clf.to(DEVICE)
model_clf.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def decode_char_folder(name: str) -> str:
    chars = []
    for part in name.split("_"):
        if not part.startswith("U+"):
            raise ValueError(f"Invalid format: {part}")
        codepoint = int(part[2:], 16)
        chars.append(chr(codepoint))
    return "".join(chars)



@torch.no_grad()
def classify_detections(image_path):
    output_json_path = "outputs/classification_results.json"
    image = Image.open(image_path).convert("RGB")

    with open("outputs/detection_results.json", "r") as f:
        detections = json.load(f)

    results = []

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])

        crop = image.crop((x1, y1, x2, y2))
        x = transform(crop).unsqueeze(0).to(DEVICE)

        logits = model_clf(x)
        pred_idx = logits.argmax(dim=1).item()
        pred_class = classes[pred_idx]

        results.append({
            "char": decode_char_folder(pred_class),
            "bbox": det["bbox"]
        })

    results.append({
        "char": "_",
        "bbox": coords_xy
    })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)



# RECONSTRUCTION

class TextReconstructor:
    def __init__(self, json_data):
        self.data = json_data

    def _get_center_y(self, bbox):
        # bbox format assumed: [x_min, y_min, x_max, y_max]
        _, y_min, _, y_max = bbox
        return (y_min + y_max) / 2

    def _boxes_overlap_vertically(self, bbox1, bbox2, threshold=0.5):
        """
        Check if two boxes overlap vertically by a certain percentage.
        This handles variable heights (e.g., a tall char next to a short one).
        """
        y1_min, y1_max = bbox1[1], bbox1[3]
        y2_min, y2_max = bbox2[1], bbox2[3]

        # Calculate intersection height
        intersection_min = max(y1_min, y2_min)
        intersection_max = min(y1_max, y2_max)
        intersection_h = max(0, intersection_max - intersection_min)

        # Height of the smaller box (to be lenient with short chars)
        h1 = y1_max - y1_min
        h2 = y2_max - y2_min
        min_height = min(h1, h2)
        
        if min_height == 0: return False

        # If intersection covers > 50% of the smaller box's height, they are on the same line
        return (intersection_h / min_height) > threshold

    def reconstruct(self):
        # 1. Sort all items by Y-center initially to process top-to-bottom
        sorted_data = sorted(self.data, key=lambda x: self._get_center_y(x['bbox']))

        lines = []  # Will store lists of items: [ [item1, item2], [item3, item4] ]

        for item in sorted_data:
            placed = False
            
            # Try to fit this item into an existing line
            for line in lines:
                # We check overlap with the LAST item added to that line
                # (Or you could check the average Y of the line)
                reference_item = line[-1]
                
                if self._boxes_overlap_vertically(item['bbox'], reference_item['bbox']):
                    line.append(item)
                    placed = True
                    break
            
            # If it didn't fit in any existing line, start a new one
            if not placed:
                lines.append([item])

        # 2. Sort lines vertically (just to be safe)
        # We use the average Y-center of the line to sort the lines themselves
        def get_line_avg_y(line_items):
            return sum(self._get_center_y(i['bbox']) for i in line_items) / len(line_items)
        
        lines.sort(key=get_line_avg_y)

        # 3. Sort items WITHIN each line horizontally (by x_min)
        final_output_lines = []
        for line in lines:
            # Sort by x_min (bbox[0])
            line.sort(key=lambda x: x['bbox'][0])
            
            # Join the glyphs
            line_text = "".join(item['char'] for item in line)
            final_output_lines.append(line_text)

        # 4. Join all lines with a newline character
        return "\n".join(final_output_lines)

def get_text(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_json = json.load(f)
    return TextReconstructor(raw_json).reconstruct().replace('\n', '')



# TRANSFORMER

MODEL_PATH = 'models/transformer.pth'
VOCAB_PATH = 'vocab.json' # Loading from JSON now
CONTEXT_SIZE = 5
EMBED_DIM = 64
N_HEADS = 4
N_LAYERS = 2
WINDOW_SIZE = (CONTEXT_SIZE * 2) + 1

class BrahmiTokenizer:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.char2idx = json.load(f)
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        self.unk_idx = self.char2idx.get('<unk>', 2)

    def get_graphemes(self, text):
        return re.findall(r'\X', text)

    def encode(self, list_of_graphemes):
        return [self.char2idx.get(x, self.unk_idx) for x in list_of_graphemes]

class CharTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_encoder = nn.Parameter(torch.randn(1, WINDOW_SIZE, EMBED_DIM))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=N_HEADS, batch_first=True), 
            num_layers=N_LAYERS
        )
        self.fc_out = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder
        return self.fc_out(self.transformer(x)[:, CONTEXT_SIZE, :])

tokenizer = BrahmiTokenizer(VOCAB_PATH)
model = CharTransformer(tokenizer.vocab_size).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def predict_char(char_str):
    g = tokenizer.get_graphemes(char_str)
    if '_' not in g:
        return "No mask"
    
    idx = g.index('_')
    left = g[max(0, idx - CONTEXT_SIZE) : idx]
    right = g[idx + 1 : idx + 1 + CONTEXT_SIZE]

    # Pad 0s manually
    ids = ([0]*(CONTEXT_SIZE-len(left))) + tokenizer.encode(left) + \
          [1] + \
          tokenizer.encode(right) + ([0]*(CONTEXT_SIZE-len(right)))

    with torch.no_grad():
        logits = model(torch.tensor([ids]).to(DEVICE))
        pred = torch.argmax(logits, dim=1).item()
        
    return tokenizer.idx2char.get(pred, '')



# classify_detections(
#     image_path="data/imgs/0.jpg",
#     bbox_json_path="page_res_det.json"
# )



if __name__ == "__main__":
    masked_image_path = mask_character(image_path, coords_xy)
    get_boxes(masked_image_path)
    classify_detections(masked_image_path)
    reconstructed_text = get_text("outputs/classification_results.json")
    # Find the position of the underscore (missing character)
    missing_char_idx = reconstructed_text.index('_')
    before_char = reconstructed_text[:missing_char_idx]
    after_char = reconstructed_text[missing_char_idx + 1:]
    
    predicted_char = predict_char(reconstructed_text)
    
    print("Reconstructed Text:")
    print(before_char)
    print(predicted_char)
    print(after_char)