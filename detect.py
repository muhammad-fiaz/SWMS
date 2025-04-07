import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import glob
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file, load_file
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QTextEdit
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from PIL import Image
from transformers import YolosImageProcessor, YolosForObjectDetection


# ===== Dataset + Model Setup =====
class RecyclingDataset(Dataset):
    def __init__(self, df):
        self.labels = df['label'].astype(str).tolist()
        self.data = df.drop(columns=['label']).values.astype(float)
        self.label2idx = {label: idx for idx, label in enumerate(set(self.labels))}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        self.encoded_labels = [self.label2idx[lbl] for lbl in self.labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded_labels[idx]), torch.tensor(self.data[idx], dtype=torch.float)


class ComponentPredictor(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, 16)
        self.fc = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # plastic, metal, glass
        )

    def forward(self, x):
        x = self.embedding(x)
        return self.fc(x)


def load_and_train_model():
    print("📥 Loading datasets...")
    csv_files = glob.glob("Datasets/*.csv")
    if not csv_files:
        print("❌ No datasets found.")
        return None, None

    df_all = pd.concat([pd.read_csv(f) for f in csv_files])
    dataset = RecyclingDataset(df_all)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = ComponentPredictor(len(dataset.label2idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    print("🔁 Training started...")
    for epoch in range(100):
        for labels, targets in dataloader:
            preds = model(labels)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    tensor_dict = {k: v for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
    save_file(tensor_dict, "model.safetensors")
    print("✅ Training complete. Model saved as model.safetensors.")
    return model, {v: k for k, v in dataset.label2idx.items()}


def load_classifier():
    print("📦 Loading model and label mappings...")
    csv_files = glob.glob("Datasets/*.csv")
    if not csv_files:
        print("❌ No datasets available to infer label2idx.")
        sys.exit()

    df_all = pd.concat([pd.read_csv(f) for f in csv_files])
    dataset = RecyclingDataset(df_all)
    label2idx = dataset.label2idx

    model = ComponentPredictor(len(label2idx))

    if not os.path.exists("model.safetensors"):
        print("❌ model.safetensors not found. Please run with --train to generate it.")
        sys.exit()

    state = load_file("model.safetensors")
    model.load_state_dict(state)
    model.eval()
    print("✅ Model loaded successfully.")
    return model, label2idx, {v: k for k, v in label2idx.items()}


# ===== YOLO Analysis with multiple item breakdown =====
def analyze_with_yolo(image_path, model, label2idx):
    print(f"🔍 Analyzing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = yolo_model(**inputs)
    logits = outputs.logits
    probs = logits.softmax(-1)[0, :, :-1]
    max_probs, labels = probs.max(-1)
    threshold = 0.5
    detected = []
    for score, label in zip(max_probs, labels):
        if score > threshold:
            name = yolo_model.config.id2label[label.item()]
            print(f"🧠 Detected: {name} (score: {score.item():.2f})")
            detected.append(name)

    report = ""
    count = 1
    for item in detected:
        report += f"\nDetected Items {count} in {os.path.basename(image_path)}:\n"
        report += f"- {item}\n"
        count += 1

        if item in label2idx:
            input_id = torch.tensor([label2idx[item]])
            with torch.no_grad():
                pred = model(input_id).squeeze()
            plastic, metal, glass = pred.tolist()
            total = plastic + metal + glass
            plastic_pct = round((plastic / total) * 100) if total > 0 else 0
            metal_pct = round((metal / total) * 100) if total > 0 else 0
            glass_pct = round((glass / total) * 100) if total > 0 else 0

            report += f"\nEstimated Recyclable Components:\n"
            report += f"Plastic: {plastic_pct}%\n"
            report += f"Metal: {metal_pct}%\n"
            report += f"Glass: {glass_pct}%\n"

            non_degradable_pct = plastic_pct + metal_pct + glass_pct
            degradable_pct = 100 - non_degradable_pct if total > 0 else 0

            report += "\n♻️ Waste Classification:\n"
            report += f"✅Degradable Waste: {degradable_pct}% \n"
            report += f"❗Non-Degradable Waste: {non_degradable_pct}%"
        else:
            report += f"\n⚠️ Unrecognized Item: {item} (not in model.safetensors)"

        report += "\n" + "-" * 40

    return report


# ===== GUI Setup =====
class ImageAnalyzer(QWidget):
    def __init__(self, model, label2idx):
        super().__init__()
        self.model = model
        self.label2idx = label2idx
        self.setWindowTitle("♻️ Recycling Waste Analyzer")
        self.resize(600, 800)
        self.setAcceptDrops(True)

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.init_upload_view()

    def init_upload_view(self):
        print("🟢 UI Ready: Upload view displayed.")
        self.clear_layout()

        title = QLabel("📤 Upload an Image")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        self.main_layout.addWidget(title)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; padding: 40px;")
        self.image_label.setText("Drop or Upload an Image")
        self.main_layout.addWidget(self.image_label)

        self.button = QPushButton("Upload Image")
        self.button.setStyleSheet("padding: 10px; font-size: 16px;")
        self.button.clicked.connect(self.upload_image)
        self.main_layout.addWidget(self.button)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"📥 Dropped image: {file_path}")
                report = analyze_with_yolo(file_path, self.model, self.label2idx)
                self.show_result_view(file_path, report)

    def show_result_view(self, image_path, report):
        print("✅ Displaying results view.")
        self.clear_layout()

        title = QLabel("📷 Image Preview")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        self.main_layout.addWidget(title)

        image = QPixmap(image_path).scaledToHeight(400, Qt.TransformationMode.SmoothTransformation)
        self.image_label = QLabel()
        self.image_label.setPixmap(image)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.image_label)

        report_title = QLabel("📋 Detection Report")
        report_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        report_title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        self.main_layout.addWidget(report_title)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setText(report)
        self.result_text.setStyleSheet("font-family: Consolas; font-size: 14px;")
        self.main_layout.addWidget(self.result_text)

        self.try_again_button = QPushButton("🔁 Try Another Image")
        self.try_again_button.setStyleSheet("padding: 10px; font-size: 16px;")
        self.try_again_button.clicked.connect(self.init_upload_view)
        self.main_layout.addWidget(self.try_again_button)

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName()
        if file_name:
            print(f"📂 Selected image: {file_name}")
            report = analyze_with_yolo(file_name, self.model, self.label2idx)
            self.show_result_view(file_name, report)

    def clear_layout(self):
        while self.main_layout.count():
            child = self.main_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()


# ===== CLI Entry Point =====
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train model from Datasets/*.csv")
    parser.add_argument("--gui", action="store_true", help="Launch PyQt6 GUI")
    parser.add_argument("--image", type=str, help="Analyze single image and print result")

    args = parser.parse_args()

    yolo_model = YolosForObjectDetection.from_pretrained('hustvl/yolos-base')
    processor = YolosImageProcessor.from_pretrained('hustvl/yolos-base')

    if args.train:
        load_and_train_model()

    elif args.gui:
        if not os.path.exists("model.safetensors"):
            print("❌ Model not found. Please run with --train first.")
            sys.exit()
        trained_model, label2idx, _ = load_classifier()
        app = QApplication(sys.argv)
        window = ImageAnalyzer(trained_model, label2idx)
        window.show()
        sys.exit(app.exec())

    elif args.image:
        if not os.path.exists("model.safetensors"):
            print("❌ Model not found. Please run with --train first.")
            sys.exit()
        trained_model, label2idx, _ = load_classifier()
        result = analyze_with_yolo(args.image, trained_model, label2idx)
        print(result)

    else:
        print("ℹ️ Please provide one of the following arguments:")
        print("   --train       Train model from Datasets/*.csv")
        print("   --gui         Launch the PyQt6 GUI")
        print("   --image PATH  Analyze a single image")
