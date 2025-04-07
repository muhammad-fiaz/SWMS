"""
Recycling Waste Management System

Modes:
1. --train       : Trains the classification model using Datasets/*.csv
2. --gui         : Launches the PyQt6 GUI for drag-and-drop image analysis
3. --image PATH  : Analyzes a single image via command line

This tool detects objects in an image using a YOLOS model and predicts
their recyclable material composition (plastic, metal, glass) using a
custom-trained PyTorch model saved as `model.safetensors`.

Author: Muhammad Fiaz
Date: 2025-04-07
License: MIT License
"""

import json
import sys
import os
from time import sleep

from tqdm import tqdm

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_NO_TQDM"] = "1"
import glob
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file, load_file
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QTextEdit,
)
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import Qt, QSize
from PIL import Image
from transformers import YolosImageProcessor, YolosForObjectDetection


# ===== Dataset + Model Setup =====


class RecyclingDataset(Dataset):
    """
    Dataset for converting labeled CSV entries into tensors.

    Args:
        df (pd.DataFrame): DataFrame containing 'label' and material columns.

    Attributes:
        labels (List[str]): List of item labels.
        data (np.ndarray): Feature values (plastic, metal, glass).
        label2idx (Dict[str, int]): Label to index mapping.
        idx2label (Dict[int, str]): Index to label mapping.
        encoded_labels (List[int]): Encoded labels as integers.
    """

    def __init__(self, df):
        self.labels = df["label"].astype(str).tolist()
        self.data = df.drop(columns=["label"]).values.astype(float)
        self.label2idx = {label: idx for idx, label in enumerate(set(self.labels))}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        self.encoded_labels = [self.label2idx[lbl] for lbl in self.labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded_labels[idx]), torch.tensor(
            self.data[idx], dtype=torch.float
        )


class ComponentPredictor(nn.Module):
    """
    PyTorch model to predict component breakdown using label embedding.

    Args:
        num_classes (int): Number of unique item classes.

    Structure:
        - Embedding Layer
        - Fully connected layers to output plastic, metal, and glass scores.
    """

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
    """
    Loads CSV datasets, trains a model, and saves it along with label mappings.

    Returns:
        model (ComponentPredictor): Trained model instance.
        idx2label (Dict[int, str]): Index to label mapping.
    """
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
    # Total steps = epochs × batches per epoch
    total_steps = 100 * len(dataloader)
    progress = tqdm(total=total_steps, desc="🧠 Training Progress", unit="step")

    for epoch in range(100):
        for labels, targets in dataloader:
            preds = model(labels)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress.set_postfix(epoch=epoch + 1, loss=loss.item())
            progress.update(1)

    progress.close()

    tensor_dict = {
        k: v for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)
    }
    save_file(tensor_dict, "model.safetensors")

    with open("label2idx.json", "w+") as f:
        json.dump(dataset.label2idx, f)

    print(
        "✅ Training complete. Model saved as model.safetensors."
        "\n✅ Label mappings saved as label2idx.json."
        "\n🚀 You can now use the --gui or --image options to analyze images."
    )
    return model, {v: k for k, v in dataset.label2idx.items()}


def load_classifier():
    """
    Loads model and label mappings from disk.

    Returns:
        model (ComponentPredictor): Trained PyTorch model.
        label2idx (Dict[str, int]): Mapping from label to index.
        idx2label (Dict[int, str]): Reverse mapping from index to label.
    """
    print("📦 Loading model and label mappings...")

    if not os.path.exists("model.safetensors") or not os.path.exists("label2idx.json"):
        print("❌ Required files not found. Please run with --train first.")
        sys.exit()

    with open("label2idx.json", "r") as f:
        label2idx = json.load(f)

    model = ComponentPredictor(len(label2idx))
    state = load_file("model.safetensors")
    model.load_state_dict(state)
    model.eval()

    print("✅ Model and label2idx loaded successfully.")
    return model, label2idx, {v: k for k, v in label2idx.items()}


# ===== Custom Progress bar =====
def load_model_with_progress():
    """
    Downloads the YOLOS model and processor from HuggingFace with bottom-fixed tqdm.
    """
    print("⬇️ Downloading YOLOS model and processor from HuggingFace...")

    bar = tqdm(
        total=2,
        desc="📦 Model Loading",
        unit="task",
        position=1,
        leave=True,
        file=sys.stdout,
    )

    # Downloading YOLOS model
    yolo_model = YolosForObjectDetection.from_pretrained("hustvl/yolos-base")
    bar.update(1)
    sleep(0.3)  # Sleep to simulate the download time

    # Downloading YOLOS processor
    processor = YolosImageProcessor.from_pretrained("hustvl/yolos-base")
    bar.update(1)
    bar.close()

    print("\n✅ Model and Processor downloaded and loaded successfully!")

    return yolo_model, processor


# ===== YOLO Analysis with multiple item breakdown =====


def analyze_with_yolo(image_path, model, label2idx):
    """
    Analyze image and detect recyclable components for each item.

    Args:
        image_path (str): Path to image file.
        model (nn.Module): Trained model.
        label2idx (Dict[str, int]): Mapping of label to index.

    Returns:
        str: Generated report of object predictions and material estimates.
    """
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
            report += f"✅ Degradable Waste: {degradable_pct}% \n"
            report += f"❗ Non-Degradable Waste: {non_degradable_pct}%"
        else:
            report += f"\n⚠️ Unrecognized Item: {item} (not in model.safetensors)"

        report += "\n" + "-" * 40

    return report


# ===== GUI Setup =====


class ImageAnalyzer(QWidget):
    """
    PyQt6 GUI application to upload and analyze waste images.

    Args:
        model (nn.Module): Trained classification model.
        label2idx (dict): Label-to-index mapping.
    """

    def __init__(self, model, label2idx):
        super().__init__()
        self.model = model
        self.label2idx = label2idx
        self.setWindowIcon(QIcon("assets/logo.png"))
        self.setWindowTitle("Recycling Waste Analyzer")
        self.resize(600, 800)
        self.setAcceptDrops(True)

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.init_upload_view()

    def init_upload_view(self):
        """Sets up the UI view for uploading an image."""
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
        """Enables drag-and-drop support for images."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Handles dropped image and performs analysis."""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                print(f"📥 Dropped image: {file_path}")
                report = analyze_with_yolo(file_path, self.model, self.label2idx)
                self.show_result_view(file_path, report)

    def show_result_view(self, image_path, report):
        """Displays the image and analysis result in a new view."""
        self.clear_layout()

        title = QLabel("📷 Image Preview")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        self.main_layout.addWidget(title)

        image = QPixmap(image_path).scaledToHeight(
            400, Qt.TransformationMode.SmoothTransformation
        )
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
        """Opens file dialog to upload an image for analysis."""
        file_name, _ = QFileDialog.getOpenFileName()
        if file_name:
            print(f"📂 Selected image: {file_name}")
            report = analyze_with_yolo(file_name, self.model, self.label2idx)
            self.show_result_view(file_name, report)

    def clear_layout(self):
        """Clears all widgets from the current layout."""
        while self.main_layout.count():
            child = self.main_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()


# ===== CLI Entry Point =====

if __name__ == "__main__":
    """
   Main entry point for the Recycling Waste Management System.

    Supports the following command-line options:
    --train       : Trains the classification model from Datasets/*.csv files
    --gui         : Launches the drag-and-drop PyQt6 GUI for image analysis
    --image PATH  : Analyzes a single image file and outputs the report

    Automatically loads YOLOS model and processor from HuggingFace on startup.
    Ensures model.safetensors exists before GUI or image analysis is run.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", action="store_true", help="Train model from Datasets/*.csv"
    )
    parser.add_argument("--gui", action="store_true", help="Launch PyQt6 GUI")
    parser.add_argument(
        "--image", type=str, help="Analyze single image and print result"
    )

    args = parser.parse_args()

    yolo_model, processor = load_model_with_progress()

    if args.train:
        load_and_train_model()

    elif args.gui:
        if not os.path.exists("model.safetensors"):
            from PyQt6.QtWidgets import QMessageBox

            QApplication(sys.argv)
            QMessageBox.critical(
                None,
                "Model Not Found",
                "❌ model.safetensors not found.\nPlease run with --train first.",
            )
            sys.exit()
        trained_model, label2idx, _ = load_classifier()
        app = QApplication(sys.argv)
        app.setWindowIcon(QIcon("assets/logo.png"))
        icon = QIcon()
        icon.addFile("assets/logo.png", QSize(16, 16))
        icon.addFile("assets/logo.png", QSize(32, 32))
        icon.addFile("assets/logo.png", QSize(64, 64))
        app.setWindowIcon(icon)
        window = ImageAnalyzer(trained_model, label2idx)
        window.show()
        try:
            sys.exit(app.exec())
        finally:
            print("🛑 Application closed.")

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
