# ♻️ Smart Waste Management System (SWMS)

An intelligent system to detect, classify, and analyze waste using AI-powered object detection and material composition prediction. This project combines YOLOS object detection with a custom-trained classifier to identify and categorize waste as degradable or non-degradable.

---

## 🚀 Features

- 🧠 Object detection using YOLOS (Vision Transformer)
- 📊 Predicts plastic, metal, and glass composition
- 🔍 Classifies waste as **Degradable** or **Non-Degradable**
- 📤 Simple drag-and-drop **GUI** using PyQt6
- 📁 CLI support for training, single-image analysis, and GUI mode
- 🧪 Easily extensible dataset-based training
- ✅ MIT Licensed & open source

---

## 📦 Installation

> Requires Python 3.12+

```bash
pip install -r requirements.txt
```

Or using [`pyproject.toml`](pyproject.toml):

```bash
pip install .
```

---

## 🛠️ Usage

### 🔧 Train the model

Place your CSV files in `Datasets/` (e.g. `Datasets/data1.csv`), then:

```bash
python main.py --train
```

### 🖼️ Analyze an image via GUI

```bash
python main.py --gui
```

### 🖼️ Analyze an image via command-line

```bash
python main.py --image path/to/image.jpg
```

---

## 📁 Dataset Format

CSV files should be like:

```csv
label,plastic,metal,glass
bottle,80,10,10
can,10,85,5
```

- `label`: The item name (must match YOLOS labels for detection).
- `plastic`, `metal`, `glass`: Composition percentages (should sum to ~100).

---

## 🧠 Model

- **Object Detection**: `hustvl/yolos-base` (transformers)
- **Composition Predictor**: PyTorch feed-forward model trained from CSV data
- **File Format**: `.safetensors`

---

## 📊 Output Report Example

```
Detected Items in sample.jpg:
- bottle
- can

Estimated Recyclable Components:
Plastic: 45%
Metal: 50%
Glass: 5%

♻️ Waste Classification:
Degradable Waste: 0%
Non-Degradable Waste: 100% ❌
```

---

## 🪪 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.
