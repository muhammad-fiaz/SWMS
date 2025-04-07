<p align="center">
  <img src="https://github.com/user-attachments/assets/4a7266a0-105d-4555-8b6c-1462c03608de" alt="preview">
</p>

<h1 align="center">♻️ Smart Waste Management System (SWMS)</h1>


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
phone,42,38,20
bottle,100,0,0
```

- `label`: The item name (must match YOLOS labels for detection).
- `plastic`, `metal`, `glass`: Composition percentages (should sum to ~100).

---

## 🧠 Model

- **Object Detection**: `hustvl/yolos-base` (transformers)
- **Composition Predictor**: PyTorch feed-forward model trained from CSV data
- **File Format**: `.safetensors`

---

## 📸 Project Structure

```text
assets/
    └── logo.png    # Logo
datasets/           # Dataset folder
    └── data1.csv   # dataset example 1
    └── data2.csv   # dataset example 2
.gitignore          # Git ignore file
detect.py           # main detection script
LICENSE             # License file
pyproject.toml      # Project metadata
README.md           # Project documentation
uv.lock             # Dependency lock file
requirements.txt    # Python dependencies
```

## 📊 Output Report Example

```
Detected Items 1 in image.jpg:
- cell phone

Estimated Recyclable Components:
Plastic: 42%
Metal: 38%
Glass: 20%

♻️ Waste Classification:
✅ Degradable Waste: 0% 
❗ Non-Degradable Waste: 100%
----------------------------------------
```

---

## 🪪 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.
