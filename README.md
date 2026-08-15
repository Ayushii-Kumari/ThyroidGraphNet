# ThyroidGraphNet: Uncertainty-Guided Hybrid ConvNeXt-Swin-GATv2 Network

A deep-learning framework for **thyroid ultrasound image classification** that combines **ConvNeXt CNN, Swin Transformer, Multi-Head GATv2, and uncertainty-guided learning** for robust and reliable prediction.

---

## 🚀 Overview

The proposed system combines local visual features, global transformer features, and graph-based feature relationships.

### 📌 Core Pipeline

```text
Thyroid Ultrasound Image
          │
          ▼
    Preprocessing
          │
          ▼
ConvNeXt Feature Extraction
          │
          ▼
  Uncertainty Estimation
          │
      ┌───┴───┐
      │       │
    Low U   High U
      │       │
      ▼       ▼
   CNN Path   Swin Transformer
      │       │
      └───┬───┘
          │
          ▼
     Feature Fusion
          │
          ▼
   Graph Construction
          │
          ▼
  Uncertainty Estimation
          │
      ┌───┴───┐
      │       │
    Low U   High U
      │       │
      ▼       ▼
 Sparse Graph  Dense Graph
      │       │
      └───┬───┘
          │
          ▼
    Multi-Head GATv2
          │
          ▼
    Softmax Classifier
          │
      ┌───┴───┐
      ▼       ▼
   Benign  Malignant
```

---

## 🧠 Models & Techniques Used

### ConvNeXt

Used as the **CNN backbone** for extracting local spatial and texture features.

### Swin Transformer

Used to learn **global and hierarchical image representations** using shifted-window attention.

### Multi-Head GATv2

Used for **graph-based feature learning**, modeling relationships between extracted feature representations through multi-head graph attention.

### Uncertainty-Guided Learning

Used to estimate **prediction uncertainty/confidence** and improve the reliability of the final classification.

---

## 🧩 Pipeline Stages

```text
Phase 1 → Dataset Preparation
Phase 2 → Image Preprocessing
Phase 3 → Data Augmentation
Phase 4 → ConvNeXt Feature Extraction
Phase 5 → Swin Transformer Feature Extraction
Phase 6 → Feature Fusion
Phase 7 → Multi-Head GATv2
Phase 8 → Uncertainty-Guided Prediction
Phase 9 → Classification & Evaluation
```

---

## 📂 Project Structure

```text
ThyroidGraphNet/
│
├── data/                    # Dataset
│
├── models/                  # Model architectures
│   ├── convnext.py
│   ├── swin.py
│   ├── gatv2.py
│   └── uncertainty.py
│
├── dataset/                 # Dataset loading & preprocessing
│
├── training/                # Training pipeline
│
├── evaluation/              # Evaluation & metrics
│
├── results/                 # Generated results
│
├── checkpoints/             # Saved model weights
│
├── .github/
│   └── workflows/           # GitHub Actions CI/CD
│
├── main.py                  # Main execution
├── requirements.txt         # Dependencies
├── Dockerfile               # Docker configuration
├── .dockerignore
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Running

### Clone Repository

```bash
git clone https://github.com/Ayushii-Kumari/ThyroidGraphNet.git
cd ThyroidGraphNet
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train / Run

```bash
python main.py
```

The project supports **CUDA GPU acceleration** and was tested using **Google Colab with NVIDIA Tesla T4**.

---

## 🧪 Dataset

The project uses a **public thyroid ultrasound dataset** for training, validation, and testing.

Dataset source and related research references are documented below.

---

## 📊 Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Generated model checkpoints and results are stored in the respective output directories.

---

## 🐳 Docker & CI/CD

### Docker

Docker is used to **containerize the project and its dependencies**, allowing consistent execution across environments.

```bash
docker build -t thyroidgraphnet .
docker run thyroidgraphnet
```

### GitHub Actions

GitHub Actions is used for **CI/CD automation**, including project checks and automated Docker build/deployment workflow.

```text
GitHub Push
     ↓
GitHub Actions
     ↓
Checks / Tests
     ↓
Docker Build
     ↓
Deployment
```

---

## 🧰 Technologies Used

* Python
* PyTorch
* ConvNeXt
* Swin Transformer
* Multi-Head GATv2
* Uncertainty-Guided Learning
* CUDA
* Google Colab
* Docker
* GitHub Actions

---

## 🔗 Public Resources

* **GitHub Repository:** https://github.com/Ayushii-Kumari/ThyroidGraphNet
* **Dataset:** TN5000 — An Ultrasound Image Dataset for Thyroid Nodule Detection and Classification
  https://figshare.com/s/cb6a67f17c04b29e7edd
* **Research Paper:** TN5000: An Ultrasound Image Dataset for Thyroid Nodule Detection and Classification
  https://www.nature.com/articles/s41597-025-05757-4
* **Docker Hub:** https://hub.docker.com/repository/docker/ayushiikumari/thyroid-graph-net/general
* **GitHub Actions CI/CD:** https://github.com/Ayushii-Kumari/ThyroidGraphNet/actions
* **GitHub Actions:** https://github.com/features/actions

---

## 📌 Project Status

**Research / Experimental Implementation**

ThyroidGraphNet integrates **ConvNeXt + Swin Transformer + Multi-Head GATv2 + Uncertainty-Guided Learning** into a unified thyroid ultrasound classification pipeline, with **Docker and GitHub Actions** supporting containerization and CI/CD.
