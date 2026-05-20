[README (1).md](https://github.com/user-attachments/files/28038964/README.1.md)

# 🔧 CWRU Bearing Fault Detection — Predictive Maintenance

> 10-class mechanical bearing fault detection using a CNN-LSTM hybrid model on the CWRU vibration dataset — **97.83% test accuracy** with SHAP explainability and an interactive analytics dashboard.

---

## 📌 Overview

Industrial rotating machinery failures cause significant downtime and economic loss. Early and accurate fault detection is critical for manufacturing reliability. This project builds an end-to-end predictive maintenance pipeline that classifies **10 types of bearing faults** from raw vibration signals using a deep learning approach.

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | [CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter) |
| Samples | 4,600 (balanced — 460 per class) |
| Sampling Rate | 48 kHz |
| Input Shape | 32 × 32 vibration matrices |
| Classes | 10 (Normal + 9 fault types) |

**Fault Classes:**
`Normal` · `Ball_007` · `Ball_014` · `Ball_021` · `IR_007` · `IR_014` · `IR_021` · `OR_007` · `OR_014` · `OR_021`

---

## 🏗️ ML Pipeline

```
Raw .NPZ Data
     │
     ▼
Feature Engineering
(RMS · Crest Factor · Peak-to-Peak · Dominant FFT Frequency)
     │
     ▼
StandardScaler Normalization → Reshape (4600, 1024, 1)
     │
     ▼
CNN-LSTM Model Training
(EarlyStopping · ReduceLROnPlateau · ModelCheckpoint)
     │
     ▼
Evaluation (Accuracy · Precision · Recall · F1 · ROC-AUC)
     │
     ▼
SHAP Explainability + Interactive Dashboard
```

---

## 🧠 Model Architecture

**CNN-LSTM Hybrid** — 86,218 parameters

| Layer | Details |
|---|---|
| Conv1D Block 1 | 32 filters, kernel=5, ReLU + BatchNorm + MaxPool |
| Conv1D Block 2 | 64 filters, kernel=3, ReLU + BatchNorm + MaxPool |
| Conv1D Block 3 | 128 filters, kernel=3, ReLU + BatchNorm + MaxPool |
| LSTM | 64 units — captures temporal vibration dependencies |
| Dense Output | 10 classes, Softmax |

**Training Config:**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 50 (with EarlyStopping, patience=10)
- Batch Size: 64
- Split: 70% train · 15% val · 15% test

---

## 📈 Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **97.83%** |
| Test Loss | 0.0642 |
| Avg Precision | 0.9796 |
| RF Baseline Accuracy | 92.72% |
| Model Size | 1,087.8 KB |
| Inference Latency | ~108 ms/sample |

**Per-class real-time inference confidence:**

| Fault | Confidence |
|---|---|
| Normal | 100.0% |
| Ball_007 | 96.3% |
| IR_007 | 100.0% |
| OR_021 | 100.0% |

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) was applied on the Random Forest model to identify the most fault-discriminating signal features:

- 🔴 **Dominant Frequency** — most important feature; fault type clearly shifts the frequency signature
- 🟡 **Peak-to-Peak Amplitude** — second most important; fault severity changes vibration amplitude range

This adds interpretability to model predictions, making results actionable for maintenance engineers.

---

## 📁 Repository Structure

```
cwru-bearing-fault-detection/
│
├── predictive.ipynb          # Full ML pipeline notebook
├── cwru_dashboard_v2.html    # Interactive analytics dashboard
├── dataset/                  # CWRU .npz dataset files
├── CWRU_Predictive_Maintenance.pptx  # Project presentation
└── README.md
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

- **Deep Learning:** TensorFlow, Keras
- **ML & Explainability:** Scikit-learn, SHAP
- **Data Processing:** NumPy, Pandas
- **Visualisation:** Matplotlib, Seaborn, Plotly (dashboard)

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/arnav_gupta/cwru-bearing-fault-detection.git
cd cwru-bearing-fault-detection

# Install dependencies
pip install tensorflow scikit-learn shap numpy pandas matplotlib seaborn

# Run the notebook
jupyter notebook predictive.ipynb

# View the dashboard
open cwru_dashboard_v2.html
```

---

## 👥 Team

Developed as part of the **AI coursework at Thapar Institute of Engineering & Technology (2024–25)**.

| Member | Roll No. |
|---|---|
| Arnav Gupta | 1024030780 |
| Divyam Mittal | 1024030008 |
| Paarth Mendiratta | 1024030030 |

---

## 📄 References

- [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [TensorFlow Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
