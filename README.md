# 🦴 AI-Assisted Clinical Decision Support System for Rotator Cuff Calcific Tendinopathy Detection

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/egiiddo/TFG)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red)](https://pytorch.org)

> **Final Degree Project (TFG) — Computer Engineering**  
> Universidad Rey Juan Carlos · Academic year 2025/2026  
> Supervised by Juan Miranda Bautista & Joaquín Moreno Guzmán

---

## 📌 Overview

A web-based **Clinical Decision Support System (CDSS)** for detecting and classifying
**Rotator Cuff Calcific Tendinopathy (RCCT)** from shoulder radiographs.
The system combines deep learning, semantic segmentation, object detection,
and radiomic feature extraction into two integrated diagnostic pipelines.

> ⚠️ This tool is intended as **AI-assisted clinical support only** — not as an autonomous diagnostic system.
> All findings must be reviewed by a qualified healthcare professional.

---

## 🧠 System Architecture

### Pipeline 1 — COMBINED\_512 (Standard)

| Step | Component | Detail |
|------|-----------|--------|
| 1 | **CNN Classifier** | VGG19 + Global Max Pooling (512 px, TensorFlow/Keras) |
| 2 | **Hybrid Model** | CNN features + age/sex → scikit-learn ML classifier |
| 3 | **Explainability** | Grad-CAM heatmap on last convolutional block |
| 4 | **Detection** | YOLOv8 (896 px) → bounding box + tendon localization |
| 5 | **Morphology** | U-Net++ calcium segmentation → 23 radiomic features → Logistic Regression (Gärtner Type I–II vs. III) |

### Pipeline 2 — SEG\_300 (Segmentation-guided)

| Step | Component | Detail |
|------|-----------|--------|
| 1 | **U-Net++ Segmentation** | EfficientNet-B2 encoder → humeral head mask (PyTorch) |
| 2 | **Auto-crop ROI** | Mask-guided square crop centered on humeral head |
| 3 | **CNN Classifier** | 300 px crop → TensorFlow classifier |
| 4 | **Explainability** | Grad-CAM on the cropped ROI |
| 5 | **Detection** | Same YOLOv8 + calcium pipeline as Pipeline 1 |

---

## 🚀 Live Demo

The application is deployed on Hugging Face Spaces (Docker SDK):

👉 **[https://huggingface.co/spaces/egiiddo/TFG](https://huggingface.co/spaces/egiiddo/TFG)**

Accepts DICOM, JPG, and PNG shoulder radiographs.  
DICOM files automatically extract laterality, patient age, and sex from metadata tags (`PatientAge`, `PatientSex`, `ImageLaterality`).

---

## 🗂️ Repository Structure

```
TFG-Shoulder-RCCT/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── run.sh
├── Dockerfile
│
├── Backend/
│   ├── app.py                  # Flask API — all endpoints
│   ├── ai_logic.py             # Grad-CAM, U-Net crop, pipeline_300
│   ├── model_loader.py         # Model downloads (gdown) and loading
│   ├── image_utils.py          # DICOM/JPG preprocessing, laterality heuristic
│   ├── config.py               # Env vars and file paths
│   ├── stats_manager.py        # Usage statistics (stats.json)
│   ├── calcium_unet.py         # Calcium segmentation U-Net++
│   ├── calcium_radiomics.py    # 23 GLCM + morphology + intensity features
│   ├── calcium_classifier.py   # Logistic Regression (Gärtner typing)
│   └── Calcium_Pipeline.py     # Orchestrator: U-Net → Radiomics → Classifier
│
└── Frontend/
    ├── index.html              # Pipeline selection + stats dashboard
    ├── model.html              # Main diagnostic interface
    ├── style.css
    └── js/
        ├── index.js            # Stats loading, model selection
        ├── model.js            # Upload, predict, render results
        └── pdf.js              # PDF report generation (jsPDF + html2canvas)
```

---

## ⚙️ Local Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/TFG-Shoulder-RCCT.git
cd TFG-Shoulder-RCCT
pip install -r requirements.txt
```

### Environment Variables

Model weights are downloaded automatically from Google Drive on first run via `gdown`.  
Create a `.env` file in the root with your Drive file IDs:

```env
CNN_FILENAME_URL=your_cnn_model.h5
CNN_DRIVE_ID=xxxxxxxxxxxxxxxxxxxx

ML_FILENAME_URL=your_ml_model.pkl
ML_DRIVE_ID=xxxxxxxxxxxxxxxxxxxx

FEATURES_FILENAME_URL=final_features.pkl
FEAT_DRIVE_ID=xxxxxxxxxxxxxxxxxxxx

UNET_FILENAME_URL=unet_model.pth
UNET_DRIVE_ID=xxxxxxxxxxxxxxxxxxxx

CLASS300_FILENAME_URL=classifier_300.h5
CLASS300_DRIVE_ID=xxxxxxxxxxxxxxxxxxxx

RE_PT_FILENAME_URL=yolo_re.pt
RE_PT_DRIVE_ID=xxxxxxxxxxxxxxxxxxxx

RI_PT_FILENAME_URL=yolo_ri.pt
RI_PT_DRIVE_ID=xxxxxxxxxxxxxxxxxxxx

MODELJOBLIB_FILENAME_URL=calcium_lr.joblib
MODELJOBLIB_DRIVE_ID=xxxxxxxxxxxxxxxxxxxx

MODELPTH_FILENAME_URL=calcium_unet.pth
MODELPTH_DRIVE_ID=xxxxxxxxxxxxxxxxxxxx
```

### Run

```bash
bash run.sh
# App available at http://localhost:7860
```

---

## 🔬 Key Features

- **Dual diagnostic pipeline** with configurable decision threshold (0.3–0.7, default 0.5)
- **DICOM support** with automatic metadata extraction (laterality, age, sex)
- **Grad-CAM explainability** for both CNN pipelines
- **YOLOv8 object detection** with tendon localization (supraspinatus, infraspinatus, subscapularis)
- **Gärtner classification** (Type I–II vs. Type III) via radiomic features + Logistic Regression
- **PDF diagnostic report** generation (jsPDF + html2canvas)
- **Real-time usage statistics** dashboard (per model, laterality, diagnosis outcome)
- **Regulatory alignment**: designed as a CDSS under EU MDR 2017/745 and the EU AI Act; GDPR privacy-by-design

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask 3.1, Python 3.10 |
| Deep Learning | TensorFlow/Keras 2.15, PyTorch 2.2 |
| Segmentation | segmentation-models-pytorch 0.3.3 (U-Net++) |
| Detection | Ultralytics YOLOv8 8.4 |
| Classical ML | scikit-learn 1.7 (Logistic Regression, ColumnTransformer) |
| Image Processing | OpenCV 4.9, SimpleITK 2.3, pydicom 2.4 |
| Radiomics | scikit-image 0.21 (GLCM, regionprops) |
| Frontend | Vanilla JS, CSS3, jsPDF, html2canvas |
| Deployment | Hugging Face Spaces (Docker SDK), Google Drive (model hosting via gdown) |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/stats` | Usage statistics |
| `POST` | `/predict` | Legacy unified prediction |
| `POST` | `/predict_512_fast` | COMBINED_512 — fast phase (CNN + Hybrid) |
| `POST` | `/predict_512_gradcam` | COMBINED_512 — Grad-CAM (background) |
| `POST` | `/predict_300_gradcam` | SEG_300 — Grad-CAM on ROI (background) |
| `POST` | `/predict_yolo` | YOLOv8 detection + calcium typing pipeline |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👥 Authors

- **Sergio Egido Rodríguez** — Biomedical Engineering student, URJC
- **Juan Miranda Bautista** — Supervisor
- **Joaquín Moreno Guzmán** — Co-supervisor

---

*AI-DR System © 2025 · Final Degree Project, Universidad Rey Juan Carlos*
