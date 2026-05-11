<div align="center">

![SmartLeaf AI Banner](results/hero_banner.jpg)

# SmartLeaf AI
# Mohamed Hassanein | Shady Mohammed | Yehia Haitham

### Intelligent Plant Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

**SmartLeaf AI** is an intelligent web-based agricultural assistant that detects plant diseases using Deep Learning and Computer Vision. Upload a leaf image and get instant disease predictions, treatment recommendations, severity estimation, and a live comparison between two AI approaches.

[Live Demo](#-quick-start) · [Report Bug](issues) · [Dataset](#-dataset)

</div>

---

## 📸 Screenshots

<div align="center">

### Homepage
![Homepage](results/homepage.png)

### Upload & Scan
![Upload Area](results/upload_area.png)

### Results & Treatment
![Results](results/results.png)

</div>

---

## ✨ Features

- 🔬 **Dual AI Prediction** — EfficientNet-B0 CNN and SVM run simultaneously on every scan
- 🎯 **98.67% Accuracy** — fine-tuned EfficientNet-B0 on 54,000+ PlantVillage images
- 💊 **Treatment Recommendations** — causes, treatment steps, prevention, and plant care for all 38 disease classes
- 📊 **Disease Severity Estimation** — OpenCV HSV color analysis estimates mild / moderate / severe
- 📷 **Camera Support** — capture directly from device camera on mobile and desktop
- 📈 **Model Comparison Dashboard** — side-by-side CNN vs SVM metrics with live charts
- 🌱 **Smart Farmer Dashboard** — scan history, statistics, and disease distribution tracking
- 📱 **Mobile-Friendly** — fully responsive design

---

## 🤖 AI Approaches

| | Approach 1 | Approach 2 |
|---|---|---|
| **Name** | EfficientNet-B0 + SVM | EfficientNet-B0 End-to-End |
| **Method** | Frozen feature extractor → SVM (RBF) | Fine-tuned CNN classifier |
| **Accuracy** | 97.63% | 98.67% |
| **F1-Score** | 97.62% | 98.67% |
| **Train Time** | ~51 min (CPU) | ~4 hours (GPU) |
| **Inference** | ~45ms | ~28ms |

Both models run on every prediction, and their outputs are shown side by side on the results page.

---

## 🏗️ Architecture

```
SmartLeaf-AI/
├── backend/
│   ├── app.py                  # Flask REST API
│   ├── predictor.py            # CNN + SVM inference pipeline
│   └── treatment_data.json     # 38 disease treatment entries
├── frontend/
│   ├── index.html              # Single-page web app
│   ├── css/styles.css          # Dark bioluminescent theme
│   └── js/script.js            # Upload, camera, results rendering
├── notebooks/
│   └── train.ipynb             # Full training pipeline notebook
├── models/                     # Saved model files (after training)
├── results/                    # Training plots and metrics
├── dataset/                    # PlantVillage dataset (not included)
└── requirements.txt
```

### API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Serve frontend SPA |
| POST | `/api/predict` | Run CNN + SVM prediction on uploaded image |
| GET | `/api/compare` | Model comparison metrics |
| GET | `/api/stats` | Dashboard statistics and scan history |
| GET | `/api/treatments` | List all 38 disease treatments |
| GET | `/api/treatments/<key>` | Get treatment for specific disease |
| GET | `/api/health` | Server and model status |

---

## 📦 Dataset

**PlantVillage Dataset** — [Download from Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

| Property | Value |
|----------|-------|
| Total images | 54,305 |
| Classes | 38 (healthy + diseased) |
| Plants covered | Apple, Tomato, Potato, Corn, Grape, Peach, Pepper, Cherry, and more |
| Image type | RGB, 224×224px |
| Format | JPG |

After downloading, organize as:
```
dataset/
└── plantvillage dataset/
    └── color/
        ├── Apple___Apple_scab/
        ├── Apple___Black_rot/
        ├── Tomato___Early_blight/
        └── ... (38 folders total)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10
- NVIDIA GPU recommended (CPU works but CNN training takes ~6–10 hours)
- [Miniconda](https://docs.anaconda.com/miniconda/) for GPU setup

### 1. Clone the repository
```bash
git clone https://github.com/medoyea/SmartLeaf-AI.git
cd SmartLeaf-AI
```

### 2. Create and activate environment
```bash
conda create -n smartleaf python=3.10 -y
conda activate smartleaf
```

### 3. Install CUDA (for GPU support)
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1 -y
```

### 4. Install Python dependencies
```bash
pip install tensorflow==2.10.0 scikit-learn numpy opencv-python Pillow Flask flask-cors pandas matplotlib seaborn tqdm python-dotenv jupyter
```

### 5. Download and prepare dataset
Download PlantVillage from Kaggle and place the `color/` folder at:
```
dataset/plantvillage dataset/color/
```

### 6. Train the models
Open `notebooks/train.ipynb` in VS Code or Jupyter, set your `DATA_DIR` in Section 2, and run all cells.

Expected training time:
| Hardware | SVM | CNN |
|----------|-----|-----|
| GTX 1650 (4GB) | ~51 min | ~4 hours |
| RTX 3080 | ~30 min | ~1.5 hours |
| CPU only | ~1 hour | ~6–10 hours |

### 7. Start the backend
```bash
cd backend
python app.py
```

### 8. Open the app
Navigate to **http://localhost:5000** in your browser.

---

## 📊 Results

### Model Performance

| Metric | CNN | SVM |
|--------|-----|-----|
| Accuracy | **98.67%** | 97.63% |
| Precision | **98.67%** | 97.63% |
| Recall | **98.67%** | 97.63% |
| F1-Score | **98.67%** | 97.62% |
| Training Time | 14,449s | 3,092s |

### Key Findings
- End-to-end CNN fine-tuning outperforms feature extraction + SVM by **~1%** in accuracy
- SVM training is **~4.7x faster** than CNN while remaining highly competitive
- Both models exceed the 90% accuracy target specified in the SRS
- The dual-model approach provides prediction confidence comparison on every scan

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3, JavaScript (Vanilla) |
| **Backend** | Python, Flask, Flask-CORS |
| **Deep Learning** | TensorFlow 2.10, Keras, EfficientNet-B0 |
| **Machine Learning** | scikit-learn, SVM (RBF kernel) |
| **Computer Vision** | OpenCV (severity estimation) |
| **Data Processing** | NumPy, Pandas, Pillow |
| **Visualization** | Matplotlib, Seaborn |
| **Dataset** | PlantVillage (Kaggle) |

---

## 🗺️ Future Enhancements

- [ ] Mobile application (React Native / Flutter)
- [ ] Arabic language support
- [ ] Voice assistant integration
- [ ] Edge deployment on Raspberry Pi devices
- [ ] Real-time video stream disease detection
- [ ] Expanded dataset with more plant species

---

## 👥 Team

> Built as a Deep Learning university project.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with 🌿 and a lot of GPU time

</div>