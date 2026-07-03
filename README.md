# Industrial Defect Detection — End-to-End MLOps Computer Vision Pipeline

An end-to-end Computer Vision and MLOps project for industrial anomaly detection using the **MVTec AD** dataset.

The project detects whether an industrial image is **normal** or **defective** using two anomaly-detection approaches:

1. **Autoencoder** — reconstruction-error based anomaly detection  
2. **Vision Transformer (ViT)** — embedding-distance based anomaly detection  

The project demonstrates the complete machine-learning workflow:

```text
Dataset → Validation → Preprocessing → Training → Evaluation
→ Experiment Tracking → Inference API → Docker → CI
```

---

## Project Architecture

```text
MVTec AD Dataset
      │
      ▼
Data Ingestion and Validation
      │
      ▼
Preprocessing and Augmentation
      │
      ▼
Model Training
 ┌──────────────────────────────┐
 │  Autoencoder                 │
 │  Vision Transformer          │
 └──────────────────────────────┘
      │
      ▼
Model Artifacts and Anomaly Scores
      │
      ▼
MLflow Experiment Tracking
      │
      ▼
FastAPI Inference API
      │
      ▼
Docker Container
      │
      ▼
GitHub Actions CI
```

---

## Models

### 1. Autoencoder

The Autoencoder learns to reconstruct **normal industrial images**.

```text
Input Image
    │
    ▼
Encoder
    │
    ▼
Latent Representation
    │
    ▼
Decoder
    │
    ▼
Reconstructed Image
    │
    ▼
Reconstruction Error = Anomaly Score
```

A defective image is expected to have a larger reconstruction error because the model was trained primarily on normal images.

### 2. Vision Transformer

The Vision Transformer converts an image into an embedding vector.

```text
Input Image
    │
    ▼
Vision Transformer
    │
    ▼
Image Embedding
    │
    ▼
Distance from Mean Normal Embedding
    │
    ▼
Anomaly Score
```

A larger embedding distance means that the image is less similar to the normal-image distribution.

---

## Tech Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch, Torchvision |
| Computer Vision | Pillow, Albumentations, OpenCV |
| API | FastAPI, Uvicorn |
| Experiment Tracking | MLflow |
| Configuration | Hydra |
| Data Versioning | DVC |
| Testing | pytest |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Dataset | MVTec AD |

---

## Project Structure

```text
INDUSTRIAL_DEFECTS/
│
├── configs/                         # Hydra configuration files
├── data/
│   ├── raw/                         # MVTec AD dataset (not committed to Git)
│   └── processed/                   # Processed data
├── models/
│   ├── autoencoder/
│   │   ├── config.py
│   │   ├── model.py
│   │   └── train.py
│   └── vision_transformer/
│       ├── config.py
│       ├── model.py
│       └── train.py
├── src/
│   ├── api/                         # FastAPI application
│   ├── data/                        # Data ingestion and validation
│   ├── evaluation/                  # Model evaluation utilities
│   ├── inference/                   # Inference engine
│   ├── preprocessing/               # Image transforms and augmentation
│   ├── tracking/                    # MLflow logging
│   └── utils/                       # Shared helper utilities
├── tests/                           # Unit tests
├── artifacts/                       # Trained model artifacts (not committed)
├── requirements/                    # Dependency files
├── Dockerfile                       # Docker image instructions
├── docker-compose.yml               # Local container orchestration
├── .dockerignore                    # Files excluded from Docker image
├── pyproject.toml                   # Tool and test configuration
└── README.md
```

---

## Local Setup

### 1. Clone the Repository

```powershell
git clone https://github.com/YOUR_USERNAME/Industrial_defects.git
cd Industrial_defects
```

### 2. Create and Activate a Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements/base.txt
```

### 4. Download the Dataset

Download the MVTec AD dataset and place it at:

```text
data/raw/mvtec_ad/
```

---

## Training

### Train the Autoencoder

```powershell
python -m models.autoencoder.train
```

### Train the Vision Transformer

```powershell
python -m models.vision_transformer.train
```

> For CPU-based development, training can be reduced using a smaller subset, batch size, and number of epochs in the model configuration files.

---

## MLflow Experiment Tracking

Run the MLflow UI locally:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

```text
http://127.0.0.1:5000
```

---

## Testing

Run all local tests:

```powershell
pytest -v
```

Run CI-safe tests:

```powershell
pytest tests/test_model_utils.py tests/test_evaluator.py tests/test_mlflow.py -v
```

Dataset-dependent tests require MVTec AD to be present locally.

---

## Run the Inference API with Docker

Ensure Docker Desktop is running, then:

```powershell
docker compose up --build
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Test either endpoint:

```text
POST /predict/autoencoder
POST /predict/vit
```

Example response:

```json
{
  "model": "autoencoder",
  "anomaly_score": 0.071503,
  "prediction": "defective"
}
```

---

## Docker Flow

```text
Browser / Swagger UI
        │
        │ Upload image request
        ▼
localhost:8000
        │
        ▼
Docker Container
        │
        ▼
FastAPI API
        │
        ▼
Inference Engine
        │
        ├── Autoencoder
        └── Vision Transformer
        │
        ▼
Anomaly Score + Prediction
        │
        ▼
JSON Response to Browser
```

The Docker container mounts the local `artifacts/` directory so trained model files can be used without rebuilding the Docker image.

---

## CI/CD Pipeline

GitHub Actions runs automatically on every push or pull request to the `main` branch.

```text
Push to GitHub
      │
      ▼
Install Dependencies
      │
      ▼
Run CI-Safe Unit Tests
      │
      ▼
Build Docker Image
```

The CI pipeline currently runs dataset-independent tests because the MVTec AD dataset is intentionally not uploaded to GitHub.

---
---

## Key MLOps Concepts Demonstrated

- Modular ML project structure
- Dataset ingestion and validation
- Experiment tracking with MLflow
- Configuration management with Hydra
- Data versioning with DVC
- Unit testing with pytest
- Docker containerization
- FastAPI model-serving API
- GitHub Actions CI pipeline
- Separation of training and inference environments
