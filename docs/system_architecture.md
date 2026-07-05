# System Architecture

## End-to-End Flow

```text
MVTec AD Dataset
      │
      ▼
Data Ingestion and Validation
      │
      ▼
Preprocessing and Augmentation
      │
      ├── Autoencoder Training
      │       │
      │       ▼
      │   Reconstruction-error anomaly score
      │
      └── Vision Transformer Feature Extraction
              │
              ▼
          Embedding-distance anomaly score
      │
      ▼
Evaluation Pipeline
      │
      ▼
MLflow Experiment Tracking
      │
      ▼
Saved Model Artifacts
      │
      ▼
FastAPI Inference API
      │
      ▼
Docker Container
      │
      ├── Kubernetes Deployment and Service manifests
      └── AWS ECS / ECR deployment design