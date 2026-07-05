from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader, Subset

from models.vision_transformer.model import VisionTransformer
from src.data.dataset import MVTecDataset
from src.evaluation.evaluator import Evaluator


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARTIFACTS_DIR = Path("artifacts")
DATASET_PATH = "data/raw/mvtec_ad"
MEAN_EMBEDDING_PATH = ARTIFACTS_DIR / "vit_mean_embedding.pt"
METRICS_PATH = ARTIFACTS_DIR / "vit_evaluation_metrics.json"

BATCH_SIZE = 4
EVALUATION_SAMPLES = 100
THRESHOLD = 10.0


def extract_images_and_labels(batch):
    """Extract images and labels from the dictionary returned by MVTecDataset."""
    return batch["image"], batch["label"]


def main():
    if not MEAN_EMBEDDING_PATH.exists():
        raise FileNotFoundError(
            f"Mean embedding not found: {MEAN_EMBEDDING_PATH}. "
            "Train the Vision Transformer pipeline before evaluation."
        )

    dataset = MVTecDataset(
        dataset_path=DATASET_PATH,
        split="test",
    )

    evaluation_dataset = Subset(
        dataset,
        range(min(EVALUATION_SAMPLES, len(dataset))),
    )

    loader = DataLoader(
        evaluation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = VisionTransformer().to(DEVICE)
    model.eval()

    mean_embedding = torch.load(
        MEAN_EMBEDDING_PATH,
        map_location=DEVICE,
    ).to(DEVICE)

    labels = []
    scores = []

    with torch.no_grad():
        for batch in loader:
            images, batch_labels = extract_images_and_labels(batch)
            images = images.to(DEVICE)

            embeddings = model(images)

            batch_scores = torch.norm(
                embeddings - mean_embedding,
                dim=1,
            )

            scores.extend(batch_scores.cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())

    evaluator = Evaluator(threshold=THRESHOLD)

    metrics = evaluator.evaluate(
        labels=labels,
        scores=scores,
    )

    print("\nVision Transformer Evaluation Results")
    print("-" * 40)
    print(f"Threshold: {THRESHOLD:.6f}")
    print(f"Evaluation samples: {len(evaluation_dataset)}")

    for metric_name, value in metrics.items():
        if hasattr(value, "shape"):
            print(f"{metric_name}:")
            print(value)
        else:
            print(f"{metric_name}: {value:.4f}")

    serializable_metrics = {}

    for metric_name, value in metrics.items():
        if hasattr(value, "tolist"):
            serializable_metrics[metric_name] = value.tolist()
        else:
            serializable_metrics[metric_name] = float(value)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(serializable_metrics, file, indent=4)

    print(f"\nEvaluation metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()