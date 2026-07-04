from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

from models.autoencoder.model import Autoencoder
from src.data.dataset import MVTecDataset
from src.evaluation.evaluator import Evaluator


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARTIFACTS_DIR = Path("artifacts")
DATASET_PATH = "data/raw/mvtec_ad"
MODEL_PATH = ARTIFACTS_DIR / "autoencoder.pth"
METRICS_PATH = ARTIFACTS_DIR / "autoencoder_evaluation_metrics.json"

BATCH_SIZE = 8
THRESHOLD = 0.05


def extract_images_and_labels(batch):
    """
    Extract images and labels from the dictionary returned by MVTecDataset.
    """
    images = batch["image"]
    labels = batch["label"]

    return images, labels


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}. "
            "Train the Autoencoder before evaluation."
        )

    dataset = MVTecDataset(
        dataset_path=DATASET_PATH,
        split="test",
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = Autoencoder().to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    labels = []
    scores = []

    with torch.no_grad():
        for batch in loader:
            images, batch_labels = extract_images_and_labels(batch)

            images = images.to(DEVICE)
            reconstructions = model(images)

            batch_scores = torch.mean(
                (images - reconstructions) ** 2,
                dim=(1, 2, 3),
            )

            scores.extend(batch_scores.cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())

    evaluator = Evaluator(threshold=THRESHOLD)

    metrics = evaluator.evaluate(
        labels=labels,
        scores=scores,
    )

    print("\nAutoencoder Evaluation Results")
    print("-" * 35)
    print(f"Threshold: {THRESHOLD:.6f}")

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