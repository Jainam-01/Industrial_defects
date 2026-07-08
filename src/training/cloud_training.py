import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from models.autoencoder.model import Autoencoder
from models.vision_transformer.model import VisionTransformer
from src.data.dataset import MVTecDataset
from src.evaluation.evaluator import Evaluator


class NormalTrainingDataset(MVTecDataset):
    """Normal-sample dataset that can switch between train and validation preprocessing."""

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        label: str | None = None,
        use_augmentations: bool = True,
    ):
        super().__init__(dataset_path=dataset_path, split=split, label=label)
        self.use_augmentations = use_augmentations

    def __getitem__(self, index):
        sample = self.metadata[index]

        image = self.preprocessor.preprocess(sample["image_path"])
        if self.use_augmentations:
            image = self.augmentor.augment_train(image)
        else:
            image = self.augmentor.augment_test(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        label = 0 if sample["label"] == "good" else 1

        return {
            "image": image,
            "label": label,
            "category": sample["category"],
            "image_path": str(sample["image_path"]),
        }


def _load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover - fallback for lightweight environments
        yaml = None

    if yaml is not None:
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    config: dict[str, Any] = {}
    with config_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if value.lower() in {"true", "false"}:
                parsed_value = value.lower() == "true"
            elif value.replace(".", "", 1).isdigit():
                parsed_value = float(value) if "." in value else int(value)
            else:
                parsed_value = value
            config[key] = parsed_value

    return config


def _ensure_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = _ensure_dir(Path(path).parent) / Path(path).name
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(payload), handle, indent=2)
    return output_path


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_train_validation_split(
    dataset_path: str | Path,
    validation_split_ratio: float,
    seed: int,
) -> tuple[Subset, Subset]:
    base_dataset = NormalTrainingDataset(
        dataset_path=str(dataset_path),
        split="train",
        label="good",
        use_augmentations=False,
    )

    if not base_dataset:
        raise ValueError("No normal training images were found in the training split.")

    val_size = int(len(base_dataset) * validation_split_ratio)
    val_size = max(1, min(val_size, len(base_dataset) - 1))
    train_size = len(base_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(base_dataset), generator=generator).tolist()
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_subset = Subset(base_dataset, train_indices)
    val_subset = Subset(base_dataset, val_indices)

    return train_subset, val_subset


def _build_dataloader(
    dataset: Subset,
    batch_size: int,
    shuffle: bool,
    config: dict[str, Any],
    device: torch.device,
) -> DataLoader:
    num_workers = int(config.get("num_workers", 2 if device.type == "cuda" else 0))
    pin_memory = bool(config.get("pin_memory", device.type == "cuda"))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def train_autoencoder(
    train_dataset: Subset,
    validation_dataset: Subset,
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> tuple[Autoencoder, dict[str, Any], float]:
    batch_size = int(config.get("autoencoder_batch_size", config.get("batch_size", 8)))
    epochs = int(config.get("epochs", 1))
    learning_rate = float(config.get("learning_rate", 1e-3))

    model = Autoencoder().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset.dataset.use_augmentations = True
    validation_dataset.dataset.use_augmentations = False

    train_loader = _build_dataloader(train_dataset, batch_size, shuffle=True, config=config, device=device)
    validation_loader = _build_dataloader(validation_dataset, batch_size, shuffle=False, config=config, device=device)

    best_validation_loss = float("inf")
    best_model_path = output_dir / "autoencoder_best.pth"
    last_model_path = output_dir / "autoencoder_last.pth"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            reconstructions = model(images)
            loss = criterion(reconstructions, images)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / max(1, len(train_loader))
        validation_loss = _compute_autoencoder_validation_loss(model, validation_loader, device, criterion)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            model.save_model(str(best_model_path))

        model.save_model(str(last_model_path))

    model.load_model(str(best_model_path))
    metrics = {
        "best_validation_loss": best_validation_loss,
        "best_checkpoint": str(best_model_path),
        "last_checkpoint": str(last_model_path),
        "epochs": epochs,
        "train_loss": avg_train_loss,
    }
    return model, metrics, best_validation_loss


def _compute_autoencoder_validation_loss(
    model: Autoencoder,
    validation_loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in validation_loader:
            images = batch["image"].to(device)
            reconstructions = model(images)
            losses.append(criterion(reconstructions, images).item())
    return float(np.mean(losses)) if losses else float("inf")


def _compute_autoencoder_scores(
    model: Autoencoder,
    loader: DataLoader,
    device: torch.device,
) -> list[float]:
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            reconstructions = model(images)
            errors = torch.mean((reconstructions - images) ** 2, dim=(1, 2, 3))
            scores.extend(errors.detach().cpu().tolist())
    return scores


def _compute_vit_scores(
    model: VisionTransformer,
    mean_embedding: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
) -> list[float]:
    model.eval()
    mean_embedding = mean_embedding.to(device)
    scores: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            embeddings = model(images)
            for embedding in embeddings:
                similarity = F.cosine_similarity(embedding.unsqueeze(0), mean_embedding.unsqueeze(0))
                scores.append(float(1.0 - similarity.item()))
    return scores


def calibrate_threshold(scores: list[float], percentile: float) -> float:
    return float(np.percentile(scores, percentile))


def _summarize_scores(scores: list[float]) -> dict[str, float | int]:
    values = np.asarray(scores, dtype=np.float32)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "percentile_95": float(np.percentile(values, 95)),
    }


def build_vit_reference_embedding(
    dataset_path: str | Path,
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> tuple[VisionTransformer, torch.Tensor, dict[str, Any]]:
    batch_size = int(config.get("vit_batch_size", config.get("batch_size", 8)))
    model = VisionTransformer().to(device)
    model.eval()

    normal_train_dataset = MVTecDataset(
        dataset_path=str(dataset_path),
        split="train",
        label="good",
    )
    loader = DataLoader(normal_train_dataset, batch_size=batch_size, shuffle=False)

    embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            features = model(images)
            embeddings.append(features.detach().cpu())

    if not embeddings:
        raise ValueError("No embeddings were generated for the Vision Transformer reference embedding.")

    mean_embedding = torch.cat(embeddings, dim=0).mean(dim=0)
    embedding_path = output_dir / "vit_mean_embedding.pt"
    torch.save(mean_embedding.detach().cpu(), embedding_path)

    metrics = {
        "embedding_path": str(embedding_path),
        "embedding_dim": int(mean_embedding.numel()),
    }
    return model, mean_embedding.to(device), metrics


def evaluate_dataset(
    scores: list[float],
    labels: list[int],
    threshold: float,
) -> dict[str, Any]:
    evaluator = Evaluator(threshold=threshold)
    metrics = evaluator.evaluate(labels, scores)
    return {
        "threshold": threshold,
        "metrics": {
            "roc_auc": float(metrics["roc_auc"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        },
    }


def run_cloud_training(
    config_path: str | Path = "configs/cloud_training.yaml",
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    config = _load_yaml_config(config_path)
    _set_seed(int(config.get("seed", 42)))

    dataset_path = str(dataset_path or config.get("dataset_path", "data/raw/mvtec_ad"))
    output_dir = _ensure_dir(config.get("artifact_output_dir", "artifacts/cloud_training"))

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and str(config.get("device", "auto")).lower() != "cpu"
        else "cpu"
    )

    train_dataset, validation_dataset = _create_train_validation_split(
        dataset_path=dataset_path,
        validation_split_ratio=float(config.get("validation_split_ratio", 0.1)),
        seed=int(config.get("seed", 42)),
    )

    autoencoder_model, autoencoder_metrics, best_validation_loss = train_autoencoder(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    validation_loader = DataLoader(validation_dataset, batch_size=int(config.get("autoencoder_batch_size", config.get("batch_size", 8))), shuffle=False)
    autoencoder_validation_scores = _compute_autoencoder_scores(autoencoder_model, validation_loader, device)
    autoencoder_threshold = calibrate_threshold(
        autoencoder_validation_scores,
        float(config.get("threshold_percentile", 95)),
    )

    vit_model, mean_embedding, vit_embedding_metrics = build_vit_reference_embedding(
        dataset_path=dataset_path,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    vit_validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config.get("vit_batch_size", config.get("batch_size", 8))),
        shuffle=False,
    )
    vit_validation_scores = _compute_vit_scores(vit_model, mean_embedding, vit_validation_loader, device)
    vit_threshold = calibrate_threshold(vit_validation_scores, float(config.get("threshold_percentile", 95)))

    test_dataset = MVTecDataset(dataset_path=dataset_path, split="test")
    test_loader = DataLoader(test_dataset, batch_size=int(config.get("autoencoder_batch_size", config.get("batch_size", 8))), shuffle=False)

    autoencoder_test_scores = _compute_autoencoder_scores(autoencoder_model, test_loader, device)
    vit_test_loader = DataLoader(test_dataset, batch_size=int(config.get("vit_batch_size", config.get("batch_size", 8))), shuffle=False)
    vit_test_scores = _compute_vit_scores(vit_model, mean_embedding, vit_test_loader, device)

    test_labels = [int(item["label"]) for item in test_dataset]

    autoencoder_evaluation = evaluate_dataset(
        autoencoder_test_scores,
        test_labels,
        autoencoder_threshold,
    )
    vit_evaluation = evaluate_dataset(
        vit_test_scores,
        test_labels,
        vit_threshold,
    )

    metrics_payload = {
        "config": config,
        "device": str(device),
        "autoencoder": {
            **autoencoder_metrics,
            "validation_threshold": autoencoder_threshold,
            "validation_scores": {
                **_summarize_scores(autoencoder_validation_scores),
                "threshold_percentile": float(config.get("threshold_percentile", 95)),
            },
            "test_scores": _summarize_scores(autoencoder_test_scores),
            "evaluation": autoencoder_evaluation,
        },
        "vision_transformer": {
            **vit_embedding_metrics,
            "validation_threshold": vit_threshold,
            "validation_scores": {
                **_summarize_scores(vit_validation_scores),
                "threshold_percentile": float(config.get("threshold_percentile", 95)),
            },
            "test_scores": _summarize_scores(vit_test_scores),
            "evaluation": vit_evaluation,
        },
    }

    thresholds_payload = {
        "autoencoder_threshold": autoencoder_threshold,
        "vit_threshold": vit_threshold,
    }

    metrics_path = _write_json(output_dir / "metrics.json", metrics_payload)
    thresholds_path = _write_json(output_dir / "thresholds.json", thresholds_payload)

    return {
        "device": str(device),
        "artifacts_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "thresholds_path": str(thresholds_path),
        "best_validation_loss": best_validation_loss,
        "autoencoder_threshold": autoencoder_threshold,
        "vit_threshold": vit_threshold,
    }
