import torch
from torch.utils.data import DataLoader, Subset

from models.vision_transformer.config import (
    DATASET_PATH,
    DEV_DATASET_SIZE,
    EMBEDDING_PATH,
)
from models.vision_transformer.model import VisionTransformer
from src.data.dataset import MVTecDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionTransformer().to(device)
model.eval()

dataset = MVTecDataset(
    dataset_path=DATASET_PATH,
    split="train",
)

if DEV_DATASET_SIZE is not None:
    dataset = Subset(
        dataset,
        range(min(DEV_DATASET_SIZE, len(dataset))),
    )

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
)

embeddings = []

with torch.no_grad():

    for batch in loader:

        images = batch["image"].to(device)

        features = model(images)
        embeddings.append(features.cpu())

embeddings = torch.cat(embeddings)

mean_embedding = embeddings.mean(dim=0)

torch.save(mean_embedding, EMBEDDING_PATH)

print("Mean embedding saved.")