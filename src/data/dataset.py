from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.data_ingestion import DataIngestion
from src.preprocessing.augmentation import ImageAugmentation
from src.preprocessing.preprocessing import ImagePreprocessor


class MVTecDataset(Dataset):
    """
    PyTorch Dataset for MVTec AD.
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        label: str | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split

        self.ingestion = DataIngestion(dataset_path)
        self.preprocessor = ImagePreprocessor()
        self.augmentor = ImageAugmentation()

        self.metadata = [
            sample
            for sample in self.ingestion.get_dataset_metadata()
            if sample["split"] == split
            and (label is None or sample["label"] == label)
        ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        sample = self.metadata[index]

        image = self.preprocessor.preprocess(sample["image_path"])

        if self.split == "train":
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