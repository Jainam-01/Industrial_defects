import torch

from src.data.dataset import MVTecDataset


def test_dataset():

    dataset = MVTecDataset(
        dataset_path="data/raw/mvtec_ad",
        split="train",
    )

    sample = dataset[0]

    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 256, 256)