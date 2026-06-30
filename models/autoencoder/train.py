import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path

from models.autoencoder.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    DATASET_PATH,
    DEV_DATASET_SIZE,
    EPOCHS,
    LAST_MODEL_PATH,
    LEARNING_RATE,
    RESUME_TRAINING
)
from models.autoencoder.model import Autoencoder
from src.data.dataset import MVTecDataset


class AutoencoderTrainer:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = Autoencoder().to(self.device)

        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
        )

        self.best_validation_loss = float("inf")

        self.train_dataset = MVTecDataset(
            dataset_path=DATASET_PATH,
            split="train",
        )

        self.validation_dataset = MVTecDataset(
            dataset_path=DATASET_PATH,
            split="test",
            label="good",
        )

        if DEV_DATASET_SIZE is not None:

            self.train_dataset = Subset(
                self.train_dataset,
                range(min(DEV_DATASET_SIZE, len(self.train_dataset))),
            )

            self.validation_dataset = Subset(
                self.validation_dataset,
                range(min(DEV_DATASET_SIZE, len(self.validation_dataset))),
            )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        self.validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        if (
                RESUME_TRAINING
                and Path(LAST_MODEL_PATH).exists()
            ):

                self.model.load_model(LAST_MODEL_PATH)
                print("Previous checkpoint loaded.")

    def validate(self):

        self.model.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for batch in self.validation_loader:

                images = batch["image"].to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, images)

                validation_loss += loss.item()

        return validation_loss / len(self.validation_loader)

    def train(self):

        for epoch in range(EPOCHS):

            self.model.train()

            train_loss = 0.0

            progress = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{EPOCHS}",
            )

            for batch in progress:

                images = batch["image"].to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, images)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                train_loss += loss.item()

                progress.set_postfix(loss=loss.item())

            train_loss /= len(self.train_loader)

            validation_loss = self.validate()

            if validation_loss < self.best_validation_loss:

                self.best_validation_loss = validation_loss

                self.model.save_model(BEST_MODEL_PATH)

            self.model.save_model(LAST_MODEL_PATH)

            print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
            print(f"Train Loss      : {train_loss:.6f}")
            print(f"Validation Loss : {validation_loss:.6f}")
            print(f"Best Val Loss   : {self.best_validation_loss:.6f}")
            print(f"Learning Rate   : {self.optimizer.param_groups[0]['lr']:.6f}\n")


if __name__ == "__main__":

    trainer = AutoencoderTrainer()

    trainer.train()