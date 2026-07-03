from pathlib import Path

import torch

from models.autoencoder.config import BEST_MODEL_PATH
from models.autoencoder.model import Autoencoder

from models.vision_transformer.config import (
    EMBEDDING_PATH,
)
from models.vision_transformer.model import (
    VisionTransformer,
)

from src.inference.scoring import (
    autoencoder_score,
    vit_score,
)


class InferenceEngine:

    AE_THRESHOLD = 0.05
    VIT_THRESHOLD = 0.30

    def __init__(self, model: str):

        self.model_name = model.lower()

        if self.model_name == "autoencoder":

            self.model = Autoencoder()

            if Path(BEST_MODEL_PATH).exists():
                self.model.load_model(
                    BEST_MODEL_PATH
                )

        elif self.model_name == "vit":

            self.model = VisionTransformer()

            self.mean_embedding = torch.load(
                EMBEDDING_PATH,
                map_location="cpu",
            )

        else:

            raise ValueError(
                "Supported models: autoencoder, vit"
            )

        self.model.eval()

    def predict(self, image):

        with torch.no_grad():

            if self.model_name == "autoencoder":

                reconstruction = self.model(
                    image
                )

                score = autoencoder_score(
                    image,
                    reconstruction,
                )

                prediction = (
                    "defective"
                    if score >= self.AE_THRESHOLD
                    else "normal"
                )

            else:

                embedding = self.model(
                    image
                )

                score = vit_score(
                    embedding,
                    self.mean_embedding,
                )

                prediction = (
                    "defective"
                    if score >= self.VIT_THRESHOLD
                    else "normal"
                )

        return {
            "anomaly_score": round(score, 6),
            "prediction": prediction,
        }