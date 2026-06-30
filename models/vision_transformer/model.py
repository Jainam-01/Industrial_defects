import torch
import torch.nn as nn

from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
)
from torchvision.models.vision_transformer import interpolate_embeddings

from models.base.base_model import BaseModel
from models.vision_transformer.config import (
    IMAGE_SIZE,
    PATCH_SIZE,
    PRETRAINED,
)


class VisionTransformer(BaseModel):
    """
    Vision Transformer Feature Extractor
    """

    def __init__(self):
        super().__init__()

        weights = (
            ViT_B_16_Weights.DEFAULT
            if PRETRAINED
            else None
        )

        self.model = vit_b_16(
            weights=None,
            image_size=IMAGE_SIZE,
        )

        if PRETRAINED:

            state_dict = weights.get_state_dict(progress=True)

            state_dict = interpolate_embeddings(
                image_size=IMAGE_SIZE,
                patch_size=PATCH_SIZE,
                model_state=state_dict,
            )

            self.model.load_state_dict(
                state_dict,
                strict=False,
            )

        # Remove classification head
        self.model.heads = nn.Identity()

    def forward(self, x):
        """
        Returns a 768-dimensional embedding.
        """
        return self.model(x)

    def predict(self, x):

        self.eval()

        with torch.no_grad():
            return self.forward(x)