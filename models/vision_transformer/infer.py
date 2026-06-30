import torch
import torch.nn.functional as F

from models.vision_transformer.config import EMBEDDING_PATH
from models.vision_transformer.model import VisionTransformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionTransformer().to(device)
model.eval()

reference = torch.load(EMBEDDING_PATH).to(device)


def anomaly_score(image):

    with torch.no_grad():

        features = model(image)

        score = 1 - F.cosine_similarity(
            features,
            reference.unsqueeze(0),
        )

    return score.item()