import torch
import torch.nn.functional as F


def autoencoder_score(
    original,
    reconstructed,
):

    return F.mse_loss(
        reconstructed,
        original,
    ).item()


def vit_score(
    embedding,
    mean_embedding,
):

    score = 1 - F.cosine_similarity(
        embedding,
        mean_embedding.unsqueeze(0),
    )

    return score.item()