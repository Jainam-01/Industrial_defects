import torch

from src.inference.inference import (
    InferenceEngine,
)


def test_autoencoder():

    engine = InferenceEngine(
        "autoencoder"
    )

    image = torch.randn(
        1,
        3,
        256,
        256,
    )

    result = engine.predict(image)

    assert "prediction" in result
    assert "anomaly_score" in result


def test_vit():

    engine = InferenceEngine(
        "vit"
    )

    image = torch.randn(
        1,
        3,
        256,
        256,
    )

    result = engine.predict(image)

    assert "prediction" in result
    assert "anomaly_score" in result