from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app


def create_image():

    image = Image.new(
        "RGB",
        (256, 256),
        "white",
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def test_root():

    with TestClient(app) as client:

        response = client.get("/")

        assert response.status_code == 200
        assert response.json()["status"] == "running"


def test_models():

    with TestClient(app) as client:

        response = client.get("/models")

        assert response.status_code == 200
        assert "autoencoder" in response.json()["models"]
        assert "vit" in response.json()["models"]


def test_autoencoder():

    image = create_image()

    with TestClient(app) as client:

        response = client.post(
            "/predict/autoencoder",
            files={
                "file": (
                    "image.png",
                    image,
                    "image/png",
                )
            },
        )

        assert response.status_code == 200
        assert "prediction" in response.json()
        assert "anomaly_score" in response.json()


def test_vit():

    image = create_image()

    with TestClient(app) as client:

        response = client.post(
            "/predict/vit",
            files={
                "file": (
                    "image.png",
                    image,
                    "image/png",
                )
            },
        )

        assert response.status_code == 200
        assert "prediction" in response.json()
        assert "anomaly_score" in response.json()