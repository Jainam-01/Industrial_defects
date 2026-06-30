import torch

from models.base.model_utils import count_parameters
from models.vision_transformer.model import VisionTransformer


def test_forward_pass():

    model = VisionTransformer()

    x = torch.randn(2, 3, 256, 256)

    y = model(x)

    assert y.shape == (2, 2)


def test_prediction():

    model = VisionTransformer()

    x = torch.randn(2, 3, 256, 256)

    prediction = model.predict(x)

    assert prediction.shape == (2,)


def test_parameter_count():

    model = VisionTransformer()

    params = count_parameters(model)

    assert params > 0


def test_save_load(tmp_path):

    model = VisionTransformer()

    save_path = tmp_path / "vit.pth"

    model.save_model(save_path)

    loaded_model = VisionTransformer()

    loaded_model.load_model(save_path)

    for p1, p2 in zip(
        model.parameters(),
        loaded_model.parameters(),
    ):
        assert torch.equal(p1, p2)