import torch

from models.autoencoder.model import Autoencoder
from models.base.model_utils import count_parameters


def test_forward_pass():

    model = Autoencoder()

    x = torch.randn(2, 3, 256, 256)

    y = model(x)

    assert y.shape == x.shape


def test_prediction():

    model = Autoencoder()

    x = torch.randn(1, 3, 256, 256)

    y = model.predict(x)

    assert y.shape == x.shape


def test_parameter_count():

    model = Autoencoder()

    params = count_parameters(model)

    assert params > 0

import torch

from models.autoencoder.model import Autoencoder
from models.base.model_utils import count_parameters


def test_forward_pass():

    model = Autoencoder()

    x = torch.randn(2, 3, 256, 256)

    y = model(x)

    assert y.shape == x.shape


def test_prediction():

    model = Autoencoder()

    x = torch.randn(1, 3, 256, 256)

    y = model.predict(x)

    assert y.shape == x.shape


def test_parameter_count():

    model = Autoencoder()

    params = count_parameters(model)

    assert params > 0