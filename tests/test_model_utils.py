import torch.nn as nn

from models.base.model_utils import count_parameters


def test_count_parameters():

    model = nn.Linear(10, 5)

    assert count_parameters(model) == 55