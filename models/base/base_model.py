from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models.
    """

    def __init__(self):
        super().__init__()

    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass.
        """
        pass


    @abstractmethod
    def predict(self, x):
        """
        Perform inference.
        """
        pass

    def save_model(self, path: str):
        """
        Save model weights.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        """
        Load model weights.
        """
        self.load_state_dict(torch.load(path))
        self.eval()