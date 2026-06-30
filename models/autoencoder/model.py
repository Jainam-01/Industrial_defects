import torch
import torch.nn as nn

from models.base.base_model import BaseModel


class Encoder(nn.Module):
    """
    Convolutional Encoder
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """
    Convolutional Decoder
    """

    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                256, 128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32, 3,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(BaseModel):
    """
    Convolutional Autoencoder
    """

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def build_model(self):
        pass

    def forward(self, x):

        latent = self.encoder(x)
        reconstruction = self.decoder(latent)

        return reconstruction

    def predict(self, x):

        self.eval()

        with torch.no_grad():
            return self.forward(x)