from pathlib import Path

import cv2
import numpy as np


class ImagePreprocessor:
    """
    Handles image preprocessing for the MVTec AD dataset.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (256, 256),
    ):
        self.image_size = image_size

    def load_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Unable to load image: {image_path}")

        return image

    def resize(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.image_size)

    def bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0

    def preprocess(self, image_path: Path) -> np.ndarray:
        image = self.load_image(image_path)
        image = self.resize(image)
        image = self.bgr_to_rgb(image)
        image = self.normalize(image)

        return image

    