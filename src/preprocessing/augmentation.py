import albumentations as A


class ImageAugmentation:
    """
    Handles image augmentation for training data.
    """

    def __init__(self, image_size=(256, 256)):
        self.train_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5,
                ),
                A.GaussNoise(
                    std_range=(0.02, 0.08),
                    p=0.3,
                ),
            ]
        )

        self.test_transform = A.Compose([])

    def augment_train(self, image):
        """
        Apply training augmentations.
        """
        return self.train_transform(image=image)["image"]

    def augment_test(self, image):
        """
        No augmentation for validation/test.
        """
        return self.test_transform(image=image)["image"]