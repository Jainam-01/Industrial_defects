from io import BytesIO

from PIL import Image
from torchvision import transforms


transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)


def preprocess_image(image_bytes):

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    image = transform(image)

    return image.unsqueeze(0)