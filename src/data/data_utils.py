from pathlib import Path

# Supported image formats
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image_file(file_path: Path) -> bool:
    """
    Check if a file is a supported image.
    """
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def get_subdirectories(directory: Path) -> list[Path]:
    """
    Return all immediate subdirectories.
    """
    return sorted([d for d in directory.iterdir() if d.is_dir()])


def get_image_files(directory: Path) -> list[Path]:
    """
    Return all image files in a directory.
    """
    return sorted(
        [f for f in directory.iterdir() if f.is_file() and is_image_file(f)]
    )