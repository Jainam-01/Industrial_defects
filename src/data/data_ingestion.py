from pathlib import Path

from src.data.data_utils import get_subdirectories
from src.data.data_utils import get_image_files


class DataIngestion:
    """
    Handles discovery of the MVTec AD dataset.
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def dataset_exists(self) -> bool:
        return self.dataset_path.exists()

    def get_dataset_root(self) -> Path:
        return self.dataset_path

    def get_categories(self) -> list[str]:
        """
        Return all MVTec AD categories.
        """
        categories = get_subdirectories(self.dataset_path)
        return [category.name for category in categories]

    def get_category_paths(self) -> dict:
        """
        Return paths for each category.
        """
        category_paths = {}

        for category in self.get_categories():
            category_root = self.dataset_path / category

            category_paths[category] = {
                "train": category_root / "train",
                "test": category_root / "test",
                "ground_truth": category_root / "ground_truth",
            }

        return category_paths
    def get_dataset_metadata(self) -> list[dict]:
        """
        Collect metadata for all images in the dataset.
        """
        metadata = []

        category_paths = self.get_category_paths()

        for category, paths in category_paths.items():

            for split in ["train", "test"]:

                split_path = paths[split]

                for label_dir in split_path.iterdir():

                    if not label_dir.is_dir():
                        continue

                    label = label_dir.name

                    for image_path in get_image_files(label_dir):

                        metadata.append(
                            {
                                "category": category,
                                "split": split,
                                "label": label,
                                "image_path": image_path,
                            }
                        )

        return metadata