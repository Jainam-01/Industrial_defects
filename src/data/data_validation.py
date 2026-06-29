from pathlib import Path
from src.data.data_utils import get_image_files


class DataValidation:
    """
    Validates the integrity of the MVTec AD dataset.
    """

    EXPECTED_CATEGORIES = {
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    }

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def dataset_exists(self) -> bool:
        return self.dataset_path.exists()

    def validate_categories(self) -> bool:
        """
        Validate that all expected categories exist.
        """
        found_categories = {
            folder.name
            for folder in self.dataset_path.iterdir()
            if folder.is_dir()
        }

        return found_categories == self.EXPECTED_CATEGORIES
    
    def validate_category_structure(self) -> bool:
        """
        Validate that each category contains the required folders.
        """
        required_folders = {"train", "test", "ground_truth"}

        for category in self.EXPECTED_CATEGORIES:
            category_path = self.dataset_path / category

            if not category_path.exists():
                return False

            found_folders = {
                folder.name
                for folder in category_path.iterdir()
                if folder.is_dir()
            }

            if not required_folders.issubset(found_folders):
                return False

        return True
    
    def validate_images(self) -> bool:
        """
        Validate that all train and test folders contain images.
        """
        for category in self.EXPECTED_CATEGORIES:

            category_path = self.dataset_path / category

            for split in ["train", "test"]:

                split_path = category_path / split

                for defect_folder in split_path.iterdir():

                    if not defect_folder.is_dir():
                        continue

                    images = get_image_files(defect_folder)

                    if len(images) == 0:
                        return False

        return True

    def generate_validation_report(self) -> dict:
        """
        Generate a validation report for the dataset.
        """
        report = {
            "dataset_exists": self.dataset_exists(),
            "categories_valid": self.validate_categories(),
            "folder_structure_valid": self.validate_category_structure(),
            "images_valid": self.validate_images(),
        }

        report["overall_status"] = all(report.values())

        return report