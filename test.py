from src.data.data_validation import DataValidation

validator = DataValidation("data/raw/mvtec_ad")

assert validator.dataset_exists()
assert validator.validate_categories()
assert validator.validate_category_structure()
assert validator.validate_images()

print("\nAll validation tests passed successfully!\n")

report = validator.generate_validation_report()

for key, value in report.items():
    print(f"{key}: {value}")