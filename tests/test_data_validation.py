from src.data.data_ingestion import DataIngestion


def test_dataset_exists():
    ingestion = DataIngestion("data/raw/mvtec_ad")
    assert ingestion.dataset_exists()


def test_categories():
    ingestion = DataIngestion("data/raw/mvtec_ad")
    assert len(ingestion.get_categories()) == 15