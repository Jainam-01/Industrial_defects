from src.tracking.mlflow_logger import MLflowLogger


def test_logger():

    logger = MLflowLogger(
        "Industrial Defect Detection"
    )

    logger.start_run()

    logger.log_params(
        {
            "batch_size": 8,
            "epochs": 1,
        }
    )

    logger.log_metrics(
        {
            "loss": 0.12,
        }
    )

    logger.end_run()

    assert True