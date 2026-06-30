from pathlib import Path

import mlflow


class MLflowLogger:

    def __init__(self, experiment_name):

        db_path = Path("mlflow.db").resolve()

        mlflow.set_tracking_uri(
            f"sqlite:///{db_path}"
        )

        mlflow.set_experiment(experiment_name)

    def start_run(self):
        mlflow.start_run()

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics):
        mlflow.log_metrics(metrics)

    def log_model(self, model, artifact_path):
        mlflow.pytorch.log_model(model, artifact_path)

    def end_run(self):
        mlflow.end_run()