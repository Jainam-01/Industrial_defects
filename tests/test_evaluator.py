from src.evaluation.evaluator import Evaluator


def test_evaluator():

    labels = [0, 0, 1, 1]

    scores = [0.1, 0.2, 0.8, 0.9]

    evaluator = Evaluator()

    metrics = evaluator.evaluate(
        labels,
        scores,
    )

    assert "roc_auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics