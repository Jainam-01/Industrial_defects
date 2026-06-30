from src.evaluation.metrics import compute_metrics


class Evaluator:

    def __init__(self, threshold=0.5):

        self.threshold = threshold

    def evaluate(self, labels, scores):

        predictions = [
            int(score >= self.threshold)
            for score in scores
        ]

        return compute_metrics(
            labels,
            predictions,
            scores,
        )