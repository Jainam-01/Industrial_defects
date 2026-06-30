from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(labels, predictions, scores):

    return {
        "roc_auc": roc_auc_score(labels, scores),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
        ),
    }