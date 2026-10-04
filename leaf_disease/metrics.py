"""Binary classification metrics, with 'diseased' as the positive class."""


def classification_metrics(y_true: list[int], y_pred: list[int], class_to_idx: dict[str, int]) -> dict:
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Expected matching nonempty truth and prediction lists.")
    positive = class_to_idx["diseased"]
    tp = sum(a == positive and b == positive for a, b in zip(y_true, y_pred))
    fn = sum(a == positive and b != positive for a, b in zip(y_true, y_pred))
    fp = sum(a != positive and b == positive for a, b in zip(y_true, y_pred))
    tn = len(y_true) - tp - fn - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(y_true),
        "diseased_precision": precision,
        "diseased_recall": recall,
        "diseased_f1": f1,
        "confusion_matrix": {"true_diseased": {"pred_diseased": tp, "pred_healthy": fn},
                             "true_healthy": {"pred_diseased": fp, "pred_healthy": tn}},
        "support": {"diseased": tp + fn, "healthy": fp + tn},
    }
