import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
    brier_score_loss,
)


def metrics(y, p, threshold=0.5):
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return dict(
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
        roc_auc=float(roc_auc_score(y, p)) if both else None,
        average_precision=float(average_precision_score(y, p)) if both else None,
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        brier=float(brier_score_loss(y, p)),
        false_positive_rate=float(fp / (tn + fp)) if tn + fp else None,
        false_negative_rate=float(fn / (tp + fn)) if tp + fn else None,
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
    )


def threshold_study(y, p):
    rows = [
        dict(threshold=float(t), **metrics(y, p, t))
        for t in np.linspace(0.05, 0.95, 91)
    ]
    # F1 is a transparent educational proxy; no lending cost matrix is available.
    winner = max(rows, key=lambda r: (r["f1"], r["threshold"]))
    return winner["threshold"], rows


def bootstrap(y, p, threshold, groups, repeats=300, seed=42):
    rng = np.random.default_rng(seed)
    keys = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in keys}
    scores = []
    for _ in range(repeats):
        idx = np.concatenate(
            [members[g] for g in rng.choice(keys, len(keys), replace=True)]
        )
        if len(np.unique(y[idx])) == 2:
            scores.append(metrics(y[idx], p[idx], threshold))
    return {
        k: np.quantile([s[k] for s in scores], [0.025, 0.975]).tolist()
        for k in scores[0]
        if k != "confusion_matrix"
    }
