"""Score predicted vote shares against the true ones.

The score is a skill score against a constant baseline:

    score = 1 - mean(JSD(pred, true) ** 0.75) / mean(JSD(prior, true) ** 0.75)

JSD is the Jensen-Shannon divergence in bits, and `prior` is the average vote split over all
clips. 0 means no better than always predicting the average split, 1 means every clip's split is
predicted exactly. Raising each clip's JSD to 0.75 keeps a few very hard clips from dominating.

Usage:
    python evaluate.py TRUE_CSV PRED_CSV
"""
import sys

import numpy as np
import pandas as pd

EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
PRIOR = (0.1424, 0.1291, 0.1195, 0.0694, 0.4389, 0.1007)  # mean vote split of the training clips
POWER = 0.75


def _plogp(a, m):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = a * np.log2(a / m)
    return np.where(a > 0, t, 0.0).sum(axis=1)


def jsd(q, p):
    """Row-wise Jensen-Shannon divergence in bits."""
    q = np.asarray(q, np.float64)
    p = np.asarray(p, np.float64)
    q = q / q.sum(axis=1, keepdims=True)
    q, p = np.broadcast_arrays(q, p)
    m = 0.5 * (q + p)
    return np.maximum(0.5 * _plogp(q, m) + 0.5 * _plogp(p, m), 0.0)


def score(true_shares, pred_shares, prior=PRIOR):
    d_pred = (jsd(pred_shares, true_shares) ** POWER).mean()
    d_prior = (jsd(np.asarray(prior)[None], true_shares) ** POWER).mean()
    return 1.0 if d_pred == 0 else max(0.0, 1.0 - d_pred / d_prior)


if __name__ == "__main__":
    true = pd.read_csv(sys.argv[1])
    pred = pd.read_csv(sys.argv[2]).set_index("id").loc[true.id]
    print(f"score {score(true[EMOTIONS].to_numpy(), pred[EMOTIONS].to_numpy()):.4f}")
