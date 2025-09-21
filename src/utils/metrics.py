import numpy as np

def pinball_loss(y_true, y_pred, q: float):
    # quantile loss
    u = y_true - y_pred
    return np.mean(np.maximum(q*u, (q-1)*u))

def crps_from_quantiles(y_true, q_preds, quantiles):
    # approximate crps via discrete quantiles
    # expects q_preds shape [n, Q] in ascending quantiles
    crps = 0.0
    for i, q in enumerate(quantiles):
        crps += pinball_loss(y_true, q_preds[:, i], q)
    return crps / len(quantiles)

def coverage(y_true, lo, hi):
    # fraction of true within [lo, hi]
    return np.mean((y_true >= lo) & (y_true <= hi))
