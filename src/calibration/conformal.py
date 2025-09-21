import numpy as np
import pandas as pd

# split-conformal for quantiles: adjust offsets to reach desired coverage

def calibrate_quantiles(y_val, qpred_val, quantiles):
    # computes additive adjustments for each quantile to hit empirical coverage targets
    adj = {}
    y = np.asarray(y_val)
    Q = np.asarray(qpred_val)  # shape [n, len(q)]
    qs = list(quantiles)
    # simple median shift for p50
    median_idx = qs.index(0.5)
    adj[0.5] = np.median(y - Q[:, median_idx])
    # lower / upper via empirical residual quantiles
    lo_idx = 0
    hi_idx = -1
    lo_res = y - Q[:, lo_idx]
    hi_res = y - Q[:, hi_idx]
    adj[qs[lo_idx]] = np.quantile(lo_res, 0.1)   # target: 10% below p10
    adj[qs[hi_idx]] = np.quantile(hi_res, 0.9)   # target: 10% above p90
    return adj  # dict quantile->additive_offset

def apply_adjustment(qpred, quantiles, adj):
    qs = list(quantiles)
    out = qpred.copy()
    for i,q in enumerate(qs):
        if q in adj:
            out[:, i] = out[:, i] + adj[q]
    # enforce monotonicity
    out = np.maximum.accumulate(out, axis=1)
    return out
