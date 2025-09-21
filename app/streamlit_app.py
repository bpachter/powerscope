import streamlit as st
import pandas as pd, numpy as np, joblib, os, matplotlib.pyplot as plt

st.set_page_config(page_title="Load Forecast Fan Chart", layout="wide")
st.title("Probabilistic Load Forecast")

kind = st.sidebar.selectbox("model", ["lightgbm","lstm"])
if kind=="lightgbm":
    bundle = joblib.load("artifacts/models/lightgbm/model.joblib")
    models = bundle['models']; scaler = bundle['scaler']; feat_cols = bundle['feat_cols']; quantiles = bundle['quantiles']
else:
    import torch
    from src.models.torch_models import QuantileLSTM
    ckpt = torch.load("artifacts/models/lstm/model.pt", map_location="cpu")
    model = QuantileLSTM(len(ckpt['feat_cols']), 128, 2, 0.1, tuple(ckpt['quantiles']))
    model.load_state_dict(ckpt['state_dict']); model.eval()
    scaler = ckpt['scaler']; feat_cols = ckpt['feat_cols']; quantiles = ckpt['quantiles']

st.markdown("upload a feature csv (with same columns as training) to visualize forecast bands.")
file = st.file_uploader("feature csv", type=["csv"])
if file is not None:
    df = pd.read_csv(file, parse_dates=['timestamp'])
    X = scaler.transform(df[feat_cols].values)
    if kind=="lightgbm":
        from src.models.lightgbm_quantile import predict_quantiles
        qs, qpred = predict_quantiles(models, X)
    else:
        import torch
        seq_len = 128
        Xseq = []
        for i in range(seq_len, len(X)):
            Xseq.append(X[i-seq_len:i,:])
        Xseq = np.array(Xseq)
        with torch.no_grad():
            qpred = model(torch.tensor(Xseq, dtype=torch.float32)).numpy()
        df = df.iloc[seq_len:].reset_index(drop=True)
    lo, mid, hi = qpred[:,0], qpred[:,1], qpred[:,2]
    ts = df['timestamp'].values
    fig, ax = plt.subplots(figsize=(14,4))
    ax.fill_between(ts, lo, hi, alpha=0.3, label="p10-p90")
    ax.plot(ts, mid, label="p50")
    if 'load_mw' in df.columns:
        ax.plot(ts, df['load_mw'].values, label="actual")
    ax.legend(); st.pyplot(fig)
else:
    st.info("waiting for feature csv...")
