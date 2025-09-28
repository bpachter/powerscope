from fastapi import FastAPI
import os, joblib, torch
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="Load Forecaster", version="0.1")

@app.get("/")
def root():
    return {"message": "PowerScope Load Forecasting API", "model": MODEL_KIND, "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_KIND}

class FeatureVector(BaseModel):
    features: list  # same ordering as training feat_cols

# note: this minimal api expects pre-extracted features for simplicity
# in production you'd encapsulate feature engineering here too

MODEL_KIND = os.getenv("MODEL_KIND","lightgbm")
if MODEL_KIND == "lightgbm":
    bundle = joblib.load("artifacts/models/lightgbm/model.joblib")
    models = bundle['models']; scaler = bundle['scaler']; feat_cols = bundle['feat_cols']; quantiles = bundle['quantiles']
else:
    from sklearn.preprocessing import StandardScaler
    torch.serialization.add_safe_globals([StandardScaler])
    ckpt = torch.load("artifacts/models/lstm/model.pt", map_location="cpu", weights_only=False)
    from src.models.torch_models import QuantileLSTM
    model = QuantileLSTM(len(ckpt['feat_cols']), 128, 2, 0.1, tuple(ckpt['quantiles']))
    model.load_state_dict(ckpt['state_dict']); model.eval()
    scaler = ckpt['scaler']; feat_cols = ckpt['feat_cols']; quantiles = ckpt['quantiles']

@app.post("/predict")
def predict(fv: FeatureVector):
    x = np.array(fv.features, dtype=float).reshape(1, -1)
    x = scaler.transform(x)
    if MODEL_KIND == "lightgbm":
        from src.models.lightgbm_quantile import predict_quantiles
        qs, qpred = predict_quantiles(models, x)
        p = qpred[0].tolist()
    else:
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).repeat(1,128,1)  # naive pad
            out = model(xt).numpy()[0].tolist()
        p = out
    return {
        "quantiles": [float(q) for q in quantiles],
        "pred": [float(x) for x in p],
        "num_features": int(len(feat_cols))
    }
