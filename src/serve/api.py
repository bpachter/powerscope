from fastapi import FastAPI
import os, joblib, torch
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="load forecaster", version="0.1")

class FeatureVector(BaseModel):
    features: list  # same ordering as training feat_cols

# note: this minimal api expects pre-extracted features for simplicity
# in production you'd encapsulate feature engineering here too

MODEL_KIND = os.getenv("MODEL_KIND","lightgbm")
if MODEL_KIND == "lightgbm":
    bundle = joblib.load("artifacts/models/lightgbm/model.joblib")
    models = bundle['models']; scaler = bundle['scaler']; feat_cols = bundle['feat_cols']; quantiles = bundle['quantiles']
else:
    ckpt = torch.load("artifacts/models/lstm/model.pt", map_location="cpu")
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
    return {"quantiles": quantiles, "pred": p, "feat_cols": feat_cols}
