from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="Test API")

class FeatureVector(BaseModel):
    features: list

# Load model at startup
print("Loading model...")
bundle = joblib.load("artifacts/models/lightgbm/model.joblib")
models = bundle['models']
scaler = bundle['scaler']
quantiles = bundle['quantiles']
print(f"Model loaded: {len(models)} quantile models")

@app.get("/")
def root():
    return {"status": "ok", "quantiles": len(models)}

@app.post("/predict")
def predict(fv: FeatureVector):
    try:
        # Convert to numpy
        x = np.array(fv.features, dtype=float).reshape(1, -1)
        print(f"Input shape: {x.shape}")

        # Scale features
        x_scaled = scaler.transform(x)
        print("Features scaled")

        # Predict with each quantile model
        predictions = []
        for q in quantiles:
            pred = models[q].predict(x_scaled)[0]
            predictions.append(float(pred))
            print(f"Q{q}: {pred}")

        return {
            "quantiles": [float(q) for q in quantiles],
            "predictions": predictions,
            "status": "success"
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e), "status": "failed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)