import os, warnings
import numpy as np, pandas as pd
from omegaconf import OmegaConf
import hydra
from hydra.utils import to_absolute_path as abspath
from src.utils.io import ensure_dir
from src.data.iso_local_csv import load_local_csv
from src.data.weather_meteostat import fetch_weather
from src.features.build_features import add_time_features, add_holiday_features, build_derived_weather, assemble_feature_frame
from src.models.lightgbm_quantile import train_quantile_models, predict_quantiles
from src.models.torch_models import QuantileLSTM, quantile_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
warnings.filterwarnings("ignore")

def build_dataset(cfg):
    # load load series
    if cfg.data.loader == "local_csv":
        load_df = load_local_csv(abspath(cfg.data.local_csv_path), cfg.data.tz)
    else:
        raise ValueError("only local_csv implemented in this scaffold; plug pjM/nyiso later")

    # weather
    w = fetch_weather(cfg.data.weather.stations, cfg.data.weather.start, cfg.data.weather.end, cfg.data.tz)
    w = w.rename(columns={'temp':'temp','dwpt':'dwpt','rhum':'rhum','wspd':'wspd','prcp':'prcp'})
    w_derived = build_derived_weather(w, cfg.features.temp_lags, cfg.features.rolling_hours, cfg.features.use_degree_hours)

    # time & holiday
    tfeat = add_time_features(w_derived[['timestamp']].assign(timestamp=w_derived['timestamp']), cfg.data.tz, cfg.features.fourier_k)
    hfeat = add_holiday_features(w_derived[['timestamp']], cfg.features.holidays_country, cfg.data.tz)

    df = assemble_feature_frame(load_df, w_derived, tfeat, hfeat)
    return df

def make_xy(df, horizon):
    # build supervised pairs: predict load at t+h from features at t
    df = df.copy()
    df['y'] = df['load_mw'].shift(-horizon)
    feat_cols = [c for c in df.columns if c not in ['timestamp','load_mw','y']]
    df = df.dropna(subset=['y'])
    X = df[feat_cols].values
    y = df['y'].values
    times = df['timestamp'].values
    return X, y, feat_cols, times

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    ensure_dir(cfg.paths.artifacts_dir)
    df = build_dataset(cfg)
    X, y, feat_cols, times = make_xy(df, cfg.data.horizon_hours)

    # split by time (train/val/test as per config)
    t = pd.to_datetime(df['timestamp'])
    train_mask = t <= pd.to_datetime(cfg.split.train_end)
    val_mask   = (t > pd.to_datetime(cfg.split.train_end)) & (t <= pd.to_datetime(cfg.split.val_end))
    test_mask  = t > pd.to_datetime(cfg.split.val_end)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_val   = scaler.transform(X[val_mask])
    X_test  = scaler.transform(X[test_mask])
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

    if cfg.model == "lightgbm":
        params = dict(
            learning_rate=cfg.lightgbm.learning_rate,
            n_estimators=cfg.lightgbm.n_estimators,
            max_depth=cfg.lightgbm.max_depth,
            num_leaves=cfg.lightgbm.num_leaves,
            min_child_samples=cfg.lightgbm.min_child_samples,
            subsample=cfg.lightgbm.subsample,
            colsample_bytree=cfg.lightgbm.colsample_bytree
        )
        models = train_quantile_models(X_train, y_train, X_val, y_val, params, cfg.lightgbm.quantiles)
        # save
        import joblib, os
        outdir = os.path.join(cfg.paths.models_dir, "lightgbm")
        ensure_dir(outdir)
        joblib.dump({'models': models, 'scaler': scaler, 'feat_cols': feat_cols, 'quantiles': cfg.lightgbm.quantiles}, os.path.join(outdir, "model.joblib"))
        print(f"saved lightgbm quantile models to {outdir}")

    elif cfg.model == "lstm":
        # build sequences for lstm
        T = cfg.data.history_hours
        def to_seq(Xmat, yvec, mask):
            Xm = Xmat[mask]
            ym = yvec[mask]
            seqX, seqY = [], []
            for i in range(T, len(Xm)):
                seqX.append(Xm[i-T:i, :])
                seqY.append(ym[i])
            return np.array(seqX), np.array(seqY)

        Xtr, ytr = to_seq(X_train, y_train, np.ones_like(y_train, dtype=bool))
        Xva, yva = to_seq(np.vstack([X_train[-T:], X_val]), np.hstack([y_train[-T:], y_val]), np.ones(len(y_val), dtype=bool))
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = QuantileLSTM(input_size=Xtr.shape[-1], hidden=cfg.lstm.hidden_size, layers=cfg.lstm.num_layers, dropout=cfg.lstm.dropout, quantiles=tuple(cfg.lstm.quantiles)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lstm.lr)

        train_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
        val_ds   = TensorDataset(torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32))
        train_loader = DataLoader(train_ds, batch_size=cfg.lstm.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=cfg.lstm.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        scaler_path = os.path.join(cfg.paths.models_dir, "lstm")
        ensure_dir(scaler_path)

        # simple training loop
        for epoch in range(cfg.lstm.max_epochs):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                    pred = model(xb)
                    loss = quantile_loss(pred, yb, model.qs)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item()*len(xb)
            epoch_loss /= len(train_loader.dataset)

            # val
            model.eval()
            with torch.no_grad():
                vloss = 0.0
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    vloss += quantile_loss(pred, yb, model.qs).item()*len(xb)
                vloss /= len(val_loader.dataset)
            if (epoch+1)%5==0:
                print(f"epoch {epoch+1} train {epoch_loss:.4f} val {vloss:.4f}")

        # save
        torch.save({
            'state_dict': model.state_dict(),
            'scaler': scaler,
            'feat_cols': feat_cols,
            'quantiles': model.qs,
            'history': cfg.data.history_hours
        }, os.path.join(scaler_path, "model.pt"))
        print(f"saved lstm model to {scaler_path}")
    else:
        raise ValueError("unknown model type")

if __name__ == "__main__":
    main()
