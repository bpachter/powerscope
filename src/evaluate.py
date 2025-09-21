import os, numpy as np, pandas as pd, joblib, torch
from omegaconf import OmegaConf
import hydra
from hydra.utils import to_absolute_path as abspath
from src.utils.io import ensure_dir
from src.data.iso_local_csv import load_local_csv
from src.data.weather_meteostat import fetch_weather
from src.features.build_features import add_time_features, add_holiday_features, build_derived_weather, assemble_feature_frame
from src.models.lightgbm_quantile import predict_quantiles
from src.models.torch_models import QuantileLSTM
from src.calibration.conformal import calibrate_quantiles, apply_adjustment
from src.utils.metrics import pinball_loss, crps_from_quantiles, coverage
import matplotlib.pyplot as plt

def prepare_features(cfg):
    load_df = load_local_csv(abspath(cfg.data.local_csv_path), cfg.data.tz)
    w = fetch_weather(cfg.data.weather.stations, cfg.data.weather.start, cfg.data.weather.end, cfg.data.tz)
    w = w.rename(columns={'temp':'temp','dwpt':'dwpt','rhum':'rhum','wspd':'wspd','prcp':'prcp'})
    w_derived = build_derived_weather(w, cfg.features.temp_lags, cfg.features.rolling_hours, cfg.features.use_degree_hours)
    tfeat = add_time_features(w_derived[['timestamp']], cfg.data.tz, cfg.features.fourier_k)
    hfeat = add_holiday_features(w_derived[['timestamp']], cfg.features.holidays_country, cfg.data.tz)
    df = assemble_feature_frame(load_df, w_derived, tfeat, hfeat)
    return df

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    df = prepare_features(cfg)
    t = pd.to_datetime(df['timestamp'])
    horizon = cfg.data.horizon_hours
    df['y'] = df['load_mw'].shift(-horizon)
    feat_cols = [c for c in df.columns if c not in ['timestamp','load_mw','y']]
    df = df.dropna(subset=['y']).reset_index(drop=True)

    # splits
    train = df[t <= pd.to_datetime(cfg.split.train_end)]
    val   = df[(t > pd.to_datetime(cfg.split.train_end)) & (t <= pd.to_datetime(cfg.split.val_end))]
    test  = df[t > pd.to_datetime(cfg.split.val_end)]

    from sklearn.preprocessing import StandardScaler
    scaler = None

    # load model
    models_dir = cfg.paths.models_dir
    report_dir = os.path.join(cfg.paths.reports_dir, "eval")
    ensure_dir(report_dir)

    if cfg.model == "lightgbm":
        bundle = joblib.load(os.path.join(models_dir, "lightgbm", "model.joblib"))
        models = bundle['models']; scaler = bundle['scaler']; quantiles = bundle['quantiles']; feat_cols = bundle['feat_cols']
        Xtr = scaler.transform(train[feat_cols].values); ytr = train['y'].values
        Xva = scaler.transform(val[feat_cols].values);   yva = val['y'].values
        Xte = scaler.transform(test[feat_cols].values);  yte = test['y'].values
        qs_va, qpred_va = predict_quantiles(models, Xva)
        adj = calibrate_quantiles(yva, qpred_va, qs_va)
        qs_te, qpred_te = predict_quantiles(models, Xte)
        qpred_te_adj = apply_adjustment(qpred_te, qs_te, adj)

    else:
        ckpt = torch.load(os.path.join(models_dir, "lstm", "model.pt"), map_location="cpu")
        quantiles = ckpt['quantiles']; scaler = ckpt['scaler']; hist = ckpt['history']; feat_cols = ckpt['feat_cols']
        model = QuantileLSTM(input_size=len(feat_cols), hidden=128, layers=2, dropout=0.1, quantiles=tuple(quantiles))
        model.load_state_dict(ckpt['state_dict']); model.eval()
        def to_seq(Xmat):
            seqX = []
            for i in range(hist, len(Xmat)):
                seqX.append(Xmat[i-hist:i, :])
            return np.array(seqX)
        Xtr = scaler.transform(train[feat_cols].values); ytr = train['y'].values
        Xva = scaler.transform(val[feat_cols].values);   yva = val['y'].values
        Xte = scaler.transform(test[feat_cols].values);  yte = test['y'].values
        Xva_seq, Xte_seq = to_seq(Xva), to_seq(Xte)
        with torch.no_grad():
            qpred_va = model(torch.tensor(Xva_seq, dtype=torch.float32)).numpy()
            qpred_te = model(torch.tensor(Xte_seq, dtype=torch.float32)).numpy()
        # align targets for seq truncation
        yva = yva[hist:]
        yte = yte[hist:]
        adj = calibrate_quantiles(yva, qpred_va, quantiles)
        qpred_te_adj = apply_adjustment(qpred_te, quantiles, adj)
        qs_te = quantiles

    # metrics
    lo_idx = 0; mid_idx = 1; hi_idx = -1
    pbl_mid = pinball_loss(yte, qpred_te_adj[:, mid_idx], 0.5)
    cov = coverage(yte, qpred_te_adj[:, lo_idx], qpred_te_adj[:, hi_idx])
    crps = crps_from_quantiles(yte, qpred_te_adj, qs_te)
    print(f"pinball@0.5: {pbl_mid:.4f}  coverage(P10-P90): {cov:.3f}  crps: {crps:.4f}")

    # plot fan chart
    ts = test['timestamp'].values[-len(yte):]
    plt.figure(figsize=(12,4))
    plt.plot(ts, yte, label="actual")
    plt.fill_between(ts, qpred_te_adj[:, lo_idx], qpred_te_adj[:, hi_idx], alpha=0.3, label="p10-p90")
    plt.plot(ts, qpred_te_adj[:, mid_idx], label="p50")
    ensure_dir(report_dir)
    outpath = os.path.join(report_dir, "fan_chart.png")
    plt.legend(); plt.tight_layout(); plt.savefig(outpath); print(f"saved {outpath}")

if __name__ == "__main__":
    main()
