import numpy as np
import lightgbm as lgb

# note: we train one model per quantile for simplicity and stability

def train_quantile_models(X_train, y_train, X_val, y_val, params, quantiles):
    models = {}
    for q in quantiles:
        qp = params.copy()
        qp['objective'] = 'quantile'
        qp['alpha'] = q
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        model = lgb.train(qp, dtrain, valid_sets=[dtrain, dval], valid_names=['train','val'],
                          callbacks=[lgb.log_evaluation(200)])
        models[q] = model
    return models

def predict_quantiles(models, X):
    qs = sorted(models.keys())
    preds = np.column_stack([models[q].predict(X) for q in qs])
    # enforce monotonicity to avoid p10>p50 etc.
    preds = np.maximum.accumulate(preds, axis=1)
    return qs, preds
