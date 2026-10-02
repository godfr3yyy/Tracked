from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import FEATURES

CLASSES = ["H", "D", "A"]                      # column order for every probability array
LABEL = {c: i for i, c in enumerate(CLASSES)}


def make_models() -> dict:
    return {
        "logreg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(C=0.5, max_iter=2000)),
        "xgboost": make_pipeline(SimpleImputer(strategy="median"),
                                 XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.03, subsample=0.8,
                                               colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                                               objective="multi:softprob", eval_metric="mlogloss",
                                               n_jobs=2, random_state=7)),
    }


def fit_all(train: pd.DataFrame) -> dict:
    y = train["FTR"].map(LABEL)
    models = make_models()
    for m in models.values():
        m.fit(train[FEATURES], y)
    return models


def predict_proba(models: dict, X: pd.DataFrame) -> dict:
    out = {n: m.predict_proba(X[FEATURES]) for n, m in models.items()}
    out["blend"] = (out["logreg"] + out["xgboost"]) / 2
    return out


def market_probs(df: pd.DataFrame) -> np.ndarray:
    """Bookmaker odds -> probabilities with the overround removed. The benchmark to beat."""
    raw = 1.0 / df[["oddsH", "oddsD", "oddsA"]].to_numpy(dtype=float)
    return raw / raw.sum(axis=1, keepdims=True)


def score(y: pd.Series, p: np.ndarray) -> dict:
    yi = y.map(LABEL).to_numpy()
    onehot = np.eye(3)[yi]
    return {"n": len(yi), "accuracy": float((p.argmax(1) == yi).mean()),
            "log_loss": float(log_loss(yi, p, labels=[0, 1, 2])),
            "brier": float(((p - onehot) ** 2).sum(1).mean())}


def walk_forward(feat: pd.DataFrame, test_seasons: list[int], min_train_seasons: int = 5):
    """Retrain before each test season using only earlier seasons, then predict that season blind."""
    parts = []
    for s in test_seasons:
        train, test = feat[feat.season < s], feat[feat.season == s].copy()
        train = train[train.season >= train.season.min() + 1]      # first season has no history -> noisy
        models = fit_all(train)
        for name, p in predict_proba(models, test).items():
            test[[f"p_{name}_{c}" for c in CLASSES]] = p
        parts.append(test)
    return pd.concat(parts, ignore_index=True)
