"""Профиль данных для LLM и для автоматических решений.

Главное здесь — adversarial validation: если train и test легко различимы,
случайный K-fold врёт, и нужна валидация, имитирующая сдвиг (например,
по году, как в задаче про здания).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import agent_config as config
from agent_log import log


def column_profile(train: pd.DataFrame, test: pd.DataFrame, target: str, id_col: str | None) -> list[dict]:
    rows = []
    for c in train.columns:
        if c == target:
            continue
        s = train[c]
        d = {"col": c, "dtype": str(s.dtype), "na": round(float(s.isna().mean()), 3),
             "nunique": int(s.nunique(dropna=True))}
        if c == id_col:
            d["role"] = "id"
        if pd.api.types.is_numeric_dtype(s):
            q = s.describe()
            d.update(min=_r(q.get("min")), mean=_r(q.get("mean")), max=_r(q.get("max")))
            if c in test.columns and pd.api.types.is_numeric_dtype(test[c]):
                d.update(test_min=_r(test[c].min()), test_max=_r(test[c].max()))
        else:
            d["top"] = [str(v)[:30] for v in s.value_counts().index[:4]]
            if c in test.columns:
                d["unseen_in_test"] = int(len(set(test[c].dropna().astype(str)) - set(s.dropna().astype(str))))
        rows.append(d)
    return rows


def _r(v):
    try:
        v = float(v)
        return round(v, 4) if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def target_profile(y: pd.Series, task_type: str) -> dict:
    if task_type == "regression":
        yv = y.astype(float)
        return {"mean": _r(yv.mean()), "std": _r(yv.std()), "min": _r(yv.min()), "max": _r(yv.max()),
                "skew": _r(yv.skew()), "q01": _r(yv.quantile(.01)), "q50": _r(yv.median()),
                "q99": _r(yv.quantile(.99)), "n_missing": int(yv.isna().sum())}
    vc = y.value_counts()
    return {"n_classes": int(len(vc)), "counts": {str(k): int(v) for k, v in vc.items()},
            "imbalance_ratio": _r(vc.max() / max(1, vc.min()))}


def detect_shift_columns(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> list[dict]:
    """Колонки, по которым test лежит «вне» train: новые значения или диапазон за пределами."""
    out = []
    for c in cols:
        if c not in test.columns:
            continue
        a, b = train[c].dropna(), test[c].dropna()
        if len(a) == 0 or len(b) == 0:
            continue
        if not pd.api.types.is_numeric_dtype(a):
            ta, tb = _as_time(a), _as_time(b)
            if ta is not None and tb is not None:
                if tb.min() >= ta.max():
                    kind = "time" if a.nunique() <= 50 else "time_continuous"
                    out.append({"col": c, "kind": kind, "train_values": int(a.nunique()),
                                "test_values": int(b.nunique()), "test_after_train": True, "datetime": True})
                continue
        ua, ub = set(a.unique()), set(b.unique())
        if pd.api.types.is_numeric_dtype(a) and len(ua) <= 50:
            inter = len(ub & ua) / max(1, len(ub))
            if inter == 0 and len(ua) >= 2:
                after = float(b.min()) > float(a.max())
                out.append({"col": c, "kind": "time" if after else "group", "train_values": len(ua),
                            "test_values": len(ub), "test_after_train": after})
        elif pd.api.types.is_numeric_dtype(a) and float(b.min()) > float(a.quantile(0.99)) and len(ua) > 50:
            out.append({"col": c, "kind": "time_continuous", "test_after_train": True})
    return out


def _as_time(s: pd.Series):
    """Строковая колонка как время (секунды) или None, если это не даты."""
    sample = s.astype(str).head(300)
    if not sample.str.contains(r"\d{2,4}[-/.]\d{1,2}[-/.]\d{1,4}", regex=True).mean() > 0.9:
        return None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().mean() < 0.9:
        return None
    return (dt.astype("int64") // 10**9).where(dt.notna()).dropna().astype(float)


def adversarial_validation(Xtr: pd.DataFrame, Xte: pd.DataFrame, max_rows: int = 40000) -> dict:
    """AUC классификатора train-vs-test и самые «выдающие» признаки."""
    try:
        import lightgbm as lgb
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score

        rng = np.random.RandomState(config.SEED)
        a = Xtr.sample(min(len(Xtr), max_rows), random_state=rng)
        b = Xte.sample(min(len(Xte), max_rows), random_state=rng)
        X = pd.concat([a, b], ignore_index=True)
        y = np.r_[np.zeros(len(a)), np.ones(len(b))]
        oof = np.zeros(len(y))
        imp = np.zeros(X.shape[1])
        for tr, va in StratifiedKFold(3, shuffle=True, random_state=0).split(X, y):
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, verbose=-1,
                                   n_jobs=config.N_JOBS).fit(X.iloc[tr], y[tr])
            oof[va] = m.predict_proba(X.iloc[va])[:, 1]
            imp += m.booster_.feature_importance("gain")
        auc = float(roc_auc_score(y, oof))
        order = np.argsort(-imp)[:8]
        tot = imp.sum() or 1
        top = [(X.columns[i], round(float(imp[i] / tot), 3)) for i in order]
        return {"auc": round(auc, 4), "top_features": top}
    except Exception as e:  # noqa: BLE001
        log("adversarial validation failed:", repr(e))
        return {"auc": None, "top_features": []}


def profile_text(spec, cols: list[dict], tgt: dict, adv: dict, shifts: list[dict], max_cols: int = 120) -> str:
    """Компактное текстовое представление профиля для промпта."""
    lines = [f"rows train={spec.n_train}, test={spec.n_test}, features={len(cols)}",
             f"target `{spec.target}` ({spec.task_type}): {tgt}",
             f"adversarial validation train-vs-test AUC={adv.get('auc')} top={adv.get('top_features')}",
             f"shift columns (test values unseen in train): {shifts}", "columns:"]
    for d in cols[:max_cols]:
        lines.append("  " + ", ".join(f"{k}={v}" for k, v in d.items()))
    if len(cols) > max_cols:
        lines.append(f"  ... ещё {len(cols) - max_cols} колонок")
    return "\n".join(lines)
