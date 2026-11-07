"""Схемы валидации. Одни и те же фолды используются во всех экспериментах,
поэтому сравнение гипотез честное, а OOF-предсказания можно ансамблировать."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (GroupKFold, KFold, StratifiedGroupKFold,
                                     StratifiedKFold)

import agent_config as config


def make_folds(train: pd.DataFrame, y: np.ndarray, task_type: str, scheme: str,
               n_splits: int = 5) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    n = len(train)
    idx = np.arange(n)
    if n > 400_000:
        n_splits = min(n_splits, 3)
    cls = task_type != "regression"

    if scheme.startswith(("group:", "time:")):
        kind, col = scheme.split(":", 1)
        if col in train.columns:
            g = train[col].astype(str).fillna("NA").values if kind == "group" else train[col].values
            if kind == "time":
                # растущее окно: обучаемся на прошлом, валидируемся на следующем периоде
                if not pd.api.types.is_numeric_dtype(train[col]):  # даты строками -> секунды
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        dt = pd.to_datetime(train[col], errors="coerce")
                    g = (dt.astype("int64") // 10**9).where(dt.notna()).astype(float).values
                gs = pd.Series(g)
                vals = np.sort(gs.dropna().unique())
                if len(vals) > 30:  # непрерывное время -> квантильные блоки
                    qs = np.quantile(gs.dropna(), np.linspace(0, 1, n_splits + 2))
                    blocks = np.digitize(gs.fillna(gs.min()).values, qs[1:-1])
                    vals = np.unique(blocks)
                    g = blocks
                folds = []
                n_time = min(3, n_splits)  # последние 3 периода: в ранних слишком мало истории для обучения
                for v in vals[-min(n_time, len(vals) - 1):]:
                    tr = idx[g < v]
                    va = idx[g == v]
                    if len(tr) and len(va):
                        folds.append((tr, va))
                if folds:
                    return folds, f"time-forward по `{col}` ({len(folds)} фолдов)"
            ng = len(pd.unique(g))
            if ng >= 2:
                k = min(n_splits, ng)
                if cls:
                    try:
                        sp = StratifiedGroupKFold(k, shuffle=True, random_state=config.SEED)
                        folds = list(sp.split(train, y, g))
                        return folds, f"StratifiedGroupKFold({k}) по `{col}`"
                    except ValueError:
                        pass
                folds = list(GroupKFold(k).split(train, y, g))
                return folds, f"GroupKFold({k}) по `{col}`"
        scheme = "stratified" if cls else "kfold"

    if cls:
        import warnings
        with warnings.catch_warnings():  # редкий класс < n_splits — только предупреждение
            warnings.simplefilter("ignore")
            sp = StratifiedKFold(n_splits, shuffle=True, random_state=config.SEED)
            return list(sp.split(train, y)), f"StratifiedKFold({n_splits})"
    sp = KFold(n_splits, shuffle=True, random_state=config.SEED)
    return list(sp.split(train)), f"KFold({n_splits})"


def oof_mask(folds, n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    for _, va in folds:
        m[va] = True
    return m
