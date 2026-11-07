"""Метрики, их направление и превращение вероятностей в ответы.

Агент не знает заранее, какая метрика будет на скрытых задачах, поэтому
здесь собраны наиболее распространённые метрики регрессии и классификации.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn import metrics as skm


@dataclass
class Metric:
    name: str
    greater_is_better: bool
    kind: str  # "regression" | "classification"
    needs: str  # "value" | "label" | "proba"
    fn: Callable

    def score(self, y_true, pred) -> float:
        return float(self.fn(y_true, pred))

    def better(self, a: float, b: float) -> bool:
        """a лучше b?"""
        if b is None or not np.isfinite(b):
            return np.isfinite(a)
        return a > b if self.greater_is_better else a < b

    def gain(self, new: float, old: float) -> float:
        """Улучшение (положительное = лучше) в единицах метрики."""
        return (new - old) if self.greater_is_better else (old - new)


def _rmse(y, p):
    return np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def _rmsle(y, p):
    p = np.clip(np.asarray(p, float), 0, None)
    return np.sqrt(np.mean((np.log1p(np.clip(np.asarray(y, float), 0, None)) - np.log1p(p)) ** 2))


def _mape(y, p):
    y = np.asarray(y, float)
    return np.mean(np.abs((y - p) / np.where(np.abs(y) < 1e-12, 1e-12, np.abs(y))))


def _smape(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    den = np.abs(y) + np.abs(p)
    return float(np.mean(np.where(den < 1e-12, 0.0, 2 * np.abs(y - p) / np.where(den < 1e-12, 1, den))) * 100)


def _brier(y, proba):
    proba = np.asarray(proba, float)
    y = np.asarray(y).astype(int)
    if proba.ndim == 1:
        return float(np.mean((proba - y) ** 2))
    if proba.shape[1] == 2:
        return float(np.mean((proba[:, 1] - y) ** 2))
    onehot = np.eye(proba.shape[1])[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _avg(y):
    return "binary" if len(np.unique(y)) <= 2 else "macro"


def _roc_auc(y, proba):
    proba = np.asarray(proba)
    if proba.ndim == 2 and proba.shape[1] == 2:
        return skm.roc_auc_score(y, proba[:, 1])
    if proba.ndim == 1:
        return skm.roc_auc_score(y, proba)
    return skm.roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=np.arange(proba.shape[1]))


def _logloss(y, proba):
    proba = np.asarray(proba)
    if proba.ndim == 1:
        proba = np.column_stack([1 - proba, proba])
    return skm.log_loss(y, np.clip(proba, 1e-15, 1 - 1e-15), labels=np.arange(proba.shape[1]))


def _binary_f1(y, p):
    return skm.f1_score(y, p, average="binary" if len(np.unique(y)) <= 2 else "macro")


METRICS: dict[str, Metric] = {
    "rmse": Metric("rmse", False, "regression", "value", _rmse),
    "mse": Metric("mse", False, "regression", "value", lambda y, p: _rmse(y, p) ** 2),
    "mae": Metric("mae", False, "regression", "value", skm.mean_absolute_error),
    "rmsle": Metric("rmsle", False, "regression", "value", _rmsle),
    "mape": Metric("mape", False, "regression", "value", _mape),
    "medae": Metric("medae", False, "regression", "value", skm.median_absolute_error),
    "r2": Metric("r2", True, "regression", "value", skm.r2_score),
    "smape": Metric("smape", False, "regression", "value", _smape),
    "precision": Metric("precision", True, "classification", "label",
                        lambda y, p: skm.precision_score(y, p, average=_avg(y), zero_division=0)),
    "recall": Metric("recall", True, "classification", "label",
                     lambda y, p: skm.recall_score(y, p, average=_avg(y), zero_division=0)),
    "brier": Metric("brier", False, "classification", "proba", _brier),
    "macro_f1": Metric("macro_f1", True, "classification", "label",
                       lambda y, p: skm.f1_score(y, p, average="macro")),
    "weighted_f1": Metric("weighted_f1", True, "classification", "label",
                          lambda y, p: skm.f1_score(y, p, average="weighted")),
    "micro_f1": Metric("micro_f1", True, "classification", "label",
                       lambda y, p: skm.f1_score(y, p, average="micro")),
    "f1": Metric("f1", True, "classification", "label", _binary_f1),
    "accuracy": Metric("accuracy", True, "classification", "label", skm.accuracy_score),
    "balanced_accuracy": Metric("balanced_accuracy", True, "classification", "label",
                                skm.balanced_accuracy_score),
    "mcc": Metric("mcc", True, "classification", "label", skm.matthews_corrcoef),
    "kappa": Metric("kappa", True, "classification", "label",
                    lambda y, p: skm.cohen_kappa_score(y, p, weights="quadratic")),
    "roc_auc": Metric("roc_auc", True, "classification", "proba", _roc_auc),
    "logloss": Metric("logloss", False, "classification", "proba", _logloss),
    "pr_auc": Metric("pr_auc", True, "classification", "proba",
                     lambda y, p: skm.average_precision_score(y, np.asarray(p)[:, 1] if np.ndim(p) == 2 else p)),
}


def get_metric(name: str) -> Metric:
    return METRICS[name]


# ---------------------------------------------------------------- решающее правило

def labels_from_proba(proba: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    proba = np.asarray(proba)
    if weights is not None:
        proba = proba * weights[None, :]
    return proba.argmax(1)


def robust_class_weights(y: np.ndarray, proba: np.ndarray, metric: Metric, max_seconds: float = 120.0,
                         seed: int = 0) -> tuple[np.ndarray | None, float, float]:
    """Подбор множителей классов с проверкой на отложенной половине.

    Делим строки пополам (со стратификацией), подбираем веса на одной половине и меряем прирост
    на другой, и наоборот. Если в среднем прироста на отложенных данных нет — веса не применяем
    (возвращаем None): иначе подбор просто подгоняется под шум OOF.
    Возвращает (веса или None, оценка с итоговым правилом, средний прирост на отложенных половинах).
    """
    from sklearn.model_selection import StratifiedKFold
    import warnings
    import agent_config as _cfg

    y = np.asarray(y)
    base = metric.score(y, labels_from_proba(proba))
    if not getattr(_cfg, "THRESH_VETO", True):
        # без проверки на отложенной половине: пороги применяются всегда (как в лучших запусках на платформе);
        # проверка с малым запасом случайно отключала пороги, и классификация проваливалась
        w, s = optimize_class_weights(y, proba, metric, n_rounds=3, max_seconds=max_seconds / 2)
        return w, s, float("nan")
    gains = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        halves = list(StratifiedKFold(2, shuffle=True, random_state=seed).split(proba, y))
    for a, b in halves:
        w, _ = optimize_class_weights(y[a], proba[a], metric, n_rounds=2, max_seconds=max_seconds / 4)
        gains.append(metric.gain(metric.score(y[b], labels_from_proba(proba[b], w)),
                                 metric.score(y[b], labels_from_proba(proba[b]))))
    held = float(np.mean(gains))
    if held <= 0:
        return None, base, held
    w, s = optimize_class_weights(y, proba, metric, n_rounds=3, max_seconds=max_seconds / 2)
    return w, s, held


def optimize_class_weights(y: np.ndarray, proba: np.ndarray, metric: Metric,
                           n_rounds: int = 3, max_seconds: float = 120.0, grid_n: int = 41) -> tuple[np.ndarray, float]:
    """Подбирает множители классов к вероятностям, максимизируя label-метрику на OOF.

    Покоординатный поиск по логарифмической сетке. Для бинарной задачи это
    эквивалентно подбору порога. Ограничен по времени.
    """
    import time

    t0 = time.time()
    k = proba.shape[1]
    w = np.ones(k)
    best = metric.score(y, labels_from_proba(proba, w))
    import agent_config as _cfg
    lo, hi = getattr(_cfg, "CLASS_WEIGHT_RANGE", (0.05, 20.0))
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), grid_n))
    for _ in range(n_rounds):
        improved = False
        for c in range(k):
            if time.time() - t0 > max_seconds:
                return w, best
            for g in grid:
                w2 = w.copy()
                w2[c] = g
                s = metric.score(y, labels_from_proba(proba, w2))
                if metric.better(s, best) and abs(s - best) > 1e-9:
                    best, w, improved = s, w2, True
        if not improved:
            break
    return w, best


def em_prior_shift(p_test: np.ndarray, p_oof: np.ndarray, max_iter: int = 200, tol: float = 1e-7):
    """Поправка на сдвиг долей классов (label shift), Saerens et al., 2002.

    p_oof — out-of-fold вероятности на train: их среднее — «доли классов, которые видит модель».
    EM по тесту оценивает доли классов в тесте (без ответов) и пересчитывает вероятности.
    Возвращает (скорректированные вероятности, доли в train по модели, оценённые доли в test).
    """
    p_test = np.clip(np.asarray(p_test, float), 1e-12, None)
    p_test = p_test / p_test.sum(1, keepdims=True)
    q_tr = np.clip(np.asarray(p_oof, float).mean(0), 1e-9, None)
    q_tr = q_tr / q_tr.sum()
    pi = q_tr.copy()
    adj = p_test
    for _ in range(max_iter):
        adj = p_test * (pi / q_tr)[None, :]
        adj = adj / adj.sum(1, keepdims=True)
        new = adj.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
    return adj, q_tr, pi
