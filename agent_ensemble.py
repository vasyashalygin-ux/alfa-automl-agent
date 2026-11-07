"""Жадный ансамбль (Caruana et al.) по OOF-предсказаниям на общих фолдах."""
from __future__ import annotations

import numpy as np


def caruana(oofs: list[np.ndarray], tests: list[np.ndarray], score_fn, greater: bool,
            n_iter: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Возвращает (веса, oof_blend, test_blend, score). score_fn(oof_blend) -> float."""
    m = len(oofs)
    sign = 1 if greater else -1
    scores = [score_fn(o) for o in oofs]
    best_i = int(np.argmax([sign * s for s in scores]))
    counts = np.zeros(m)
    counts[best_i] = 1
    cur = oofs[best_i].copy()
    best = scores[best_i]
    for it in range(1, n_iter):
        cand = []
        for j in range(m):
            blend = (cur * it + oofs[j]) / (it + 1)
            cand.append(score_fn(blend))
        j = int(np.argmax([sign * s for s in cand]))
        if sign * cand[j] <= sign * best + 1e-12:
            break
        counts[j] += 1
        cur = (cur * it + oofs[j]) / (it + 1)
        best = cand[j]
    w = counts / counts.sum()
    test = sum(wi * t for wi, t in zip(w, tests) if wi > 0)
    oof = sum(wi * o for wi, o in zip(w, oofs) if wi > 0)
    return w, oof, test, best
