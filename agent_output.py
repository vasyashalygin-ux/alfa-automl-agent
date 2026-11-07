"""Формирование и проверка файла предсказаний. Запись атомарная: сначала во
временный файл, потом rename — платформа никогда не увидит полуфайл."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from agent_log import log


def build_submission(spec, test: pd.DataFrame, pred: np.ndarray, sample: pd.DataFrame | None,
                     class_weights: np.ndarray | None = None) -> pd.DataFrame:
    ids = test[spec.id_col].values if spec.id_col and spec.id_col in test.columns else np.arange(len(test))
    out = pd.DataFrame({spec.id_col or "id": ids})
    tcols = [c for c in spec.output_cols if c != spec.id_col]
    if spec.task_type == "regression":
        v = np.asarray(pred, float)
        fill = np.nanmedian(v) if np.isfinite(v).any() else 0.0
        v = np.where(np.isfinite(v), v, fill)
        out[spec.target] = v
    else:
        proba = np.asarray(pred, float)
        proba = np.where(np.isfinite(proba), proba, 1.0 / proba.shape[1])
        classes = spec.classes
        if spec.output_mode == "label":
            w = class_weights if class_weights is not None else np.ones(proba.shape[1])
            lab = (proba * w[None, :]).argmax(1)
            out[spec.target] = [classes[i] for i in lab]
        elif spec.output_mode == "proba_single":
            pi = classes.index(spec.positive_class)
            out[tcols[0] if tcols else spec.target] = np.clip(proba[:, pi], 0, 1)
        else:  # proba_multi
            s = proba / np.clip(proba.sum(1, keepdims=True), 1e-12, None)
            mapping = match_class_columns(tcols, classes)
            if mapping is None:
                mapping = {str(c): i for i, c in enumerate(classes)}
            for c, i in mapping.items():
                out[c] = s[:, i]
    # порядок и названия колонок — как в шаблоне
    if sample is not None and all(c in out.columns for c in sample.columns):
        out = out[list(sample.columns)]
    return out


def match_class_columns(cols: list, classes: list) -> dict | None:
    """Колонка шаблона -> индекс класса: точное имя, затем «префикс_класс» / «класс» в конце имени."""
    if not cols or len(cols) != len(classes):
        return None
    names = [str(c) for c in classes]
    res = {}
    for col in cols:
        cs = str(col)
        hit = [i for i, n in enumerate(names) if cs == n]
        if not hit:
            hit = [i for i, n in enumerate(names) if cs.lower() == n.lower()]
        if not hit:
            import re as _re
            hit = [i for i, n in enumerate(names)
                   if _re.search(rf"(^|[^0-9A-Za-zА-Яа-я]){_re.escape(n.lower())}$", cs.lower())]
        if len(hit) != 1:
            return None
        res[cs] = hit[0]
    return res if len(set(res.values())) == len(classes) else None


def conform_to_sample(spec, out: pd.DataFrame, sample: pd.DataFrame | None) -> pd.DataFrame:
    """Приводит колонки ответа к шаблону: регистр имён, id из шаблона (если в test его нет),
    другое имя колонки id/таргета. Платформа сверяет формат строго — это частая причина нуля."""
    if sample is None or not len(sample.columns):
        return out
    out = out.copy()
    idc = spec.id_col or "id"
    scols = [str(c) for c in sample.columns]
    if spec.id_col is None and len(sample) == len(out) and scols[0] not in out.columns:
        out = out.drop(columns=[idc], errors="ignore")
        out.insert(0, scols[0], sample.iloc[:, 0].values)
    lower_out = {str(c).lower(): c for c in out.columns}
    ren = {}
    for c in scols:
        oc = lower_out.get(c.lower())
        if c not in out.columns and oc is not None and oc not in scols:
            ren[oc] = c
    out = out.rename(columns=ren)
    if scols[0] not in out.columns and idc in out.columns and idc not in scols:
        out = out.rename(columns={idc: scols[0]})
    miss = [c for c in scols if c not in out.columns]
    extra = [c for c in out.columns if c not in scols]
    if miss and len(miss) == len(extra):
        out = out.rename(columns=dict(zip(extra, miss)))
    if all(c in out.columns for c in scols):
        out = out[scols]
    return out


def validate(spec, sub: pd.DataFrame, test: pd.DataFrame, id_name: str | None = None) -> list[str]:
    errs = []
    if len(sub) != len(test):
        errs.append(f"rows {len(sub)} != {len(test)}")
    idc = id_name or spec.id_col or "id"
    if idc in sub.columns:
        if sub[idc].duplicated().any():
            errs.append("duplicated ids")
        if spec.id_col and spec.id_col in test.columns and not (
                sub[idc].astype(str).values == test[spec.id_col].astype(str).values).all():
            errs.append("id order mismatch")
    vals = sub.drop(columns=[idc], errors="ignore")
    if vals.isna().any().any():
        errs.append("NaN in predictions")
    num = vals.select_dtypes("number")
    if len(num.columns) and not np.isfinite(num.values).all():
        errs.append("inf in predictions")
    if spec.output_mode == "label" and spec.target in vals.columns:
        bad = set(vals[spec.target]) - set(spec.classes)
        if bad:
            errs.append(f"unknown labels {list(bad)[:5]}")
    return errs


def write_submission(spec, sub: pd.DataFrame, test: pd.DataFrame, tag: str = "", id_name: str | None = None) -> bool:
    errs = validate(spec, sub, test, id_name)
    if errs:
        log("САБМИТ НЕ ПРОШЁЛ ПРОВЕРКУ:", errs)
        return False
    path = Path(spec.output_path)
    # платформа проверяет submission.csv рядом с task.md — пишем его всегда,
    # а если условие называет другой файл, дублируем и под этим именем
    targets = [path.with_name("submission.csv")]
    if path.name != "submission.csv":
        targets.append(path)
    for p in targets:
        tmp = p.with_name(p.name + ".tmp")
        sub.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, p)
    log(f"submission записан ({tag}): {', '.join(p.name for p in targets)}, {sub.shape}")
    return True


def constant_pred(spec, train: pd.DataFrame, test: pd.DataFrame):
    y = train[spec.target]
    if spec.task_type == "regression":
        v = pd.to_numeric(y, errors="coerce")
        return np.full(len(test), float(v.median()) if v.notna().any() else 0.0)
    k = len(spec.classes)
    freq = np.array([(y == c).mean() for c in spec.classes])
    return np.tile(freq, (len(test), 1)) if k else np.zeros((len(test), 1))
