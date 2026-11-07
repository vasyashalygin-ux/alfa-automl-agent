"""Предобработка и генерация признаков.

FeatureConfig описывает набор признаков декларативно — его можно менять
гипотезами, логировать и воспроизводить. Код признаков от LLM исполняется
в отдельном процессе с таймаутом и никогда не видит таргет.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

import agent_config as config
from agent_log import log


@dataclass
class FeatureConfig:
    freq_enc: bool = False         # частотное кодирование категорий
    entity: bool = False           # история таргета по повторяющейся сущности (считается в fit_cv)
    row_stats: bool = False        # агрегаты по строке
    group_stats: bool = False      # отклонения числовых от среднего по категории
    drop_cols: list = field(default_factory=list)
    llm_code: list = field(default_factory=list)  # список фрагментов кода add_features(df)

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    def copy(self, **kw) -> "FeatureConfig":
        d = asdict(self)
        d.update(kw)
        return FeatureConfig(**d)


# ------------------------------------------------------------------ базовая предобработка

def base_frame(train: pd.DataFrame, test: pd.DataFrame, target: str, id_col: str | None,
               other_targets: list[str]) -> tuple[pd.DataFrame, list[str], dict]:
    """Объединяет train+test без таргета и приводит типы. Возвращает (df, cat_cols, info)."""
    drop = [c for c in [target, id_col, *other_targets] if c]
    common = [c for c in train.columns if c not in drop and c in test.columns]
    df = pd.concat([train[common], test[common]], ignore_index=True)
    info = {"datetime": [], "text": [], "numeric_from_obj": [], "dropped_const": []}
    out = {}
    for c in common:
        s = df[c]
        if s.dtype == bool:
            out[c] = s.astype(float)
            continue
        if pd.api.types.is_numeric_dtype(s):
            out[c] = s.astype(float) if s.dtype != float else s
            continue
        s = s.astype("string")
        num = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
        nn = s.notna().sum()
        if nn and num.notna().sum() / nn > 0.97:
            out[c] = num.astype(float)
            info["numeric_from_obj"].append(c)
            continue
        sample = s.dropna().head(500)
        if len(sample) and sample.str.contains(r"\d{2,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}:\d{2}", regex=True).mean() > 0.9:
            dt = pd.to_datetime(s, errors="coerce", utc=False)
            if dt.notna().sum() / max(1, nn) > 0.9:
                info["datetime"].append(c)
                out[c + "__year"] = dt.dt.year.astype(float)
                out[c + "__month"] = dt.dt.month.astype(float)
                out[c + "__day"] = dt.dt.day.astype(float)
                out[c + "__dow"] = dt.dt.dayofweek.astype(float)
                if (dt.dt.hour.fillna(0) != 0).any():
                    out[c + "__hour"] = dt.dt.hour.astype(float)
                out[c + "__ts"] = (dt.astype("int64") // 10**9).where(dt.notna()).astype(float)
                continue
        avg_len = sample.str.len().mean() if len(sample) else 0
        nun = s.nunique()
        if avg_len > 40 and nun > 0.3 * len(df):  # свободный текст
            info["text"].append(c)
            out[c + "__len"] = s.str.len().astype(float)
            out[c + "__words"] = s.str.count(r"\S+").astype(float)
            out[c + "__digits"] = s.str.count(r"\d").astype(float)
            continue
        if nun > 1000 and nun > 0.5 * len(df):
            # строковый идентификатор: как категория он только переобучает — оставляем частоту значения
            info.setdefault("id_like", []).append(c)
            out[c + "__freq"] = s.map(s.value_counts()).astype(float)
            continue
        out[c] = s.fillna("__NA__")
    df = pd.DataFrame(out)
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=const)
    info["dropped_const"] = const
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return df, cat_cols, info


# ------------------------------------------------------------------ трансформации

def apply_config(base: pd.DataFrame, cat_cols: list[str], fc: FeatureConfig, workdir: Path,
                 cache: dict) -> tuple[pd.DataFrame, list[str]]:
    df = base
    num_cols = [c for c in df.columns if c not in cat_cols]
    parts = [df]
    if fc.freq_enc and cat_cols:
        parts.append(pd.DataFrame({f"{c}__freq": df[c].map(df[c].value_counts(normalize=True)).astype(float)
                                   for c in cat_cols}))
    if fc.row_stats and len(num_cols) >= 3:
        nm = df[num_cols]
        parts.append(pd.DataFrame({"row__mean": nm.mean(1), "row__std": nm.std(1), "row__min": nm.min(1),
                                   "row__max": nm.max(1), "row__na": nm.isna().sum(1).astype(float)}))
    if fc.group_stats and cat_cols and num_cols:
        gcols = sorted(cat_cols, key=lambda c: df[c].nunique())[:3]
        ncols = num_cols[:8] if len(num_cols) <= 8 else _top_var(df, num_cols, 8)
        new = {}
        for g in gcols:
            for n in ncols:
                mean = df.groupby(g, observed=True)[n].transform("mean")
                new[f"{n}__minus_mean_by_{g}"] = df[n] - mean
        parts.append(pd.DataFrame(new))
    for i, code in enumerate(fc.llm_code):
        key = ("llm", code)
        if key not in cache:
            cache[key] = run_feature_code(df, code, workdir, tag=f"f{len(cache)}")
        extra = cache[key]
        if extra is not None and len(extra.columns):
            extra = extra[[c for c in extra.columns if c not in df.columns]]
            parts.append(extra)
    out = pd.concat(parts, axis=1) if len(parts) > 1 else df.copy()
    out = out.loc[:, ~out.columns.duplicated()]
    drop = [c for c in fc.drop_cols if c in out.columns]
    if drop:
        out = out.drop(columns=drop)
    cats = [c for c in out.columns if not pd.api.types.is_numeric_dtype(out[c])]
    for c in cats:
        if str(out[c].dtype) != "category":
            out[c] = out[c].astype("string").fillna("__NA__").astype("category")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out, cats


def _top_var(df, cols, k):
    v = df[cols].std() / (df[cols].abs().mean() + 1e-9)
    return v.sort_values(ascending=False).index[:k].tolist()


# ------------------------------------------------------------------ песочница для кода LLM

SANDBOX_TEMPLATE = r'''
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
df = pd.read_pickle(sys.argv[1])
orig_cols = list(df.columns)
n = len(df)
__USER_CODE__
res = add_features(df.copy())
if not isinstance(res, pd.DataFrame):
    raise TypeError("add_features должна вернуть DataFrame")
if len(res) != n:
    raise ValueError(f"add_features изменила число строк: {len(res)} != {n}")
res = res.reset_index(drop=True)
new = [c for c in res.columns if c not in orig_cols]
out = res[new].copy()
for c in out.columns:
    if out[c].dtype == bool:
        out[c] = out[c].astype(float)
    elif not pd.api.types.is_numeric_dtype(out[c]):
        out[c] = out[c].astype("string")
out.columns = [str(c) for c in out.columns]
out.to_pickle(sys.argv[2])
print(json.dumps({"new_columns": list(out.columns)}))
'''


def run_feature_code(df: pd.DataFrame, code: str, workdir: Path, tag: str) -> pd.DataFrame | None:
    """Выполняет add_features(df) от LLM в отдельном процессе. None — если код упал."""
    workdir.mkdir(parents=True, exist_ok=True)
    inp, outp = workdir / f"{tag}_in.pkl", workdir / f"{tag}_out.pkl"
    script = workdir / f"{tag}_code.py"
    try:
        tmp = df.copy()
        for c in tmp.columns:  # в песочницу отдаём категории строками — проще для LLM-кода
            if str(tmp[c].dtype) == "category":
                tmp[c] = tmp[c].astype(object)
        tmp.to_pickle(inp)
        script.write_text(SANDBOX_TEMPLATE.replace("__USER_CODE__", textwrap.dedent(code)), encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", os.defpath), "PYTHONIOENCODING": "utf-8",
               "OMP_NUM_THREADS": str(config.N_JOBS)}
        r = subprocess.run([sys.executable, "-I", str(script), str(inp), str(outp)], capture_output=True,
                           text=True, timeout=min(config.SANDBOX_TIMEOUT, max(30, config.research_remaining())),
                           env=env, cwd=str(workdir))
        if r.returncode != 0 or not outp.exists():
            err = (r.stderr or r.stdout)[-1500:]
            log(f"код признаков {tag} упал:", err.strip().splitlines()[-1] if err.strip() else "?")
            run_feature_code.last_error = err
            return None
        res = pd.read_pickle(outp)
        for c in res.columns:
            if not pd.api.types.is_numeric_dtype(res[c]):
                if res[c].nunique() > 0.5 * len(res):
                    res = res.drop(columns=c)
                else:
                    res[c] = res[c].fillna("__NA__").astype("category")
        # отбрасываем бесполезные колонки
        res = res[[c for c in res.columns if res[c].nunique(dropna=False) > 1]]
        log(f"код признаков {tag}: +{res.shape[1]} колонок {list(res.columns)[:10]}")
        run_feature_code.last_error = None
        return res
    except subprocess.TimeoutExpired:
        log(f"код признаков {tag}: таймаут")
        run_feature_code.last_error = "timeout"
        return None
    except Exception as e:  # noqa: BLE001
        log(f"код признаков {tag}: ошибка {e!r}")
        run_feature_code.last_error = repr(e)
        return None
    finally:
        for p in (inp, outp):
            try:
                p.unlink()
            except OSError:
                pass


run_feature_code.last_error = None
