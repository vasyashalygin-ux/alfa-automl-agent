"""Повторяющиеся объекты («сущности») и история их таргета.

Во многих табличных задачах одна сущность (здание, клиент, устройство) встречается
в нескольких строках — например, в разные годы. Тогда таргет этой сущности в других
строках — очень сильный признак. Модуль:
  1. автоматически ищет комбинацию колонок, которая идентифицирует сущность
     (жадный поиск: строки с одинаковым ключом должны иметь похожий таргет, ключ должен
     повторяться в train и встречаться в test);
  2. считает признаки истории таргета строго внутри фолдов валидации, чтобы не было утечки:
     для валидации/теста — по обучающей части фолда, для обучающих строк — исключая
     собственную группу (год) или по внутреннему out-of-fold.
Применять ли признак, решает цикл AutoResearch по валидации.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import agent_config as config
from agent_log import log


def _key_codes(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    parts = [df[c].astype(str).fillna("NA").values for c in cols]
    if len(parts) == 1:
        s = pd.Series(parts[0])
    else:
        s = pd.Series(parts[0])
        for p in parts[1:]:
            s = s + "\x1f" + p
    return pd.factorize(s.values)[0]


def _dispersion(k: np.ndarray, y: np.ndarray, task: str) -> tuple[float, float]:
    """(доля необъяснённого разброса внутри ключа, средний размер группы) по группам с >=2 строками."""
    d = pd.DataFrame({"k": k, "y": y})
    size = d.groupby("k")["y"].transform("size").values
    rep = size >= 2
    if rep.sum() < 20:
        return 1.0, 1.0
    dd = d[rep]
    if task == "regression":
        tot = np.var(dd["y"].values)
        if tot <= 0:
            return 1.0, 1.0
        within = (dd["y"] - dd.groupby("k")["y"].transform("mean")).pow(2).mean()
        ratio = float(within / tot)
    else:
        mode_share = dd.groupby("k")["y"].agg(lambda s: s.value_counts().iloc[0] / len(s))
        cnt = dd.groupby("k").size()
        purity = float((mode_share * cnt).sum() / cnt.sum())
        base = float(dd["y"].value_counts(normalize=True).iloc[0])
        ratio = (1 - purity) / max(1e-9, 1 - base)
    avg = len(d) / max(1, d["k"].nunique())
    return ratio, avg


def find_entity_key(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray, task: str,
                    exclude: list[str], max_cols: int = 5, max_rows: int = 200_000) -> dict | None:
    cols = [c for c in train.columns if c in test.columns and c not in exclude]
    if not cols or len(train) < 200:
        return None
    rs = np.random.RandomState(config.SEED)
    idx = rs.choice(len(train), min(len(train), max_rows), replace=False)
    tr = train.iloc[idx].reset_index(drop=True)
    yy = y[idx]
    te = test.sample(min(len(test), max_rows), random_state=config.SEED).reset_index(drop=True)
    # кандидаты: не почти-уникальные в одиночку и не константы
    cand = []
    for c in cols:
        nu = tr[c].nunique(dropna=False)
        if 2 <= nu and (tr[c].dtype == object or str(tr[c].dtype) == "category" or nu >= 10 or nu <= 50):
            cand.append(c)
    key: list[str] = []
    best_val = 0.0
    best = None
    both = pd.concat([tr[cand], te[cand]], ignore_index=True)
    # критерий = доля теста, покрытая ключом × доля разброса таргета, объяснённая ключом:
    # ключ должен и узнавать объекты теста, и делать таргет внутри объекта предсказуемым
    for _ in range(max_cols):
        step = None
        for c in cand:
            if c in key:
                continue
            k = _key_codes(both, key + [c])
            ktr, kte = k[:len(tr)], k[len(tr):]
            match = float(np.isin(kte, ktr).mean())
            if match < 0.25:
                continue
            ratio, avg = _dispersion(ktr, yy, task)
            if avg < 1.3:
                continue
            val = match * (1 - ratio)
            if step is None or val > step[4]:
                step = (c, ratio, match, avg, val)
        if step is None or step[4] < best_val * 1.02 + 1e-9:
            break
        key.append(step[0])
        best_val = step[4]
        best = {"cols": list(key), "within_ratio": round(step[1], 4), "test_match": round(step[2], 3),
                "avg_rows_per_entity": round(step[3], 2)}
    if best is None or best["within_ratio"] > 0.6:
        return None
    return best


class EntityEncoder:
    """Признаки истории таргета по сущности, считаются внутри каждого фолда."""

    def __init__(self, key_all: np.ndarray, n_train: int, task: str, k: int, groups: np.ndarray | None,
                 groups_all: np.ndarray | None = None):
        self.key_all = key_all
        self.n_train = n_train
        self.task = task
        self.k = k
        self.groups = groups
        # упорядоченные группы (время) по train+test — для «последнего известного значения»
        self.time_all = None
        if groups_all is not None and pd.api.types.is_numeric_dtype(pd.Series(groups_all)):
            ga = pd.to_numeric(pd.Series(groups_all), errors="coerce").values.astype(float)
            if np.isfinite(ga).mean() > 0.99:
                self.time_all = ga

    def _last(self, src_keys, src_g, src_y, dst_keys, dst_g) -> tuple[np.ndarray, np.ndarray]:
        """Значение сущности в последнем периоде строго раньше dst_g и разрыв по времени."""
        src = pd.DataFrame({"k": src_keys, "g": src_g, "y": src_y}).groupby(["k", "g"], as_index=False)["y"].mean()
        src["g_last"] = src["g"]
        src = src.sort_values("g")
        dst = pd.DataFrame({"k": dst_keys, "g": dst_g, "pos": np.arange(len(dst_keys))}).sort_values("g")
        m = pd.merge_asof(dst, src[["k", "g", "y", "g_last"]], on="g", by="k", direction="backward",
                          allow_exact_matches=False).sort_values("pos")
        return m["y"].values, (m["g"] - m["g_last"]).values

    def _targets(self, y: np.ndarray) -> np.ndarray:
        if self.task == "regression":
            return y.astype(float)[:, None]
        if self.k == 2:
            return (y == 1).astype(float)[:, None]
        k = min(self.k, 20)
        return np.eye(self.k)[y.astype(int)][:, :k]

    def _stats(self, src_keys, src_t, dst_keys):
        df = pd.DataFrame(src_t, columns=[f"t{i}" for i in range(src_t.shape[1])])
        df["k"] = src_keys
        g = df.groupby("k")
        s = g.sum()
        c = g.size()
        dk = pd.Index(dst_keys)
        cnt = c.reindex(dk).fillna(0).values
        sums = s.reindex(dk).fillna(0).values
        return sums, cnt

    def _frame(self, sums, cnt, sq=None) -> pd.DataFrame:
        m = sums / np.maximum(cnt, 1)[:, None]
        m[cnt == 0] = np.nan
        out = {"ent_count": cnt.astype(float)}
        for i in range(m.shape[1]):
            out[f"ent_mean_{i}" if m.shape[1] > 1 else "ent_mean"] = m[:, i]
        if sq is not None:
            var = sq / np.maximum(cnt, 1) - m[:, 0] ** 2
            out["ent_std"] = np.where(cnt >= 2, np.sqrt(np.clip(var, 0, None)), np.nan)
        return pd.DataFrame(out)

    def transform(self, tr_idx, other_idx_list, y_tr) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
        """tr_idx — индексы обучающей части (в train), other_idx_list — индексы (в общей нумерации train+test),
        для которых нужны признаки по всей обучающей части."""
        t = self._targets(y_tr)
        keys_tr = self.key_all[tr_idx]
        reg = self.task == "regression"
        # строки валидации/теста — по всей обучающей части
        others = []
        for oi in other_idx_list:
            s, c = self._stats(keys_tr, t, self.key_all[oi])
            sq = self._stats(keys_tr, t ** 2, self.key_all[oi])[0][:, 0] if reg else None
            f = self._frame(s, c, sq)
            if self.time_all is not None and t.shape[1] == 1:
                f["ent_last"], f["ent_gap"] = self._last(keys_tr, self.time_all[tr_idx], t[:, 0],
                                                         self.key_all[oi], self.time_all[oi])
            others.append(f)
        # обучающие строки — без собственной группы / out-of-fold
        n = len(tr_idx)
        s_all, c_all = self._stats(keys_tr, t, keys_tr)
        sq_all = self._stats(keys_tr, t ** 2, keys_tr)[0][:, 0] if reg else None
        if self.groups is not None:
            g = self.groups[tr_idx]
            gk = pd.factorize(pd.Series(keys_tr.astype(str)) + "|" + pd.Series(g.astype(str)))[0]
            s_own, c_own = self._stats(gk, t, gk)
            sq_own = self._stats(gk, t ** 2, gk)[0][:, 0] if reg else None
        else:
            from sklearn.model_selection import KFold
            s_own = np.zeros_like(s_all)
            c_own = np.zeros(n)
            sq_own = np.zeros(n) if reg else None
            for a, b in KFold(5, shuffle=True, random_state=config.SEED).split(np.arange(n)):
                # «своё» для строк b = то, что приходит из b же
                sb, cb = self._stats(keys_tr[b], t[b], keys_tr[b])
                s_own[b] = sb
                c_own[b] = cb
                if reg:
                    sq_own[b] = self._stats(keys_tr[b], t[b] ** 2, keys_tr[b])[0][:, 0]
        s_tr = s_all - s_own
        c_tr = c_all - c_own
        sq_tr = (sq_all - sq_own) if reg else None
        f = self._frame(s_tr, c_tr, sq_tr)
        if self.time_all is not None and t.shape[1] == 1:
            f["ent_last"], f["ent_gap"] = self._last(keys_tr, self.time_all[tr_idx], t[:, 0],
                                                     keys_tr, self.time_all[tr_idx])
        return f, others


def build_encoder(train, test, y, task, k, exclude, group_col) -> tuple[EntityEncoder | None, dict | None]:
    try:
        info = find_entity_key(train, test, y, task, exclude)
    except Exception as e:  # noqa: BLE001
        log("поиск сущностей упал:", repr(e))
        return None, None
    if info is None:
        log("повторяющиеся объекты не найдены")
        return None, None
    return make_encoder(train, test, task, k, info, group_col), info


def make_encoder(train, test, task, k, info, group_col) -> EntityEncoder:
    cols = info["cols"]
    both = pd.concat([train[cols], test[cols]], ignore_index=True)
    key_all = _key_codes(both, cols)
    groups = train[group_col].values if group_col and group_col in train.columns else None
    groups_all = None
    if groups is not None and group_col in test.columns:
        groups_all = np.concatenate([train[group_col].values, test[group_col].values])
    log(f"сущность: ключ {cols}, совпадение с test {info.get('test_match')}, "
        f"разброс внутри {info.get('within_ratio')}, группировка обучающих строк: {group_col}")
    return EntityEncoder(key_all, len(train), task, k, groups, groups_all)
