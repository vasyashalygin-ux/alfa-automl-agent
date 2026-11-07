"""Цикл AutoResearch: гипотеза → эксперимент → оценка → следующая итерация.

Порядок работы:
  1. разбор условия и константный сабмит (файл существует с первых секунд);
  2. профиль данных, adversarial validation, LLM-аналитик → схема валидации и план;
  3. базовая модель (LightGBM) → первый настоящий сабмит;
  4. цикл: гипотеза (LLM-исследователь или эвристическая очередь) → эксперимент на тех же
     фолдах → принять/отклонить → журнал; сабмит обновляется при каждом улучшении;
  5. финал: жадный ансамбль всех кандидатов + подбор решающего правила → сабмит и отчёт.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import agent_config as config, agent_prompts as prompts
from agent_ensemble import caruana
from agent_features import FeatureConfig, apply_config, base_frame, run_feature_code
from agent_llm import LLM, extract_code
from agent_log import log
from agent_metrics import (get_metric, labels_from_proba, optimize_class_weights, robust_class_weights,
                           METRICS)
from agent_models import TimeUp, default_params, fit_cv, tune_lgbm
from agent_output import build_submission, write_submission
from agent_profiling import (adversarial_validation, column_profile, detect_shift_columns,
                        profile_text, target_profile)
from agent_validation import make_folds, oof_mask

LABEL_OPT_METRICS = {"macro_f1", "weighted_f1", "f1", "balanced_accuracy", "mcc", "kappa", "precision", "recall"}
MODEL_CAPS = {"lgbm": 900, "catboost": 1500, "xgb": 1200, "linear": 300, "extratrees": 600}
# параметры, которые LLM не может менять: ресурсы, устройство, логирование, тип задачи
FORBIDDEN_PARAMS = {"n_jobs", "thread_count", "nthread", "num_threads", "device", "device_type", "task_type",
                    "gpu_id", "tree_method", "verbose", "verbosity", "silent", "objective", "loss_function",
                    "eval_metric", "random_state", "random_seed", "callbacks", "early_stopping_rounds",
                    "enable_categorical", "class_weight", "auto_class_weights", "cat_features", "train_dir",
                    "allow_writing_files", "num_class", "_workdir", "presets"}


class Agent:
    def __init__(self, spec, train: pd.DataFrame, test: pd.DataFrame, sample, writer, llm: LLM | None = None,
                 time_end: float | None = None):
        self.spec, self.test, self.sample, self.writer = spec, test, sample, writer
        self.llm = llm or LLM()
        self.time_end = time_end  # абсолютная граница времени этого агента (для мульти-таргета)
        self.work = Path(spec.workdir) / "_agent_work"
        try:
            self.work.mkdir(exist_ok=True)
            (self.work / ".probe").write_text("ok")
        except OSError:  # каталог задачи только для чтения — работаем во временной папке
            import tempfile
            self.work = Path(tempfile.mkdtemp(prefix="agent_work_"))
        self.exp_log = self.work / f"experiments_{spec.target}.jsonl"
        self.history: list[dict] = []
        self.candidates: list[dict] = []
        self.tried: set[str] = set()
        self.queue: list[dict] = []
        self.insights: list[str] = []
        self.feat_cache: dict = {}
        self.best_overall = None
        self.fc = FeatureConfig()
        self.tt = "none"
        self.wh_params: dict = {}
        self.wh_cw = False
        self.wh_score = None
        self.wh_result = None
        # таргет: выкидываем строки без таргета
        tr = train[train[spec.target].notna()].reset_index(drop=True)
        self.train = tr
        if spec.task_type == "regression":
            self.y = pd.to_numeric(tr[spec.target], errors="coerce").astype(float).values
            self.k = 1
        else:
            idx = {c: i for i, c in enumerate(spec.classes)}
            self.y = tr[spec.target].map(idx).astype(int).values
            self.k = len(spec.classes)
        self.metric = self.bind_metric(spec.metric)
        self.n_train = len(tr)

    def bind_metric(self, name: str):
        """Метрика с учётом положительного класса бинарной задачи (F1 и PR-AUC несимметричны)."""
        m = get_metric(name)
        sp = self.spec
        if sp.task_type == "binary" and name in ("f1", "pr_auc", "precision", "recall", "brier") \
                and sp.positive_class in sp.classes:
            from sklearn import metrics as skm
            from agent_metrics import Metric
            pi = sp.classes.index(sp.positive_class)
            if name == "f1":
                fn = lambda y, p: skm.f1_score(y, p, pos_label=pi, average="binary", zero_division=0)  # noqa: E731
            elif name == "precision":
                fn = lambda y, p: skm.precision_score(y, p, pos_label=pi, average="binary", zero_division=0)  # noqa: E731
            elif name == "recall":
                fn = lambda y, p: skm.recall_score(y, p, pos_label=pi, average="binary", zero_division=0)  # noqa: E731
            elif name == "brier":
                fn = lambda y, p: float(np.mean(((np.asarray(p)[:, pi] if np.ndim(p) == 2 else p)  # noqa: E731
                                                 - (np.asarray(y) == pi)) ** 2))
            else:
                fn = lambda y, p: skm.average_precision_score(  # noqa: E731
                    (np.asarray(y) == pi).astype(int), np.asarray(p)[:, pi] if np.ndim(p) == 2 else p)
            m = Metric(name, m.greater_is_better, m.kind, m.needs, fn)
        return m

    # ================================================================== время
    def research_left(self) -> float:
        left = config.research_remaining()
        if self.time_end is not None:
            left = min(left, (self.time_end - time.time())
                       - (1 - config.RESEARCH_END_FRAC) * (self.time_end - config.START_TIME))
        return left

    def total_left(self) -> float:
        left = config.remaining()
        if self.time_end is not None:
            left = min(left, self.time_end - time.time())
        return left

    # ================================================================== метрика
    def score(self, y_true, pred, opt: bool = False) -> float:
        m = self.metric
        if m.kind == "regression":
            return m.score(y_true, pred)
        if m.needs == "proba":
            return m.score(y_true, pred)
        if opt and m.name in LABEL_OPT_METRICS:
            _, s = optimize_class_weights(y_true, pred, m, n_rounds=1, max_seconds=15)
            return s
        return m.score(y_true, labels_from_proba(pred))

    def plain_scorer(self, y_true, pred):
        return self.score(y_true, pred, opt=False)

    def eval_result(self, r) -> tuple[float, list[float]]:
        o = r.oof
        fin = np.isfinite(o).all(1) if o.ndim == 2 else np.isfinite(o)
        msk = self.mask & fin
        if msk.sum() < 0.5 * self.mask.sum():
            raise TimeUp(f"{r.model}: досчитано меньше половины OOF")
        s = self.score(self.y[msk], o[msk], opt=True)
        fs = [self.plain_scorer(self.y[va], o[va]) for _, va in self.folds if fin[va].all()]
        if msk.sum() < self.mask.sum():  # неполный OOF: не даём ему выигрывать сравнения
            s = s - 1e9 if self.metric.greater_is_better else s + 1e9
        return s, fs

    # ================================================================== подготовка
    def prepare(self):
        sp = self.spec
        self.base, self.cats, info = base_frame(self.train, self.test, sp.target, sp.id_col, sp.other_targets)
        log(f"базовые признаки: {self.base.shape[1]} (категориальных {len(self.cats)}), {info}")
        sp.n_train, sp.n_test = self.n_train, len(self.test)
        self.col_prof = column_profile(self.train, self.test, sp.target, sp.id_col)
        self.tgt_prof = target_profile(self.train[sp.target], sp.task_type)
        self.shifts = detect_shift_columns(self.train, self.test, [c for c in self.train.columns
                                                                   if c not in (sp.target, sp.id_col)])
        Xa = self.base.iloc[:self.n_train]
        Xb = self.base.iloc[self.n_train:]
        self.adv = adversarial_validation(Xa, Xb) if self.research_left() > 300 else {"auc": None, "top_features": []}
        log(f"adversarial AUC={self.adv['auc']}, shift-колонки: {self.shifts}")
        self.profile = profile_text(sp, self.col_prof, self.tgt_prof, self.adv, self.shifts)

    def heuristic_validation(self) -> str:
        import os
        forced = os.environ.get("AGENT_FORCE_VAL")
        if forced:
            return forced
        for s in self.shifts:  # test позже train и периодов достаточно — честная валидация «прошлое → будущее»
            if s["kind"] == "time" and s.get("test_after_train") and s["train_values"] >= 3:
                return f"time:{s['col']}"
        for s in self.shifts:
            if s["kind"] in ("time", "group") and s["train_values"] >= 2:
                return f"group:{s['col']}"
        for s in self.shifts:
            if s["kind"] == "time_continuous":
                return f"time:{s['col']}"
        return "kfold" if self.spec.task_type == "regression" else "stratified"

    # ================================================================== LLM-аналитик
    def analyst(self):
        sp = self.spec
        self.val_scheme = self.heuristic_validation()
        if not self.llm.enabled:
            return
        dd = ""
        if sp.dict_path:
            try:
                dd = Path(sp.dict_path).read_text(encoding="utf-8", errors="replace")[:5000]
            except OSError:
                pass
        heur = {"target": sp.target, "task_type": sp.task_type, "metric": sp.metric, "id_col": sp.id_col,
                "validation": self.val_scheme, "output_mode": sp.output_mode}
        user = prompts.ANALYST_TEMPLATE.format(task_md=sp.md_text[:7000], data_dict=dd, profile=self.profile[:9000],
                                               heuristic=json.dumps(heur, ensure_ascii=False),
                                               actions=prompts.ACTIONS_DOC, metrics=prompts.METRICS_LIST)
        ans = self.llm.chat_json(prompts.SYSTEM_ANALYST, user, max_tokens=9000, timeout=180,
                                 expect=("task_type", "hypotheses", "metric"), list_key="hypotheses")
        if not ans:
            return
        log("аналитик:", json.dumps({k: v for k, v in ans.items() if k != "hypotheses"}, ensure_ascii=False)[:1500])
        # метрика: принимаем, если валидна и согласуется с типом задачи
        m = ans.get("metric")
        if m in METRICS and METRICS[m].kind == ("regression" if sp.task_type == "regression" else "classification"):
            if m != sp.metric and "метрика не найдена" in " ".join(sp.notes):
                log(f"метрика уточнена LLM: {sp.metric} -> {m}")
                sp.metric = m
                self.metric = self.bind_metric(m)
        v = str(ans.get("validation", ""))
        strong_shift = any(s["kind"] in ("time", "group") for s in self.shifts)
        if not strong_shift and v.startswith(("group:", "time:")):
            col = v.split(":", 1)[1]
            ok = col in self.train.columns and col in self.test.columns \
                and 2 <= self.train[col].nunique() <= self.n_train // 2
            if ok:
                # группировка/время честны, только если тест реально лежит «в новых группах»:
                # большинство строк теста имеют значения колонки, которых нет в train
                seen = set(self.train[col].dropna().astype(str))
                new_share = float((~self.test[col].astype(str).isin(seen)).mean())
                ok = new_share >= 0.5
                if not ok:
                    log(f"валидацию от LLM {v} не принимаю: в тесте {1 - new_share:.0%} строк с уже "
                        f"знакомыми значениями `{col}` — разбиение по ней не похоже на train→test")
            if ok:
                self.val_scheme = v
                log(f"валидация от LLM: {v} ({ans.get('validation_reason', '')})")
        self.insights = [str(x) for x in (ans.get("insights") or [])][:10]
        if ans.get("target_transform") in ("log1p", "sqrt") and sp.task_type == "regression":
            self.queue.append({"type": "target_transform", "value": ans["target_transform"], "why": "аналитик"})
        if ans.get("drop_cols"):
            cols = [c for c in ans["drop_cols"] if c in self.base.columns]
            if cols:
                self.queue.append({"type": "drop", "cols": cols, "why": "аналитик: " + str(ans.get("validation_reason", ""))[:100]})
        for h in (ans.get("hypotheses") or [])[:8]:
            if isinstance(h, dict) and h.get("type"):
                self.queue.append(h)

    # ================================================================== базовая модель
    def setup_folds(self):
        self.folds, self.folds_desc = make_folds(self.train, self.y, self.spec.task_type, self.val_scheme)
        self.mask = oof_mask(self.folds, self.n_train)
        log(f"валидация: {self.folds_desc}, покрытие OOF {self.mask.mean():.0%}")

        self.setup_entity()

    def setup_entity(self, cols: list[str] | None = None):
        """Ищет повторяющиеся сущности (или берёт ключ, предложенный LLM)."""
        from agent_entity import build_encoder, make_encoder
        sp = self.spec
        gcol = self.val_scheme.split(":", 1)[1] if self.val_scheme.startswith(("group:", "time:")) else None
        exclude = [c for c in [sp.target, sp.id_col, gcol, *sp.other_targets] if c]
        try:
            if cols:
                info = {"cols": cols, "source": "LLM"}
                self.entity = make_encoder(self.train, self.test, sp.task_type, self.k, info, gcol)
            else:
                yv = self.y if sp.task_type == "regression" else self.y.astype(int)
                self.entity, info = build_encoder(self.train, self.test, yv, sp.task_type, self.k, exclude, gcol)
            self.entity_info = info
        except Exception as e:  # noqa: BLE001
            log("сущности: ошибка", repr(e))
            self.entity, self.entity_info = None, None
        if self.entity_info:
            self.insights.append(f"повторяющиеся объекты: ключ {self.entity_info['cols']} "
                                 f"(совпадение с test {self.entity_info.get('test_match')}, "
                                 f"доля разброса таргета внутри объекта {self.entity_info.get('within_ratio')})")

    def refit_kw(self) -> dict:
        """При валидации «прошлое → будущее» ответ на тест — от модели, обученной на всех данных."""
        if getattr(self, "val_scheme", "").startswith("time:"):
            return {"refit_full": True, "refit_weight": 1.0}
        if config.REFIT_BLEND > 0:  # иначе: смесь моделей фолдов и модели на 100% данных
            return {"refit_full": True, "refit_weight": config.REFIT_BLEND}
        return {}

    def ent_for(self, fc: FeatureConfig):
        return self.entity if (fc.entity and getattr(self, "entity", None) is not None) else None

    def entity_allowed(self) -> bool:
        return config.ENTITY_RANDOM_SPLIT or self.val_scheme.startswith(("group:", "time:"))

    def X(self, fc: FeatureConfig):
        return apply_config(self.base, self.cats, fc, self.work, self.feat_cache)

    def model_cap(self, model: str) -> float:
        """Лимит времени модели: не больше фиксированного и не больше ~8 рабочих LightGBM и 20% остатка."""
        cap = MODEL_CAPS.get(model, 600)
        if model != "lgbm" and self.history:
            wh_t = self.history[0].get("time_s") or 60
            cap = min(cap, max(240, 8 * wh_t), max(240, 0.2 * self.research_left()))
        return cap

    def hit_cap_before(self, model: str) -> bool:
        return any(h.get("model") == model and h.get("cap") and h["time_s"] >= 0.95 * h["cap"] for h in self.history)

    def run_model(self, model, params, fc, tt, cw, cap, desc, typ="model"):
        if self.research_left() < 30:
            raise TimeUp("research time over")
        cap = min(cap, self.model_cap(model))
        X, cats = self.X(fc)
        tl = min(cap, max(30, self.research_left()))
        r = fit_cv(model, params, X, self.y, self.n_train, self.folds, self.spec.task_type, self.k, cats,
                   self.metric, tl, target_transform=tt, class_weight=cw, scorer=self.plain_scorer,
                   entity=self.ent_for(fc), **self.refit_kw())
        s, fs = self.eval_result(r)
        r.fold_scores = fs
        rec = {"id": len(self.history) + 1, "type": typ, "desc": desc, "model": model, "score": round(s, 6),
               "fold_scores": [round(x, 5) for x in fs], "time_s": round(r.fit_time, 1),
               "n_features": X.shape[1], "elapsed_min": round(config.elapsed() / 60, 1), "cap": round(tl, 1)}
        if r.fit_time >= 0.95 * tl:
            rec["note"] = "упёрлась в лимит времени — недообучена"
        return r, s, rec

    def baseline(self):
        p = default_params("lgbm", self.spec.task_type, self.metric.name, self.n_train)
        self.wh_params = p
        try:
            r, s, rec = self.run_model("lgbm", p, self.fc, self.tt, False, MODEL_CAPS["lgbm"], "baseline LightGBM")
        except Exception as e:  # noqa: BLE001
            log("baseline lgbm упал:", repr(e), "— пробую linear")
            p = default_params("linear", self.spec.task_type, self.metric.name, self.n_train)
            r, s, rec = self.run_model("linear", p, self.fc, self.tt, False, 300, "baseline linear")
            self.wh_params = None
        rec["accepted"] = True
        self.wh_score, self.wh_result = s, r
        self.add_candidate("lgbm_base", r, s)
        self.log_exp(rec)
        self.maybe_write(r, s, "baseline")

    # ================================================================== кандидаты/сабмит
    def add_candidate(self, name, r, s):
        self.candidates.append({"name": name, "oof": r.oof, "test": r.test, "score": s,
                                "model": getattr(r, "model", "?")})
        if len(self.candidates) > config.MAX_CANDIDATES:
            sign = 1 if self.metric.greater_is_better else -1
            self.candidates.sort(key=lambda c: -sign * c["score"])
            self.candidates = self.candidates[:config.MAX_CANDIDATES]

    def adjust_priors(self, pred, oof):
        """EM-поправка на сдвиг долей классов (если включена и сдвиг заметный)."""
        if not config.PRIOR_SHIFT or self.metric.kind != "classification" or oof is None:
            return pred
        try:
            from agent_metrics import em_prior_shift
            msk = self.mask
            adj, q_tr, pi = em_prior_shift(pred, oof[msk])
            tv = 0.5 * float(np.abs(pi - q_tr).sum())
            if tv < 0.03:
                return pred
            if not getattr(self, "_prior_logged", False):
                names = [str(c) for c in self.spec.classes]
                log(f"сдвиг долей классов train→test (TV={tv:.3f}): " + ", ".join(
                    f"{n}: {a:.3f}→{b:.3f}" for n, a, b in zip(names, q_tr, pi)))
                self._prior_logged = True
            return adj
        except Exception as e:  # noqa: BLE001
            log("поправка долей классов не удалась:", repr(e))
            return pred

    def maybe_write(self, r_or_pred, s, tag, weights=None):
        if self.best_overall is None or self.metric.better(s, self.best_overall):
            self.best_overall = s
            pred = r_or_pred.test if hasattr(r_or_pred, "test") else r_or_pred
            if hasattr(r_or_pred, "oof"):
                pred = self.adjust_priors(pred, r_or_pred.oof)
            if weights is None and self.metric.kind == "classification" and self.metric.name in LABEL_OPT_METRICS:
                oof = r_or_pred.oof if hasattr(r_or_pred, "oof") else None
                if oof is not None and config.USE_CLASS_THRESHOLDS:
                    weights, _, held = robust_class_weights(self.y[self.mask], oof[self.mask], self.metric,
                                                            max_seconds=40)
                    if weights is None:
                        log(f"  подбор порогов классов не подтвердился на отложенных данных ({held:+.4f}) — без него")
            self.writer.update(self.spec, pred, weights, tag=f"{tag}: {self.metric.name}={s:.5f}")

    def log_exp(self, rec):
        self.history.append(rec)
        best = self.wh_score
        verdict = "ПРИНЯТО" if rec.get("accepted") else (
            "кандидат в ансамбль" if rec["type"] == "model" else "отклонено")
        log(f"эксп #{rec['id']} [{rec['type']}] {rec['desc'][:80]} -> {self.metric.name}={rec['score']} "
            f"{verdict} (рабочая модель={best:.5f}, {rec['time_s']}c)")
        try:
            with open(self.exp_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    def significant(self, s, fs) -> bool:
        if not self.metric.better(s, self.wh_score):
            return False
        gain = self.metric.gain(s, self.wh_score)
        if gain < 1e-4 * max(1e-9, abs(self.wh_score)):
            return False
        base_fs = self.wh_result.fold_scores
        if len(base_fs) == len(fs) and len(fs) >= 3:
            wins = sum(self.metric.better(a, b) for a, b in zip(fs, base_fs))
            if wins < (len(fs) + 1) // 2:
                return False
            if config.STRICT_ACCEPT:
                # парный тест по фолдам: прирост должен быть устойчивым, а не шумом одного фолда
                d = np.array([self.metric.gain(a, b) for a, b in zip(fs, base_fs)], float)
                sd = d.std(ddof=1)
                t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else (np.inf if d.mean() > 0 else 0.0)
                rel = gain / max(1e-9, abs(self.wh_score))
                if not (t >= 1.0 or rel >= 0.01):
                    log(f"  прирост неустойчив по фолдам (t={t:.2f}, +{rel:.2%}) — не принимаю")
                    return False
        return True

    # ================================================================== эвристическая очередь
    def default_queue(self) -> list[dict]:
        q: list[dict] = []
        reg = self.spec.task_type == "regression"
        if reg:
            yv = self.y
            if np.nanmin(yv) >= 0 and abs(self.tgt_prof.get("skew") or 0) > 1:
                q.append({"type": "target_transform", "value": "log1p", "why": "скошенный неотрицательный таргет"})
        else:
            q.append({"type": "model", "model": "lgbm", "class_weight": True, "params": {},
                      "why": "дисбаланс классов"})
        if getattr(self, "entity", None) is not None:
            q.insert(0, {"type": "features", "op": "entity",
                         "why": f"объекты повторяются (ключ {self.entity_info['cols']}) — история их таргета"})
        q += [{"type": "features", "op": "freq_enc", "why": "частоты категорий"},
              {"type": "model", "model": "catboost", "params": {}, "why": "сильная модель для категорий"},
              {"type": "features", "op": "group_stats", "why": "отклонение от среднего по группе"},
              {"type": "model", "model": "xgb", "params": {}, "why": "разнообразие для ансамбля"},
              {"type": "tune", "model": "lgbm", "why": "подбор гиперпараметров"},
              {"type": "features", "op": "row_stats", "why": "агрегаты по строке"},
              {"type": "model", "model": "autogluon", "params": {}, "why": "AutoML-кандидат для ансамбля"},
              {"type": "model", "model": "lgbm", "params": {"num_leaves": 255, "min_child_samples": 10,
                                                            "learning_rate": 0.03, "colsample_bytree": 0.5},
               "why": "более глубокие деревья"},
              {"type": "model", "model": "extratrees", "params": {}, "why": "разнообразие для ансамбля"},
              {"type": "model", "model": "linear", "params": {}, "why": "разнообразие для ансамбля"}]
        if not reg:
            q.insert(3, {"type": "model", "model": "catboost", "class_weight": True, "params": {},
                         "why": "catboost с весами классов"})
        return q

    def late_actions(self) -> list[dict]:
        """Когда идеи кончились, а время есть: укрепляем ансамбль вариациями лучших моделей."""
        self.late_round = getattr(self, "late_round", 0) + 1
        if self.late_round > 40:
            return []
        acts = []
        r = self.late_round
        if r % 4 == 0 and self.wh_params is not None:  # время от времени — новый раунд Optuna
            acts.append({"type": "tune", "model": "lgbm", "round": r, "why": f"поздняя фаза: подбор параметров {r}"})
        if self.late_round == 1 and not any(h.get("model") == "autogluon" for h in self.history):
            acts.append({"type": "model", "model": "autogluon", "params": {}, "why": "поздняя фаза"})
        # типы моделей с лучшими оценками
        sign = 1 if self.metric.greater_is_better else -1
        best_by_type = {}
        for h in self.history:
            m = h.get("model")
            if m in ("lgbm", "catboost", "xgb") and np.isfinite(h.get("score", np.nan)):
                if m not in best_by_type or sign * h["score"] > sign * best_by_type[m]:
                    best_by_type[m] = h["score"]
        order = sorted(best_by_type, key=lambda m: -sign * best_by_type[m])[:2]
        cw = self.wh_cw
        for m in order:
            p = {"seed": config.SEED + 100 * self.late_round}
            if m == "lgbm":
                p.update(learning_rate=0.02 + 0.01 * (r % 2), colsample_bytree=round(0.45 + 0.1 * (r % 5), 2))
            elif m == "catboost":
                p.update(learning_rate=0.05, depth=6 + r % 3)
            else:
                p.update(learning_rate=0.03, max_depth=6 + r % 3)
            acts.append({"type": "model", "model": m, "params": p, "class_weight": cw,
                         "why": f"поздняя фаза: вариация {self.late_round} для ансамбля"})
        return acts

    # ================================================================== исполнение действий
    def action_key(self, a: dict) -> str:
        a2 = {k: v for k, v in a.items() if k not in ("why", "description", "thought")}
        if a2.get("type") in ("features", "target_transform", "drop", "feature_code"):
            a2["_fc"] = self.fc.key() + self.tt
        return json.dumps(a2, sort_keys=True, ensure_ascii=False, default=str)

    def execute(self, a: dict):
        t = a.get("type")
        why = str(a.get("why", ""))[:120]
        if t == "stop":
            return
        if t == "model":
            model = a.get("model", "lgbm")
            if model not in MODEL_CAPS and model != "autogluon":
                return
            if model == "autogluon":
                return self.run_autogluon(why)
            params = default_params(model, self.spec.task_type, self.metric.name, self.n_train)
            if model == "lgbm" and self.wh_params:
                params = dict(self.wh_params)
            user_p = {k: v for k, v in (a.get("params") or {}).items()
                      if isinstance(v, (int, float, str, bool)) and k not in FORBIDDEN_PARAMS}
            params.update(user_p)
            if self.hit_cap_before(model) and model in ("catboost", "xgb", "lgbm") and "learning_rate" not in user_p:
                # прошлый запуск не успел — ускоряем, а не повторяем впустую
                params["learning_rate"] = min(0.3, float(params.get("learning_rate", 0.1)) * 2)
                if model == "catboost":
                    params["depth"] = min(int(params.get("depth", 6)), 6)
                log(f"  {model} ранее упёрся в лимит — ускоряю: lr={params['learning_rate']}")
            cw = bool(a.get("class_weight")) and self.spec.task_type != "regression"
            desc = f"{model} {user_p or ''}{' class_weight' if cw else ''} — {why}"
            r, s, rec = self.run_model(model, params, self.fc, self.tt, cw, MODEL_CAPS[model], desc)
            self.add_candidate(f"{model}_{len(self.history) + 1}", r, s)
            rec["accepted"] = False
            if model == "lgbm" and self.significant(s, r.fold_scores):
                self.wh_params, self.wh_cw, self.wh_score, self.wh_result = params, cw, s, r
                rec["accepted"] = True
                rec["note"] = "новые параметры рабочей модели"
            self.log_exp(rec)
            self.maybe_write(r, s, rec["desc"][:40])
            return
        if t == "tune":
            if self.wh_params is None:
                return
            tl = min(900, 0.15 * self.research_left())
            if tl < 120:
                return
            X, cats = self.X(self.fc)
            p = tune_lgbm(X, self.y, self.n_train, self.folds, self.spec.task_type, self.k, cats, self.metric,
                          tl, self.plain_scorer, entity=self.ent_for(self.fc), target_transform=self.tt)
            return self.execute({"type": "model", "model": "lgbm", "params": {k: v for k, v in p.items()
                                 if k in ("num_leaves", "min_child_samples", "colsample_bytree", "subsample",
                                          "reg_lambda", "reg_alpha", "learning_rate")},
                                 "class_weight": self.wh_cw, "why": "параметры от Optuna"})
        # --- гипотезы о данных: оцениваются рабочей моделью
        fc, tt = self.fc, self.tt
        if t == "features":
            op = a.get("op")
            if op not in ("freq_enc", "row_stats", "group_stats", "entity") or getattr(fc, op):
                return
            if op == "entity" and (getattr(self, "entity", None) is None or not self.entity_allowed()):
                return
            fc = fc.copy(**{op: True})
            desc = f"+{op} — {why}"
        elif t == "drop":
            cols = [c for c in (a.get("cols") or []) if c in self.base.columns or c in self.X(fc)[0].columns]
            cols = [c for c in cols if c not in fc.drop_cols]
            if not cols:
                return
            fc = fc.copy(drop_cols=fc.drop_cols + cols)
            desc = f"drop {cols[:6]} — {why}"
        elif t == "target_transform":
            v = a.get("value")
            if self.spec.task_type != "regression" or v not in ("none", "log1p", "sqrt") or v == tt:
                return
            if v in ("log1p", "sqrt") and np.nanmin(self.y) < 0:
                return
            tt = v
            desc = f"target {v} — {why}"
        elif t == "entity_key":
            if not self.entity_allowed():
                return
            cols = [c for c in (a.get("cols") or []) if c in self.train.columns and c in self.test.columns
                    and c not in (self.spec.target, self.spec.id_col)]
            if not cols:
                return
            old = (getattr(self, "entity", None), getattr(self, "entity_info", None))
            self.setup_entity(cols)
            if self.entity is None:
                self.entity, self.entity_info = old
                return
            fc = fc.copy(entity=True)
            desc = f"ключ сущности {cols} — {why}"
            model = "lgbm" if self.wh_params is not None else "linear"
            params = self.wh_params if self.wh_params is not None else default_params("linear", self.spec.task_type, "", 0)
            r, s, rec = self.run_model(model, params, fc, tt, self.wh_cw, MODEL_CAPS[model], desc, typ=t)
            self.add_candidate(f"{t}_{len(self.history) + 1}", r, s)
            ok = self.significant(s, r.fold_scores)
            rec["accepted"] = ok
            if ok:
                self.fc, self.wh_score, self.wh_result = fc, s, r
            else:
                self.entity, self.entity_info = old
            self.log_exp(rec)
            self.maybe_write(r, s, desc[:40])
            return
        elif t == "feature_code":
            code = a.get("code") or ""
            if "def add_features" not in code:
                return
            code = self.ensure_code_works(code)
            if code is None:
                self.log_exp({"id": len(self.history) + 1, "type": t, "desc": f"код признаков не заработал — {why}",
                              "model": "-", "score": float("nan"), "fold_scores": [], "time_s": 0, "accepted": False})
                return
            fc = fc.copy(llm_code=fc.llm_code + [code])
            desc = f"feature_code: {str(a.get('description', ''))[:80]} — {why}"
        else:
            return
        model = "lgbm" if self.wh_params is not None else "linear"
        params = self.wh_params if self.wh_params is not None else default_params("linear", self.spec.task_type, "", 0)
        r, s, rec = self.run_model(model, params, fc, tt, self.wh_cw, MODEL_CAPS[model], desc, typ=t)
        self.add_candidate(f"{t}_{len(self.history) + 1}", r, s)
        ok = self.significant(s, r.fold_scores)
        rec["accepted"] = ok
        if ok:
            self.fc, self.tt = fc, tt
            self.wh_score, self.wh_result = s, r
        self.log_exp(rec)
        self.maybe_write(r, s, desc[:40])

    def ensure_code_works(self, code: str) -> str | None:
        for attempt in range(2):
            res = run_feature_code(self.base, code, self.work, tag=f"chk{len(self.feat_cache)}_{attempt}")
            if res is not None:
                self.feat_cache[("llm", code)] = res
                return code if len(res.columns) else None
            err = run_feature_code.last_error or ""
            if not self.llm.enabled or attempt == 1:
                return None
            cols = {c: str(self.base[c].dtype) for c in self.base.columns}
            txt = self.llm.chat(prompts.SYSTEM_CODER, prompts.FIX_CODE_TEMPLATE.format(
                columns=json.dumps(cols, ensure_ascii=False)[:4000], code=code, error=err[-1500:]))
            code = extract_code(txt or "")
            if not code:
                return None
        return None

    def run_autogluon(self, why):
        try:
            import autogluon.tabular  # noqa: F401
        except Exception:  # noqa: BLE001
            log("autogluon недоступен — пропуск")
            return
        tl = min(3000, 0.35 * self.research_left())
        if tl < 300:
            return
        X, cats = self.X(self.fc)
        params = {"presets": "medium_quality", "_workdir": str(self.work)}
        r = fit_cv("autogluon", params, X, self.y, self.n_train, self.folds, self.spec.task_type, self.k, cats,
                   self.metric, tl, target_transform=self.tt, scorer=self.plain_scorer, entity=self.ent_for(self.fc))
        s, fs = self.eval_result(r)
        r.fold_scores = fs
        rec = {"id": len(self.history) + 1, "type": "model", "desc": f"AutoGluon medium_quality — {why}",
               "model": "autogluon", "score": round(s, 6), "fold_scores": [round(x, 5) for x in fs],
               "time_s": round(r.fit_time, 1), "accepted": False, "n_features": X.shape[1]}
        self.add_candidate("autogluon", r, s)
        self.log_exp(rec)
        self.maybe_write(r, s, "autogluon")

    # ================================================================== LLM-исследователь
    def error_analysis(self) -> str:
        r = self.wh_result
        msk = self.mask
        y = self.y[msk]
        try:
            if self.spec.task_type == "regression":
                p = r.oof[msk]
                res = p - y
                lines = [f"bias(mean pred-true)={res.mean():.4f}, rmse={np.sqrt(np.mean(res ** 2)):.4f}"]
                qs = pd.qcut(pd.Series(y), 5, duplicates="drop")
                g = pd.DataFrame({"q": qs, "e2": res ** 2, "r": res}).groupby("q", observed=True)
                lines.append("ошибка по квантилям таргета: " + "; ".join(
                    f"{k}: rmse={np.sqrt(v.e2.mean()):.2f}, bias={v.r.mean():.2f}" for k, v in g))
                tr = self.train.loc[msk].reset_index(drop=True)
                for c in [c for c in self.cats if c in tr.columns][:3]:
                    gg = pd.DataFrame({"c": tr[c].astype(str), "e2": res ** 2}).groupby("c")["e2"].agg(["sum", "mean", "size"])
                    gg = gg.sort_values("sum", ascending=False).head(6)
                    lines.append(f"вклад в ошибку по `{c}`: " + "; ".join(
                        f"{k}: share={v['sum'] / (res ** 2).sum():.0%}, rmse={np.sqrt(v['mean']):.1f}, n={int(v['size'])}"
                        for k, v in gg.iterrows()))
                return "\n".join(lines)
            from sklearn.metrics import classification_report, confusion_matrix
            pred = labels_from_proba(r.oof[msk])
            names = [str(c) for c in self.spec.classes]
            rep = classification_report(y, pred, labels=range(self.k), target_names=names, zero_division=0,
                                        output_dict=True)
            lines = ["per-class: " + "; ".join(f"{n}: p={rep[n]['precision']:.2f} r={rep[n]['recall']:.2f} "
                                                 f"f1={rep[n]['f1-score']:.2f} n={int(rep[n]['support'])}"
                                                 for n in names)]
            cm = confusion_matrix(y, pred, labels=range(self.k))
            np.fill_diagonal(cm, 0)
            top = np.dstack(np.unravel_index(np.argsort(-cm.ravel())[:6], cm.shape))[0]
            lines.append("частые путаницы (истина->предсказание): " + "; ".join(
                f"{names[i]}->{names[j]}: {cm[i, j]}" for i, j in top if cm[i, j] > 0))
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"(анализ ошибок недоступен: {e!r})"

    def researcher_prompt(self) -> str:
        """Собирает промпт исследователя (в основном потоке — читает состояние агента)."""
        imp = self.wh_result.importance if self.wh_result is not None else {}
        tot = sum(imp.values()) or 1
        top = sorted(imp.items(), key=lambda kv: -kv[1])[:20]
        hist = "\n".join(f"#{h['id']} [{h['type']}] {h['desc'][:150]} -> {h['score']} "
                         f"{'ПРИНЯТО' if h.get('accepted') else 'нет'} ({h['time_s']}c)" for h in self.history[-30:])
        cost = np.median([h["time_s"] for h in self.history if h.get("time_s")]) if self.history else 60
        cols = {c: str(self.base[c].dtype) for c in self.base.columns}
        best = {"workhorse": "lgbm", "params": {k: v for k, v in (self.wh_params or {}).items()
                                                if k not in ("n_estimators",)},
                "class_weight": self.wh_cw, "target_transform": self.tt,
                "features": {k: v for k, v in asdict(self.fc).items() if k != "llm_code"},
                "n_feature_code_blocks": len(self.fc.llm_code), "score": self.wh_score,
                "best_any_candidate": self.best_overall}
        summary = (f"{self.spec.task_type}, таргет `{self.spec.target}`, метрика {self.metric.name}, "
                   f"train={self.n_train}, test={len(self.test)}\n" + self.spec.md_text[:2500])
        return prompts.RESEARCHER_TEMPLATE.format(
            summary=summary, insights="\n".join(self.insights) or "-", best=json.dumps(best, ensure_ascii=False, default=str),
            metric=self.metric.name, direction="больше лучше" if self.metric.greater_is_better else "меньше лучше",
            folds=self.folds_desc, history=hist, importance=", ".join(f"{k}={v / tot:.3f}" for k, v in top),
            errors=self.error_analysis(), columns=json.dumps(cols, ensure_ascii=False)[:3500],
            minutes=int(self.research_left() / 60), exp_cost=int(cost), actions=prompts.ACTIONS_DOC)

    @staticmethod
    def parse_researcher(ans) -> list[dict]:
        if not ans:
            return []
        log("исследователь:", str(ans.get("thought", ""))[:600])
        acts = [a for a in (ans.get("actions") or []) if isinstance(a, dict) and a.get("type")]
        return acts[:3]

    # ================================================================== главный цикл
    def loop(self):
        """Эксперименты идут без остановки; запрос к LLM-исследователю выполняется в фоне.

        Пока Qwen думает (или не отвечает), агент берёт гипотезы из эвристической очереди.
        Ответ LLM, когда приходит, добавляется в начало очереди.
        """
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=1)
        fallback = self.default_queue()
        if fallback and fallback[0].get("op") == "entity":  # самая сильная гипотеза — проверяем первой
            self.queue.insert(0, fallback.pop(0))
        llm_rounds = 0
        empty_streak = 0
        cooldown_until = 0.0
        pending = None
        try:
            while self.research_left() > 60:
                # 1) забрать ответ исследователя, если готов
                if pending is not None and pending.done():
                    try:
                        acts = self.parse_researcher(pending.result())
                    except Exception as e:  # noqa: BLE001
                        log("исследователь: ошибка", repr(e))
                        acts = []
                    pending = None
                    acts = [x for x in acts if x.get("type") != "stop"]
                    new = [x for x in acts if self.action_key(x) not in self.tried]
                    if new:
                        empty_streak = 0
                        self.queue[0:0] = new  # идеи LLM — в начало очереди
                    else:  # LLM повторяется или не ответила — пауза, экономим токены и время
                        empty_streak += 1
                        if empty_streak >= 2:
                            cooldown_until, empty_streak = time.time() + 600, 0
                # 2) отправить новый запрос, если очередь почти пуста
                if (pending is None and self.llm.enabled and llm_rounds < 25 and time.time() >= cooldown_until
                        and len(self.queue) <= 1 and self.research_left() > 300):
                    user = self.researcher_prompt()
                    pending = pool.submit(self.llm.chat_json, prompts.SYSTEM_RESEARCHER, user, max_tokens=8000,
                                          timeout=180, expect=("actions",), list_key="actions")
                    llm_rounds += 1
                # 3) если очередь пуста — эвристики
                if not self.queue:
                    if fallback:
                        self.queue.append(fallback.pop(0))
                    else:
                        late = self.late_actions()
                        if late:
                            self.queue.extend(late)
                        elif pending is not None:
                            try:  # ждём ответ LLM, но не больше 2 минут
                                pending.result(timeout=min(120, max(1, self.research_left() - 60)))
                            except Exception:  # noqa: BLE001
                                pass
                            continue
                        else:
                            log("гипотезы закончились")
                            break
                a = self.queue.pop(0)
                key = self.action_key(a)
                if key in self.tried:
                    continue
                self.tried.add(key)
                try:
                    self.execute(a)
                except TimeUp as e:
                    log("время исследования вышло:", e)
                    break
                except MemoryError:
                    log("MemoryError на гипотезе", a.get("type"))
                except Exception as e:  # noqa: BLE001
                    log(f"гипотеза {a.get('type')} упала: {e!r}")
                    log(traceback.format_exc()[-1200:])
                    self.history.append({"id": len(self.history) + 1, "type": a.get("type"),
                                         "desc": f"ошибка: {e!r}"[:150], "model": a.get("model", "-"),
                                         "score": float("nan"), "fold_scores": [], "time_s": 0, "accepted": False})
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    # ================================================================== финал
    def finalize(self):
        if not self.candidates:
            return
        if os.environ.get("AGENT_DUMP_CANDIDATES") == "1":  # для офлайн-анализа стратегий ансамбля
            import pickle
            with open(self.work / f"candidates_{self.spec.target}.pkl", "wb") as f:
                pickle.dump({"cands": self.candidates, "mask": self.mask, "y": self.y,
                             "metric": self.metric.name, "classes": self.spec.classes}, f)
        msk = self.mask
        y = self.y[msk]
        sign = 1 if self.metric.greater_is_better else -1
        ranked = [c for c in sorted(self.candidates, key=lambda c: -sign * c["score"])
                  if np.isfinite(c["oof"][msk]).all() and np.isfinite(c["test"]).all()]
        good = ranked[:15]
        if len(good) >= 2 and self.total_left() > 15:
            label_opt = (self.metric.kind == "classification" and self.metric.name in LABEL_OPT_METRICS
                         and self.total_left() > 120)
            if label_opt:
                # смеси сравниваем по той же метрике, что и финал: с быстрым подбором весов классов
                good = good[:8]
                t_end = time.time() + max(30, 0.6 * (self.total_left() - 60))

                def sfn(o):
                    if time.time() > t_end:
                        return self.plain_scorer(y, o) - (1e9 if self.metric.greater_is_better else -1e9)
                    return optimize_class_weights(y, o, self.metric, n_rounds=1, max_seconds=5, grid_n=9)[1]
            else:
                def sfn(o):
                    return self.plain_scorer(y, o)
            mode = config.ENSEMBLE_MODE
            if mode == "auto":  # регрессия — среднее лучших (устойчивее), классификация — жадный выбор
                mode = "topk" if self.spec.task_type == "regression" else "greedy"
            if mode in ("greedy", "mix"):
                w, oof_g, test_g, _ = caruana([c["oof"][msk] for c in good], [c["test"] for c in good],
                                              sfn, self.metric.greater_is_better, n_iter=25)
                desc = ", ".join(f"{good[i]['name']}×{wi:.2f}" for i, wi in enumerate(w) if wi > 0)
            if mode in ("topk", "mix"):
                # простое среднее k лучших: устойчивее жадного выбора, который на валидации
                # «вылавливает» случайно удачные копии моделей
                best_s = ranked[0]["score"]
                # в среднее — только почти равные модели: для классификации допуск уже (проверено на сдвиге)
                tol_rel = config.ENSEMBLE_TOL_REG if self.spec.task_type == "regression" else config.ENSEMBLE_TOL_CLS
                near = [c for c in ranked if self.metric.gain(c["score"], best_s) >= -tol_rel * abs(best_s)]
                if len(near) < min(3, len(ranked)):
                    near = ranked[:min(3, len(ranked))]
                k = min(config.ENSEMBLE_TOPK, len(near))
                oof_a = np.mean([c["oof"][msk] for c in near[:k]], axis=0)
                test_a = np.mean([c["test"] for c in near[:k]], axis=0)
            if mode == "topk":
                oof_b, test_b, desc = oof_a, test_a, f"среднее топ-{k}: " + ", ".join(c["name"] for c in near[:k])
            elif mode == "mix":
                oof_b, test_b = 0.5 * oof_g + 0.5 * oof_a, 0.5 * test_g + 0.5 * test_a
                desc = f"0.5×[{desc}] + 0.5×[среднее топ-{k}]"
            else:
                oof_b, test_b = oof_g, test_g
            s_plain = self.plain_scorer(y, oof_b)
            weights = None
            s = s_plain
            if (self.metric.kind == "classification" and self.metric.name in LABEL_OPT_METRICS
                    and config.USE_CLASS_THRESHOLDS):
                weights, s, held = robust_class_weights(y, oof_b, self.metric,
                                                        max_seconds=min(160, max(8, self.total_left() - 20)))
                log(f"  пороги классов: прирост на отложенных половинах {held:+.4f}"
                    f"{'' if weights is not None else ' — не применяем'}")
            log(f"ансамбль ({mode}): {desc[:300]} -> {self.metric.name}={s:.5f} "
                f"(лучший одиночный {good[0]['score']:.5f})")
            self.history.append({"id": len(self.history) + 1, "type": "ensemble", "model": mode,
                                 "desc": "ансамбль: " + desc[:300],
                                 "score": round(s, 6), "fold_scores": [], "time_s": 0, "accepted": True})
            # усреднённый ансамбль пишем, даже если на валидации он чуть хуже «удачной» одиночной модели:
            # такая разница — обычно шум отбора, а среднее устойчивее на тесте
            tol = 0.0 if mode == "greedy" else 0.01 * abs(self.best_overall or 0)
            if self.best_overall is None or self.metric.gain(s, self.best_overall) >= -tol:
                self.best_overall = s
                full_oof = np.zeros((self.n_train,) + oof_b.shape[1:]) if np.ndim(oof_b) == 2 else None
                if full_oof is not None:
                    full_oof[msk] = oof_b
                test_b = self.adjust_priors(test_b, full_oof)
                self.writer.update(self.spec, test_b, weights, tag=f"ensemble: {self.metric.name}={s:.5f}")
        self.write_report()

    def write_report(self):
        try:
            lines = [f"# Отчёт агента: {Path(self.spec.md_path).name} / `{self.spec.target}`", "",
                     f"- тип задачи: {self.spec.task_type}, метрика: {self.metric.name}",
                     f"- валидация: {self.folds_desc}", f"- adversarial AUC: {self.adv.get('auc')}",
                     f"- итоговая оценка на валидации: {self.best_overall}",
                     f"- LLM: вызовов {self.llm.calls}, токенов {self.llm.tokens_in}+{self.llm.tokens_out}",
                     f"- время: {config.elapsed() / 60:.1f} мин", "", "## Наблюдения аналитика"]
            lines += [f"- {x}" for x in self.insights] or ["- (нет)"]
            lines += ["", "## Журнал экспериментов", "", "| # | тип | гипотеза | метрика | принято | время, c |",
                      "|---|---|---|---|---|---|"]
            for h in self.history:
                d = str(h["desc"]).replace("|", "/")[:120]
                lines.append(f"| {h['id']} | {h['type']} | {d} | {h['score']} | {'да' if h.get('accepted') else 'нет'} | {h['time_s']} |")
            (self.work / f"report_{self.spec.target}.md").write_text("\n".join(lines), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            log("отчёт не записан:", repr(e))

    # ================================================================== запуск
    def run(self):
        self.prepare()
        self.analyst()
        self.setup_folds()
        self.baseline()
        try:
            self.loop()
        finally:
            self.finalize()
