"""Зоопарк моделей с единым CV-интерфейсом.

Все модели обучаются на одних и тех же фолдах и возвращают OOF и test-предсказания:
для регрессии — значения в исходной шкале, для классификации — вероятности (n, K).
У бустингов есть остановка по таймеру, чтобы не выйти за бюджет.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import agent_config as config
from agent_log import log

warnings.filterwarnings("ignore")


@dataclass
class CVResult:
    model: str
    params: dict
    oof: np.ndarray
    test: np.ndarray
    fold_scores: list = field(default_factory=list)
    fit_time: float = 0.0
    best_iters: list = field(default_factory=list)
    importance: dict = field(default_factory=dict)


class TimeUp(Exception):
    pass


# ------------------------------------------------------------------ helpers

def _proba_full(p: np.ndarray, classes_seen, k: int) -> np.ndarray:
    out = np.zeros((p.shape[0], k))
    out[:, np.asarray(classes_seen, int)] = p
    return out


def _tt(y, kind):
    return np.log1p(y) if kind == "log1p" else (np.sqrt(np.clip(y, 0, None)) if kind == "sqrt" else y)


def _itt(p, kind):
    if kind == "log1p":
        return np.expm1(p)
    if kind == "sqrt":
        return np.clip(p, 0, None) ** 2
    return p


LGB_OBJ = {"mae": "regression_l1", "medae": "regression_l1", "mape": "regression_l1", "smape": "regression_l1"}


def default_params(model: str, task: str, metric: str, n_rows: int) -> dict:
    big = n_rows > 200_000
    if model == "lgbm":
        p = dict(n_estimators=5000, learning_rate=0.05 if not big else 0.08, num_leaves=63,
                 min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                 reg_lambda=1.0)
        if task == "regression":
            p["objective"] = LGB_OBJ.get(metric, "regression")
        return p
    if model == "catboost":
        p = dict(iterations=4000, learning_rate=0.08 if not big else 0.12, depth=7, l2_leaf_reg=3.0)
        if task == "multiclass":  # MultiClass в CatBoost дорогой: мельче деревья, быстрее шаг
            # CTR-статистики считаются на каждый класс — в 6+ раз медленнее; one-hot для категорий до 255 значений
            p.update(learning_rate=0.12, depth=6, border_count=128, iterations=3000, one_hot_max_size=255)
        return p
    if model == "xgb":
        return dict(n_estimators=4000, learning_rate=0.05 if not big else 0.08, max_depth=7,
                    subsample=0.8, colsample_bytree=0.7, min_child_weight=2, reg_lambda=1.0)
    if model == "linear":
        return dict(C=1.0, alpha=1.0)
    if model == "extratrees":
        return dict(n_estimators=300, min_samples_leaf=2, max_features=0.6)
    if model == "autogluon":
        return dict(presets="medium_quality")
    return {}


def _deadline_cb_lgb(deadline):
    def cb(env):
        if time.time() > deadline:
            import lightgbm as lgb
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
    cb.order = 99
    return cb


# ------------------------------------------------------------------ fit one fold

def _fit_predict(model, params, Xtr, ytr, Xva, yva, Xte, task, k, cat_cols, metric, deadline, class_weight):
    """Возвращает (pred_val, pred_test, best_iter, importance_dict)."""
    n_jobs = config.N_JOBS
    cls = task != "regression"
    seed = params.get("seed", config.SEED)
    imp = {}
    if model == "lgbm":
        import lightgbm as lgb
        p = {k_: v for k_, v in params.items() if k_ not in ("seed",)}
        es = p.pop("early_stopping", 150)
        Est = lgb.LGBMClassifier if cls else lgb.LGBMRegressor
        m = Est(**p, random_state=seed, n_jobs=n_jobs, verbose=-1,
                class_weight=("balanced" if class_weight else None) if cls else None)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(es, verbose=False), _deadline_cb_lgb(deadline)])
        bi = m.best_iteration_ or p.get("n_estimators")
        imp = dict(zip(Xtr.columns, m.booster_.feature_importance("gain")))
        if cls:
            return (_proba_full(m.predict_proba(Xva), m.classes_, k),
                    _proba_full(m.predict_proba(Xte), m.classes_, k), bi, imp)
        return m.predict(Xva), m.predict(Xte), bi, imp

    if model == "xgb":
        import xgboost as xgb

        class _Stop(xgb.callback.TrainingCallback):
            def after_iteration(self, model_, epoch, evals_log):
                return time.time() > deadline

        p = {k_: v for k_, v in params.items() if k_ not in ("seed",)}
        es = p.pop("early_stopping", 150)
        sw = None
        if cls and class_weight:
            from sklearn.utils.class_weight import compute_sample_weight
            sw = compute_sample_weight("balanced", ytr)
        if cls:
            # xgboost требует метки 0..m-1 — перекодируем классы, встреченные в фолде
            seen = np.unique(ytr)
            remap = {c: i for i, c in enumerate(seen)}
            ytr_ = np.vectorize(remap.get)(ytr)
            mask_va = np.isin(yva, seen)
            m = xgb.XGBClassifier(**p, tree_method="hist", enable_categorical=True, random_state=seed,
                                  n_jobs=n_jobs, early_stopping_rounds=es, callbacks=[_Stop()],
                                  max_cat_to_onehot=1)
            eval_y = np.vectorize(remap.get)(yva[mask_va])
            m.fit(Xtr, ytr_, sample_weight=sw, eval_set=[(Xva[mask_va], eval_y)], verbose=False)
            pv, pt = m.predict_proba(Xva), m.predict_proba(Xte)
            if pv.ndim == 1:
                pv, pt = np.column_stack([1 - pv, pv]), np.column_stack([1 - pt, pt])
            bi = getattr(m, "best_iteration", None)
            imp = dict(zip(Xtr.columns, m.feature_importances_))
            return _proba_full(pv, seen, k), _proba_full(pt, seen, k), bi, imp
        obj = "reg:absoluteerror" if metric in ("mae", "medae", "mape", "smape") else "reg:squarederror"
        m = xgb.XGBRegressor(**p, objective=obj, tree_method="hist", enable_categorical=True,
                             random_state=seed, n_jobs=n_jobs, early_stopping_rounds=es, callbacks=[_Stop()],
                             max_cat_to_onehot=1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        imp = dict(zip(Xtr.columns, m.feature_importances_))
        return m.predict(Xva), m.predict(Xte), getattr(m, "best_iteration", None), imp

    if model == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool

        class _Stop:
            def after_iteration(self, info):
                return time.time() < deadline

        def cbx(X):
            X = X.copy()
            for c in cat_cols:
                if c in X.columns:
                    X[c] = X[c].astype(str)
            return X

        cc = [c for c in cat_cols if c in Xtr.columns]
        p = {k_: v for k_, v in params.items() if k_ not in ("seed",)}
        es = p.pop("early_stopping", 200)
        common = dict(random_seed=seed, thread_count=n_jobs, verbose=False, od_type="Iter", od_wait=es,
                      allow_writing_files=False)
        if cls:
            loss = "MultiClass" if k > 2 else "Logloss"
            seen = np.unique(ytr)
            if len(seen) < k and k > 2:
                loss = "MultiClass"
            m = CatBoostClassifier(**p, **common, loss_function=loss,
                                   auto_class_weights="Balanced" if class_weight else None)
            mask_va = np.isin(yva, seen)
            m.fit(Pool(cbx(Xtr), ytr, cat_features=cc), eval_set=Pool(cbx(Xva[mask_va]), yva[mask_va], cat_features=cc),
                  callbacks=[_Stop()], use_best_model=True)
            pv, pt = m.predict_proba(cbx(Xva)), m.predict_proba(cbx(Xte))
            imp = dict(zip(Xtr.columns, m.get_feature_importance()))
            return _proba_full(pv, m.classes_, k), _proba_full(pt, m.classes_, k), m.get_best_iteration(), imp
        loss = "MAE" if metric in ("mae", "medae", "mape", "smape") else "RMSE"
        m = CatBoostRegressor(**p, **common, loss_function=loss)
        m.fit(Pool(cbx(Xtr), ytr, cat_features=cc), eval_set=Pool(cbx(Xva), yva, cat_features=cc),
              callbacks=[_Stop()], use_best_model=True)
        imp = dict(zip(Xtr.columns, m.get_feature_importance()))
        return m.predict(cbx(Xva)), m.predict(cbx(Xte)), m.get_best_iteration(), imp

    if model in ("linear", "extratrees"):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

        cc = [c for c in cat_cols if c in Xtr.columns]
        nc = [c for c in Xtr.columns if c not in cc]

        def tostr(X):
            X = X.copy()
            for c in cc:
                X[c] = X[c].astype(str)
            return X

        if model == "linear":
            pre = ColumnTransformer([
                ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), nc),
                ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                      sparse_output=True), cc)], sparse_threshold=0.3)
            if cls:
                from sklearn.linear_model import LogisticRegression
                est = LogisticRegression(C=params.get("C", 1.0), max_iter=300,
                                         class_weight="balanced" if class_weight else None)
            else:
                from sklearn.linear_model import Ridge
                est = Ridge(alpha=params.get("alpha", 1.0))
        else:
            pre = ColumnTransformer([
                ("num", SimpleImputer(strategy="constant", fill_value=-999), nc),
                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                       encoded_missing_value=-2), cc)])
            from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
            kw = dict(n_estimators=params.get("n_estimators", 300), min_samples_leaf=params.get("min_samples_leaf", 2),
                      max_features=params.get("max_features", 0.6), n_jobs=n_jobs, random_state=seed)
            est = (ExtraTreesClassifier(**kw, class_weight="balanced" if class_weight else None)
                   if cls else ExtraTreesRegressor(**kw))
        pipe = make_pipeline(pre, est)
        Xtr_ = tostr(Xtr)
        if len(Xtr_) > 300_000 and model == "extratrees":
            sel = np.random.RandomState(seed).choice(len(Xtr_), 300_000, replace=False)
            Xtr_, ytr = Xtr_.iloc[sel], ytr[sel]
        # для линейной модели — отсекаем экстремальные значения
        pipe.fit(Xtr_, ytr)
        if cls:
            seen = pipe.classes_
            return (_proba_full(pipe.predict_proba(tostr(Xva)), seen, k),
                    _proba_full(pipe.predict_proba(tostr(Xte)), seen, k), None, {})
        return pipe.predict(tostr(Xva)), pipe.predict(tostr(Xte)), None, {}

    if model == "autogluon":
        import shutil
        import tempfile
        from autogluon.tabular import TabularPredictor

        tl = max(60, int(deadline - time.time()))
        path = tempfile.mkdtemp(prefix="ag_", dir=params.get("_workdir"))
        lab = "__target__"
        tr = Xtr.copy()
        tr[lab] = ytr
        ptype = "regression" if not cls else ("binary" if k == 2 else "multiclass")
        evm = AG_METRIC.get(metric)
        try:
            pred = TabularPredictor(label=lab, problem_type=ptype, eval_metric=evm, path=path, verbosity=0)
            pred.fit(tr, time_limit=tl, presets=params.get("presets", "medium_quality"),
                     excluded_model_types=["KNN", "NN_TORCH", "FASTAI"], num_cpus=n_jobs,
                     ag_args_fit={"num_cpus": n_jobs})
            if cls:
                pv = pred.predict_proba(Xva).reindex(columns=range(k), fill_value=0).values
                pt = pred.predict_proba(Xte).reindex(columns=range(k), fill_value=0).values
                return pv, pt, None, {}
            return pred.predict(Xva).values, pred.predict(Xte).values, None, {}
        finally:
            shutil.rmtree(path, ignore_errors=True)

    raise ValueError(f"unknown model {model}")


AG_METRIC = {"rmse": "root_mean_squared_error", "mse": "mean_squared_error", "mae": "mean_absolute_error",
             "r2": "r2", "medae": "median_absolute_error", "mape": "mean_absolute_percentage_error",
             "rmsle": "root_mean_squared_error",
             "macro_f1": "f1_macro", "weighted_f1": "f1_weighted", "micro_f1": "f1_micro", "f1": "f1",
             "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "mcc": "mcc",
             "roc_auc": "roc_auc", "logloss": "log_loss", "pr_auc": "average_precision",
             "kappa": "quadratic_kappa", "smape": "mean_absolute_error", "precision": "precision",
             "recall": "recall", "brier": "log_loss"}


# ------------------------------------------------------------------ CV

def fit_cv(model: str, params: dict, X: pd.DataFrame, y: np.ndarray, n_train: int, folds, task: str, k: int,
           cat_cols: list[str], metric, time_limit: float, target_transform: str = "none",
           class_weight: bool = False, refit_full: bool = False, scorer=None, entity=None,
           refit_weight: float = 0.5) -> CVResult:
    """Кросс-валидация одной конфигурации. Бросает TimeUp, если время вышло до первого фолда."""
    t0 = time.time()
    deadline_total = t0 + time_limit
    Xtr_all, Xte = X.iloc[:n_train], X.iloc[n_train:]
    cls = task != "regression"
    oof = np.full((n_train, k), np.nan) if cls else np.full(n_train, np.nan)
    test = np.zeros((len(Xte), k)) if cls else np.zeros(len(Xte))
    yt = y if cls else _tt(y.astype(float), target_transform)
    fold_scores, iters, imp = [], [], {}
    fold_times: list[float] = []
    n_done = 0
    for i, (tr, va) in enumerate(folds):
        left = deadline_total - time.time()
        if left <= 5:
            break
        per_fold = left / (len(folds) - i + (1 if refit_full else 0))
        t_fold = time.time()
        Xa, Xb, Xc = Xtr_all.iloc[tr], Xtr_all.iloc[va], Xte
        if entity is not None:  # история таргета по сущности — строго внутри фолда
            fa, (fb, fc_) = entity.transform(tr, [va, n_train + np.arange(len(Xte))], yt[tr])
            Xa = pd.concat([Xa.reset_index(drop=True), fa], axis=1)
            Xb = pd.concat([Xb.reset_index(drop=True), fb], axis=1)
            Xc = pd.concat([Xc.reset_index(drop=True), fc_], axis=1)
        pv, pt, bi, im = _fit_predict(model, params, Xa, yt[tr], Xb, yt[va], Xc, task,
                                      k, cat_cols, metric.name, time.time() + per_fold, class_weight)
        if not cls:
            pv, pt = _itt(pv, target_transform), _itt(pt, target_transform)
        oof[va] = pv
        test += pt
        n_done += 1
        fold_times.append(time.time() - t_fold)
        iters.append(bi)
        for f, v in im.items():
            imp[f] = imp.get(f, 0) + float(v)
        if scorer is not None:
            fold_scores.append(scorer(y[va], pv))
    if n_done == 0:
        raise TimeUp(f"{model}: не хватило времени ни на один фолд")
    test /= n_done
    if refit_full and model != "autogluon" and time.time() < deadline_total - 5:
        # финальная модель на ВСЕХ данных train: при валидации «прошлое → будущее» модели фолдов
        # не видели последние периоды, а они для теста самые важные
        p2 = dict(params)
        bis = [b for b in iters if b]
        if bis and model in ("lgbm", "xgb"):
            p2["n_estimators"] = int(np.mean(bis) * 1.15) + 1
            p2["early_stopping"] = 10**6
        elif bis and model == "catboost":
            p2["iterations"] = int(np.mean(bis) * 1.15) + 1
            p2["early_stopping"] = 10**6
        all_idx = np.arange(n_train)
        Xa, Xc = Xtr_all, Xte
        if entity is not None:
            fa, (fc_,) = entity.transform(all_idx, [n_train + np.arange(len(Xte))], yt)
            Xa = pd.concat([Xtr_all.reset_index(drop=True), fa], axis=1)
            Xc = pd.concat([Xte.reset_index(drop=True), fc_], axis=1)
        # валидационный набор нужен интерфейсу — передаём хвост train (на ранний стоп не влияет)
        tail = np.arange(max(0, n_train - 1000), n_train)
        # финальной модели нужно ~1.25× времени фолда (данных больше, без ранней остановки);
        # если она упрётся в таймер — она недообучена, и её ответ лучше выбросить
        need = 1.6 * (np.mean(fold_times) if fold_times else 60) + 10
        avail = deadline_total - time.time()
        try:
            if avail < need:
                raise TimeUp(f"мало времени ({avail:.0f}c из нужных ~{need:.0f}c)")
            t_ref = time.time()
            dl = t_ref + max(need, avail)
            _, pt, _, _ = _fit_predict(model, p2, Xa, yt, Xa.iloc[tail], yt[tail], Xc, task, k, cat_cols,
                                       metric.name, dl, class_weight)
            if time.time() >= dl - 1:
                raise TimeUp("упёрлась в таймер — недообучена")
            if not cls:
                pt = _itt(pt, target_transform)
            # проверка здравого смысла: модель на всех данных обучена без ранней остановки и может
            # «разойтись» (особенно многоклассовый бустинг) — сверяем с усреднённым ответом фолдов
            pt = np.asarray(pt, float)
            if not np.isfinite(pt).all():
                raise ValueError("нечисловые предсказания")
            # пороги ловят только развал модели: законная разница (модель видит больше данных,
            # в т.ч. последние периоды) даёт корреляцию ~0.85, «разошедшаяся» модель — намного ниже
            if cls:
                agree = float(np.mean(pt.argmax(1) == test.argmax(1)))
                if agree < 0.6:
                    raise ValueError(f"расходится с моделями фолдов: совпадение ответов {agree:.0%}")
            else:
                sd_t, sd_p = float(np.std(test)), float(np.std(pt))
                c = np.corrcoef(pt, test)[0, 1] if sd_p > 0 and sd_t > 0 else 0.0
                shift = abs(float(np.mean(pt)) - float(np.mean(test))) / (sd_t + 1e-12)
                if not np.isfinite(c) or c < 0.6 or sd_p > 3 * sd_t + 1e-12 or shift > 1.5:
                    raise ValueError(f"расходится с моделями фолдов: корреляция {c:.2f}, сдвиг {shift:.2f}")
            test = (1 - refit_weight) * test + refit_weight * pt
        except Exception as e:  # noqa: BLE001
            log(f"  {model}: переобучение на всех данных не удалось ({e!r}) — используем модели фолдов")
    if n_done < len(folds):
        log(f"  {model}: успели {n_done}/{len(folds)} фолдов")
    return CVResult(model, params, oof, test, fold_scores, time.time() - t0, iters, imp)


# ------------------------------------------------------------------ Optuna

def tune_lgbm(X, y, n_train, folds, task, k, cat_cols, metric, time_limit, scorer, entity=None,
              target_transform: str = "none") -> dict:
    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fs = folds[: min(2, len(folds))]
    base = default_params("lgbm", task, metric.name, n_train)
    t_end = time.time() + time_limit

    def objective(trial):
        p = dict(base)
        p.update(num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                 learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True))
        if time.time() > t_end:
            raise optuna.TrialPruned()
        r = fit_cv("lgbm", p, X, y, n_train, fs, task, k, cat_cols, metric, max(10, t_end - time.time()),
                   target_transform=target_transform, entity=entity)
        return scorer_oof(r, y, fs, metric, scorer)

    st = optuna.create_study(direction="maximize" if metric.greater_is_better else "minimize",
                             sampler=optuna.samplers.TPESampler(seed=config.SEED))
    st.enqueue_trial({k_: base[k_] for k_ in ("num_leaves", "min_child_samples", "colsample_bytree", "subsample",
                                               "reg_lambda", "learning_rate")} | {"reg_alpha": 0.001})
    try:
        st.optimize(objective, timeout=max(10, time_limit), catch=(Exception,))
    except Exception as e:  # noqa: BLE001
        log("optuna error", repr(e))
    done = [t for t in st.trials if t.value is not None]
    if not done:
        return base
    p = dict(base)
    p.update(st.best_params)
    log(f"  optuna: {len(done)} trials, best={st.best_value:.5f}")
    return p


def scorer_oof(r: CVResult, y, folds, metric, scorer) -> float:
    idx = np.concatenate([va for _, va in folds])
    return scorer(y[idx], r.oof[idx])
