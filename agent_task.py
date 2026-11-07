"""Разбор условия: поиск файлов, id, таргета, типа задачи, метрики и формата ответа.

Всё определяется эвристиками по тексту task.md и самим данным; LLM потом может
уточнить спецификацию, но эвристика остаётся страховкой и валидатором.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from agent_log import log

ROLE_PATTERNS = {
    "sample": re.compile(r"sample|submission|submit|шаблон", re.I),
    "dict": re.compile(r"dict|description|schema|словар|описан", re.I),
    "test": re.compile(r"test", re.I),
    "train": re.compile(r"train", re.I),
}


@dataclass
class TaskSpec:
    md_path: str
    md_text: str
    workdir: str
    train_path: str
    test_path: str
    sample_path: str | None
    dict_path: str | None
    id_col: str | None
    target: str
    task_type: str  # regression | binary | multiclass
    metric: str
    output_path: str
    output_cols: list[str]
    output_mode: str = "value"  # value | label | proba_single | proba_multi
    classes: list = field(default_factory=list)
    positive_class: object = None
    other_targets: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def short(self) -> dict:
        d = asdict(self)
        d.pop("md_text")
        return d


# ------------------------------------------------------------------ файлы

def _find_csvs(md_path: Path, text: str) -> dict[str, Path]:
    base = md_path.parent
    found: dict[str, Path] = {}
    all_csv = sorted(p for p in base.rglob("*.csv") if "_agent_work" not in p.parts)
    if not any(ROLE_PATTERNS["train"].search(p.stem) for p in all_csv):
        # данные могут лежать уровнем выше (task.md во вложенной папке)
        all_csv += sorted(p for p in base.parent.glob("*/*.csv") if p not in all_csv)
        all_csv += sorted(base.parent.glob("*.csv"))
    mentioned = re.findall(r"[\w./\\-]+\.csv", text)
    candidates: list[Path] = []
    for m in mentioned:
        p = (base / m.replace("\\", "/")).resolve()
        if p.exists():
            candidates.append(p)
        else:  # путь в описании может не совпадать со структурой каталогов
            name = Path(m).name
            same = [q for q in all_csv if q.name == name]
            candidates.extend(same[:1])
    candidates.extend(all_csv)
    for p in candidates:
        n = p.stem.lower()
        if n in {"submission", "predictions", "prediction"} and "sample" not in n:
            continue  # это файл ответа, а не вход
        for role in ("sample", "dict", "test", "train"):
            if ROLE_PATTERNS[role].search(n):
                found.setdefault(role, p)
                break
    return found


def _output_name(text: str, inputs: set[str]) -> str:
    sec = re.split(r"(?im)^#+\s*(формат\s+результата|формат\s+ответа|результат|output|submission)", text)
    scope = sec[-1] if len(sec) > 1 else text
    if len(sec) > 1:  # имя ищем только в разделе про формат результата
        for m in re.findall(r"([\w.-]+\.csv)", scope):
            if m not in inputs and "sample" not in m.lower():
                return m
    return "submission.csv"


# ------------------------------------------------------------------ метрика

METRIC_PATTERNS = [  # порядок важен: более специфичные раньше
    ("rmsle", r"rmsle|root mean squared log|msle"),
    ("smape", r"smape|symmetric mean absolute percentage|симметричн\w* средн\w* абсолютн\w* процент"),
    ("brier", r"brier|брайер"),
    ("macro_f1", r"macro[\s_-]*f1|f1[\s_-]*macro|average\s*=\s*[\"']macro"),
    ("weighted_f1", r"weighted[\s_-]*f1|f1[\s_-]*weighted"),
    ("micro_f1", r"micro[\s_-]*f1|f1[\s_-]*micro"),
    ("balanced_accuracy", r"balanced[\s_-]*accuracy"),
    ("roc_auc", r"roc[\s_-]*auc|\bauc\b|gini|джини"),
    ("pr_auc", r"pr[\s_-]*auc|average precision"),
    ("logloss", r"log[\s_-]*loss|cross[\s_-]*entropy|логлосс|логарифмическ\w* потер"),
    ("mcc", r"\bmcc\b|matthews"),
    ("kappa", r"kappa|каппа"),
    ("rmse", r"\brmse\b|root mean squared error|корень из среднеквадрат"),
    ("mse", r"\bmse\b|mean squared error|среднеквадратичн"),
    ("mape", r"\bmape\b|mean absolute percentage"),
    ("mae", r"\bmae\b|mean absolute error|средн\w* абсолютн"),
    ("medae", r"median absolute"),
    ("r2", r"\br2\b|r\^2|r²|коэффициент детерминации"),
    ("f1", r"\bf1\b|f-score|f-мера"),
    ("precision", r"\bprecision\b"),
    ("recall", r"\brecall\b|полнот[аыу]"),
    ("accuracy", r"accuracy|точность классификации|доля верных"),
]


def detect_metric(text: str) -> str | None:
    low = text.lower()
    # сначала ищем в предложениях, где явно названа основная метрика
    key = re.findall(r"[^.\n]*(?:основн\w* метрик|метрик\w* оценки|metric|метрика)[^.\n]*(?:\n[^\n#]*){0,2}", low)
    for scope in key + [low]:
        hits = []
        for name, pat in METRIC_PATTERNS:
            m = re.search(pat, scope)
            if m:
                hits.append((m.start(), name))
        if hits:
            # ближайшая к слову «метрика» / первая по тексту
            hits.sort()
            return hits[0][1]
    return None


# ------------------------------------------------------------------ основной разбор

def parse_task(md_path: Path) -> tuple[TaskSpec, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    md_path = md_path.resolve()
    text = md_path.read_text(encoding="utf-8", errors="replace")
    files = _find_csvs(md_path, text)
    log("Найденные файлы:", {k: str(v.relative_to(md_path.parent)) for k, v in files.items()})
    if "train" not in files or "test" not in files:
        raise FileNotFoundError(f"Не найдены train/test рядом с {md_path}: {files}")

    read = dict(low_memory=False)
    train = pd.read_csv(files["train"], **read)
    test = pd.read_csv(files["test"], **read)
    sample = pd.read_csv(files["sample"], **read) if "sample" in files else None
    notes: list[str] = []

    # --- id
    id_col = None
    if sample is not None and sample.columns[0] in test.columns:
        id_col = sample.columns[0]
    else:
        for c in test.columns:
            if c.lower() in ("id", "row_id", "index", "uid") or c.lower().endswith("_id") and test[c].is_unique:
                id_col = c
                break
    # --- таргет
    extra = [c for c in train.columns if c not in test.columns]
    if sample is not None:
        out_cols = list(sample.columns)
        tcols = [c for c in out_cols if c != id_col]
        in_train = [c for c in tcols if c in train.columns]
        if in_train:
            target, others = in_train[0], in_train[1:]
        elif len(extra) == 1:
            target, others = extra[0], []
        else:
            target, others = (extra[0] if extra else tcols[0]), []
    else:
        others = []
        target = extra[0] if extra else train.columns[-1]
        out_cols = [c for c in [id_col, target] if c]
    if target not in train.columns:
        raise ValueError(f"Таргет {target} не найден в train")

    metric = detect_metric(text)
    y = train[target]
    nun = y.nunique(dropna=True)
    low = text.lower()
    is_cls_text = bool(re.search(r"классификац|classif|класс\w* |категори\w* атак|label", low))
    is_reg_text = bool(re.search(r"регресс|regress", low))
    if metric and METRIC_KIND[metric] == "regression":
        task_type = "regression"
    elif metric and METRIC_KIND[metric] == "classification":
        task_type = "binary" if nun == 2 else "multiclass"
    elif y.dtype == object or y.dtype == bool or str(y.dtype) == "category":
        task_type = "binary" if nun == 2 else "multiclass"
    elif is_reg_text and not is_cls_text:
        task_type = "regression"
    elif nun <= 2:
        task_type = "binary"
    elif is_cls_text and nun <= 50 and np.allclose(y.dropna() % 1, 0):
        task_type = "multiclass"
    else:
        task_type = "regression"
    metric_defaulted = metric is None
    if metric is None:
        metric = {"regression": "rmse", "binary": "roc_auc", "multiclass": "macro_f1"}[task_type]
        notes.append(f"метрика не найдена в тексте, по умолчанию {metric}")

    # --- формат ответа
    classes = []
    if task_type != "regression":
        try:
            classes = sorted(y.dropna().unique().tolist())
        except TypeError:
            classes = sorted(y.dropna().unique().tolist(), key=lambda v: str(v))
        classes = [c.item() if hasattr(c, "item") else c for c in classes]
    positive = None
    mode = "value"
    if task_type != "regression":
        cls_names = {str(c) for c in classes}
        tcols = [c for c in out_cols if c != id_col]
        if len(tcols) > 1 and set(tcols) >= cls_names:
            mode = "proba_multi"
        elif METRICS_NEEDS[metric] == "proba":
            mode = "proba_single" if task_type == "binary" else "proba_multi"
        else:
            mode = "label"
        wants_proba = bool(re.search(r"вероятност|probabilit|predict_proba|score от 0 до 1|confidence", low))
        if sample is not None and mode == "label" and task_type == "binary":
            # вероятности вместо меток — только если метрика не названа явно или текст просит вероятность
            sv = sample[target] if target in sample.columns else None
            float_tpl = sv is not None and pd.api.types.is_float_dtype(sv) and not set(sv.dropna().unique()) <= {0.0, 1.0}
            if (float_tpl and metric_defaulted) or wants_proba and METRICS_NEEDS[metric] != "label":
                mode = "proba_single"
        if task_type == "binary":
            positive = detect_positive(classes, y, target, text)
        if mode == "proba_multi" and len(tcols) == 1:
            out_cols = [id_col] + [str(c) for c in classes] if id_col else [str(c) for c in classes]
            notes.append("multiclass+proba-метрика при одной колонке в шаблоне — выводим вероятности по классам")

    inputs = {p.name for p in files.values()}
    out_name = _output_name(text, inputs)
    spec = TaskSpec(
        md_path=str(md_path), md_text=text, workdir=str(md_path.parent),
        train_path=str(files["train"]), test_path=str(files["test"]),
        sample_path=str(files["sample"]) if "sample" in files else None,
        dict_path=str(files["dict"]) if "dict" in files else None,
        id_col=id_col, target=target, task_type=task_type, metric=metric,
        output_path=str(md_path.parent / out_name), output_cols=out_cols,
        output_mode=mode, classes=classes, positive_class=positive,
        other_targets=others, notes=notes,
    )
    log("Спецификация:", {k: v for k, v in spec.short().items() if k not in ("classes",)},
        "classes:", classes[:20])
    return spec, train, test, sample


POSITIVE_TOKENS = ("1", "1.0", "true", "yes", "y", "да", "positive", "pos", "fraud", "churn", "default",
                   "attack", "bad", "spam", "malignant", "fake", "anomaly", "outlier", "event", "sick", "defect",
                   "failure", "fail", "approved", "click", "clicked", "buy", "purchase", "response")


def detect_positive(classes: list, y: pd.Series, target: str, text: str):
    """Положительный класс бинарной задачи: явное упоминание в условии > типичные имена > миноритарный класс."""
    low = text.lower()
    names = {str(c).lower(): c for c in classes}
    # 1) явные конструкции: «по классу X», «positive class X», «`target` = X», «probability of X»
    for key, c in names.items():
        k = re.escape(key)
        pats = [rf"(по\s+классу|положительн\w*\s+класс\w*|positive\s+class|класс\w*)\s*[`'\"«(]*{k}[`'\"»)]*(\W|$)",
                rf"`?{re.escape(str(target).lower())}`?\s*(==|=)\s*[`'\"]*{k}[`'\"]*(\W|$)",
                rf"(probability\s+of|вероятност\w*)\s+(класса\s+)?[`'\"«]*{k}[`'\"»]*(\W|$)"]
        if any(re.search(pt, low) for pt in pats):
            return c
    # 2) типичные «положительные» имена
    for tok in POSITIVE_TOKENS:
        if tok in names:
            return names[tok]
    # 3) иначе — более редкий класс (обычно именно он «событие»)
    vc = y.astype(str).value_counts()
    rare = vc.index[-1]
    return names.get(rare.lower(), classes[-1])


from agent_metrics import METRICS as _M  # noqa: E402

METRIC_KIND = {k: v.kind for k, v in _M.items()}
METRICS_NEEDS = {k: v.needs for k, v in _M.items()}
