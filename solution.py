"""Точка входа AI-агента для решения ML-задач (хакатон «Альфа-Будущее»).

Запуск (как в baseline.py):
    OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... python solution.py task.md

Агент читает task.md, сам находит данные, определяет тип задачи и метрику, строит
валидацию, проводит цикл AutoResearch с LLM и пишет submission.csv рядом с task.md.
Файл ответа существует с первых секунд и обновляется только при улучшении.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

# импорт модулей агента из каталога архива, откуда бы ни запускали
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent_config as config  # noqa: E402
from agent_log import log  # noqa: E402


def main(md_arg: str) -> int:
    md_path = Path(md_arg).resolve()
    log(f"старт: {md_path}, CPU={config.N_JOBS}, бюджет={config.TIME_BUDGET / 60:.0f} мин")
    from agent_task import parse_task
    from agent_output import constant_pred
    from agent_writer import SubmissionWriter
    from agent_llm import LLM
    from agent_research import Agent

    spec, train, test, sample = parse_task(md_path)
    writer = SubmissionWriter(spec, test, sample)
    targets = [spec] + [_spec_for(spec, t, train) for t in spec.other_targets]
    if len(targets) > 1:
        log(f"несколько таргетов: {[s.target for s in targets]}")
    # самый первый ответ — константа по каждому таргету: файл существует с первых секунд
    for sp in targets:
        try:
            writer.update(sp, constant_pred(sp, train, test), None, tag="константа")
        except Exception as e:  # noqa: BLE001
            log(f"константный ответ для `{sp.target}` не записан:", repr(e))
    llm = LLM()
    for i, sp in enumerate(targets):
        share_end = config.START_TIME + config.TIME_BUDGET * (i + 1) / len(targets)
        try:
            agent = Agent(sp, train, test, sample, writer, llm=llm, time_end=share_end if len(targets) > 1 else None)
            agent.run()
        except Exception as e:  # noqa: BLE001
            log(f"агент для `{sp.target}` упал: {e!r}")
            log(traceback.format_exc()[-3000:])
            if sp.target not in writer.frames:
                _emergency(sp, train, test, writer)
    log(f"готово за {config.elapsed() / 60:.1f} мин; LLM вызовов {llm.calls}")
    return 0


def _spec_for(spec, target, train):
    """Спецификация для дополнительного таргета (мульти-таргет)."""
    import copy
    import numpy as np
    s = copy.copy(spec)
    s.target = target
    s.other_targets = [t for t in [spec.target] + spec.other_targets if t != target]
    y = train[target]
    if y.dtype == object or y.nunique() <= 2:
        s.task_type = "binary" if y.nunique() <= 2 else "multiclass"
        s.classes = sorted(y.dropna().unique().tolist(), key=str)
        s.output_mode = "label"
        s.metric = spec.metric if spec.task_type != "regression" else "accuracy"
    else:
        s.task_type, s.classes, s.output_mode = "regression", [], "value"
        s.metric = spec.metric if spec.task_type == "regression" else "rmse"
    return s


def _emergency(sp, train, test, writer):
    """Последний рубеж: простая модель без исследований."""
    try:
        import numpy as np
        from agent_features import base_frame
        from agent_models import fit_cv, default_params
        from agent_metrics import get_metric
        from agent_validation import make_folds
        tr = train[train[sp.target].notna()].reset_index(drop=True)
        base, cats, _ = base_frame(tr, test, sp.target, sp.id_col, sp.other_targets)
        if sp.task_type == "regression":
            y, k = tr[sp.target].astype(float).values, 1
        else:
            idx = {c: i for i, c in enumerate(sp.classes)}
            y, k = tr[sp.target].map(idx).astype(int).values, len(sp.classes)
        folds, _ = make_folds(tr, y, sp.task_type, "kfold" if sp.task_type == "regression" else "stratified", 3)
        r = fit_cv("lgbm", default_params("lgbm", sp.task_type, sp.metric, len(tr)), base, y, len(tr), folds,
                   sp.task_type, k, cats, get_metric(sp.metric), max(60, config.remaining() - 60))
        writer.update(sp, r.test, None, tag="аварийная модель")
    except Exception as e:  # noqa: BLE001
        log("аварийная модель тоже упала:", repr(e))


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "task.md"
    try:
        code = main(arg)
    except Exception as e:  # noqa: BLE001
        log("ФАТАЛЬНАЯ ОШИБКА:", repr(e))
        log(traceback.format_exc()[-3000:])
        code = 0  # файл ответа, если он уже записан, остаётся валидным
    sys.stdout.flush()
    os._exit(code)  # не ждём фоновые потоки библиотек
