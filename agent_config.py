"""Глобальные настройки и таймер агента.

Бюджет по умолчанию — 2 ч 15 мин из 3 часов платформы. Его можно переопределить
переменной окружения AGENT_TIME_BUDGET (секунды) — удобно для быстрых локальных тестов.
"""
from __future__ import annotations

import os
import time

START_TIME = time.time()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


TIME_BUDGET = _env_float("AGENT_TIME_BUDGET", 2 * 3600 + 40 * 60)  # 9600 c из 3 ч платформы
# Доли бюджета: исследование заканчивается раньше, чтобы осталось время на ансамбль и запись.
RESEARCH_END_FRAC = 0.93  # ~9.5 мин резерва из 2 ч 15 мин на ансамбль и запись
def _available_cpus() -> int:
    """Реально доступные ядра. os.cpu_count() в Docker часто видит все ядра сервера (32–128), а не лимит
    контейнера (у платформы — 8): тогда библиотеки создают десятки потоков и работают в разы медленнее."""
    n = os.cpu_count() or 1
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    quota = None
    try:  # cgroup v2: "max 100000" или "800000 100000"
        with open("/sys/fs/cgroup/cpu.max") as f:
            q, p = f.read().split()[:2]
            if q != "max":
                quota = float(q) / float(p)
    except (OSError, ValueError):
        try:  # cgroup v1
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                q = float(f.read())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                p = float(f.read())
            if q > 0 and p > 0:
                quota = q / p
        except (OSError, ValueError):
            pass
    if quota:
        n = min(n, max(1, int(quota + 0.5)))
    return max(1, min(n, 8))  # по условию у задачи 8 CPU — больше потоков не нужно


N_JOBS = max(1, int(_env_float("AGENT_N_JOBS", _available_cpus())))
SEED = 42
MAX_CANDIDATES = int(_env_float("AGENT_MAX_CANDIDATES", 25))  # сколько кандидатов держать для ансамбля

# Варианты для сабмитов (меняются только при сборке вариантов)
REFIT_BLEND = 0.0            # доля модели на 100% данных в ответе при обычной (не временной) валидации
USE_CLASS_THRESHOLDS = True  # подбор множителей классов под label-метрики (Macro F1 и т.п.)
ENTITY_RANDOM_SPLIT = True   # признаки истории объектов и при случайном разбиении train/test
STRICT_ACCEPT = False        # принимать изменения только при устойчивом приросте по фолдам
PRIOR_SHIFT = False          # EM-поправка вероятностей на сдвиг долей классов train→test
ENSEMBLE_MODE = "auto"       # auto (регрессия topk, классификация greedy) | greedy | topk | mix
ENSEMBLE_TOPK = 10
ENSEMBLE_TOL_REG = 0.03       # в среднее берутся модели не хуже лучшей на 3% (регрессия)
ENSEMBLE_TOL_CLS = 0.01       # и на 1% (классификация)
CLASS_WEIGHT_RANGE = (0.05, 20.0)  # пределы множителей классов при подборе порогов
THRESH_VETO = False          # True — отключать пороги, если они не дают прироста на отложенной половине OOF

# LLM
LLM_MAX_CALLS = int(_env_float("AGENT_LLM_MAX_CALLS", 40))
LLM_TIMEOUT = _env_float("AGENT_LLM_TIMEOUT", 120)
# доля бюджета, которую агент готов суммарно ждать ответов LLM
LLM_TIME_SHARE = _env_float("AGENT_LLM_TIME_SHARE", 0.25)
LLM_MAX_TOKENS = int(_env_float("AGENT_LLM_MAX_TOKENS", 7000))  # Qwen тратит часть на рассуждения
# Принудительно отключить LLM (например, для отладки): AGENT_NO_LLM=1
NO_LLM = os.environ.get("AGENT_NO_LLM", "0") == "1"

# Песочница для кода, который пишет LLM
SANDBOX_TIMEOUT = _env_float("AGENT_SANDBOX_TIMEOUT", 600)


def elapsed() -> float:
    return time.time() - START_TIME


def remaining() -> float:
    return TIME_BUDGET - elapsed()


def research_remaining() -> float:
    """Сколько секунд осталось до конца фазы исследования."""
    return TIME_BUDGET * RESEARCH_END_FRAC - elapsed()
