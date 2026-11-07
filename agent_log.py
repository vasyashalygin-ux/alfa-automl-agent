"""Простое логирование в stdout с отметкой времени от старта."""
from __future__ import annotations

import sys

import agent_config as config


def log(*args) -> None:
    t = config.elapsed()
    msg = " ".join(str(a) for a in args)
    print(f"[{int(t // 60):3d}:{int(t % 60):02d}] {msg}", flush=True)
    sys.stdout.flush()
