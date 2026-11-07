"""Держит актуальные предсказания по каждому таргету и перезаписывает файл ответа."""
from __future__ import annotations

import pandas as pd

from agent_output import build_submission, conform_to_sample, write_submission


class SubmissionWriter:
    def __init__(self, main_spec, test: pd.DataFrame, sample: pd.DataFrame | None):
        self.main, self.test, self.sample = main_spec, test, sample
        self.frames: dict[str, pd.DataFrame] = {}

    def update(self, spec, pred, weights=None, tag: str = "") -> bool:
        part = build_submission(spec, self.test, pred, None, weights)
        self.frames[spec.target] = part
        return self.flush(tag)

    def flush(self, tag: str = "") -> bool:
        idc = self.main.id_col or "id"
        main = self.frames.get(self.main.target)
        if main is None:
            return False
        out = main.copy()
        for t, fr in self.frames.items():
            if t == self.main.target:
                continue
            for c in fr.columns:
                if c != idc:
                    out[c] = fr[c].values
        if any(t not in self.frames for t in [self.main.target, *self.main.other_targets]):
            return False  # пока не у всех таргетов есть ответ — файл не трогаем
        out = conform_to_sample(self.main, out, self.sample)
        id_name = str(self.sample.columns[0]) if self.sample is not None and \
            str(self.sample.columns[0]) in out.columns else None
        return write_submission(self.main, out, self.test, tag, id_name)
