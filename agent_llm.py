"""Клиент LLM (OpenAI-совместимый API, Qwen).

Устойчивость важнее всего: таймауты, повторы, вырезание <think>, извлечение
JSON из «грязного» ответа, учёт токенов и автоматическое отключение после
серии сбоев — тогда агент продолжает на детерминированных эвристиках.
"""
from __future__ import annotations

import ast
import json
import os
import re
import time

import agent_config as config
from agent_log import log


class LLM:
    def __init__(self):
        self.enabled = False
        self.calls = 0
        self.fail_streak = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.max_total_tokens = int(os.environ.get("AGENT_LLM_MAX_TOTAL_TOKENS", 600_000))
        self.no_think_ok = True  # пробуем выключить «размышления» Qwen; если API не примет — уберём
        self.cooldown_until = 0.0
        self.time_spent = 0.0
        self.effort_ok = True  # просим модель думать меньше; если шлюз не примет параметр — уберём
        self.model = os.environ.get("OPENAI_MODEL")
        if config.NO_LLM:
            log("LLM отключена флагом AGENT_NO_LLM")
            return
        if not os.environ.get("OPENAI_API_KEY") or not self.model:
            log("LLM недоступна: нет OPENAI_API_KEY/OPENAI_MODEL — работаем на эвристиках")
            return
        try:
            from openai import OpenAI
            self.client = OpenAI(timeout=config.LLM_TIMEOUT, max_retries=0)
            self.enabled = True
            log(f"LLM: модель {self.model}")
        except Exception as e:  # noqa: BLE001
            log("LLM init error:", repr(e))

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, max_tokens: int | None = None, temperature: float = 0.4,
             timeout: float | None = None) -> str | None:
        """Один запрос к LLM с жёсткими лимитами времени. Никогда не бросает исключений."""
        if not self.enabled:
            return None
        now = time.time()
        if now < self.cooldown_until:
            return None
        if self.calls >= config.LLM_MAX_CALLS or self.tokens_in + self.tokens_out > self.max_total_tokens:
            log("LLM: исчерпан лимит вызовов/токенов")
            self.enabled = False
            return None
        if self.time_spent > config.LLM_TIME_SHARE * config.TIME_BUDGET:
            log("LLM: суммарное время ожидания LLM превысило лимит — дальше без LLM")
            self.enabled = False
            return None
        if config.remaining() < 180:
            return None
        self.calls += 1
        # Qwen3: /no_think отключает «размышления», которые съедают лимит токенов и время
        msgs = [{"role": "system", "content": system + "\n/no_think"}, {"role": "user", "content": user}]
        mt = max_tokens or config.LLM_MAX_TOKENS
        tmo = min(timeout or config.LLM_TIMEOUT, max(30, config.remaining() - 120))
        for attempt in range(2):
            t0 = time.time()
            try:
                extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}} if self.no_think_ok else {}
                if self.effort_ok:
                    extra["reasoning_effort"] = "low"
                r = self.client.chat.completions.create(model=self.model, messages=msgs, temperature=temperature,
                                                        max_tokens=mt, timeout=tmo, **extra)
                self.time_spent += time.time() - t0
                ch = r.choices[0]
                raw = ch.message.content or ""
                if getattr(r, "usage", None):
                    self.tokens_in += r.usage.prompt_tokens or 0
                    self.tokens_out += r.usage.completion_tokens or 0
                txt = strip_think(raw)
                fin = getattr(ch, "finish_reason", None)
                log(f"LLM ok за {time.time() - t0:.0f}c (finish={fin}, символов {len(txt)}), "
                    f"токены всего in={self.tokens_in} out={self.tokens_out}")
                if fin == "length" and attempt == 0 and mt < 16000 and config.remaining() > 600 \
                        and (len(txt) < 200 or not txt.rstrip().endswith("}")):
                    log("LLM: ответ обрезан (модель долго рассуждает) — переспрашиваю с большим лимитом")
                    mt = min(16000, int(mt * 1.7))
                    continue
                if not txt.strip():
                    log("LLM: пустой ответ")
                    if attempt == 0:
                        continue
                    return None
                self.fail_streak = 0
                return txt
            except Exception as e:  # noqa: BLE001
                self.time_spent += time.time() - t0
                msg = repr(e)
                if self.effort_ok and "reasoning" in msg.lower() and attempt == 0:
                    self.effort_ok = False  # шлюз не понимает reasoning_effort — повторяем без него
                    log("LLM: параметр reasoning_effort не поддерживается — убираю")
                    continue
                if self.no_think_ok and attempt == 0 and ("chat_template_kwargs" in msg or "extra_body" in msg
                                                          or "unexpected keyword" in msg.lower()):
                    self.no_think_ok = False  # провайдер не понимает параметр — повторяем без него
                    continue
                log(f"LLM ошибка (попытка {attempt + 1}, {time.time() - t0:.0f}c):", msg[:300])
                if "Timeout" in msg or "timed out" in msg.lower():
                    # сервер не успевает — не долбим его, делаем паузу и работаем без LLM
                    self.cooldown_until = time.time() + 300
                    log("LLM: таймаут — пауза 5 минут в обращениях к LLM")
                    break
                if "Budget" in msg or "budget" in msg or "401" in msg or "403" in msg:
                    log("LLM: доступ/бюджет недоступен — отключаю LLM до конца запуска")
                    self.enabled = False
                    return None
                time.sleep(3)
        self.fail_streak += 1
        if self.fail_streak >= 3:
            log("LLM: 3 сбоя подряд — отключаю LLM до конца запуска")
            self.enabled = False
        return None

    def chat_json(self, system: str, user: str, expect: tuple = (), list_key: str | None = None,
                  **kw) -> dict | None:
        """JSON-ответ. expect — ключи, которые должны быть в ответе; list_key — куда собрать
        уцелевшие действия, если ответ обрезан (из полных вложенных объектов с полем "type")."""
        txt = self.chat(system, user, **kw)
        if txt is None:
            return None
        obj = extract_json(txt)
        if obj is not None and (not expect or any(k in obj for k in expect)):
            return obj
        if list_key:  # ответ обрезан или разобран не тот объект — спасаем полные действия
            acts = salvage_actions(txt)
            if acts:
                log(f"LLM: ответ неполный — спасено действий: {len(acts)}")
                return {list_key: acts, "_salvaged": True}
        if obj is None:
            log("LLM: не удалось разобрать JSON, прошу исправить")
            fix = self.chat("Ты возвращаешь только валидный JSON без пояснений.",
                            "Исправь так, чтобы это был один валидный JSON-объект:\n" + txt[:6000],
                            max_tokens=kw.get("max_tokens"))
            obj = extract_json(fix or "")
        return obj


def strip_think(txt: str) -> str:
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    if "<think>" in txt and "</think>" not in txt:  # оборванное размышление
        txt = txt.split("<think>")[0]
    return txt.strip()


def extract_json(txt: str):
    if not txt:
        return None
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", txt, flags=re.S)
    cands = blocks + [txt]
    for c in cands:
        start = c.find("{")
        while start != -1:
            depth, ins, esc = 0, False, False
            for i in range(start, len(c)):
                ch = c[i]
                if ins:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        ins = False
                    continue
                if ch == '"':
                    ins = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        frag = c[start:i + 1]
                        for parse in (json.loads, _loose):
                            try:
                                obj = parse(frag)
                                if isinstance(obj, dict):
                                    return obj
                            except Exception:  # noqa: BLE001
                                pass
                        break
            start = c.find("{", start + 1)
    return None


def _loose(s: str):
    s2 = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        return json.loads(s2)
    except Exception:  # noqa: BLE001
        return ast.literal_eval(s2.replace("true", "True").replace("false", "False").replace("null", "None"))


def salvage_actions(txt: str) -> list[dict]:
    """Все полные JSON-объекты с полем "type" (гипотезы/действия) из возможно обрезанного текста."""
    out, seen = [], set()
    for i, ch in enumerate(txt or ""):
        if ch != "{":
            continue
        obj = extract_json(txt[i:])
        if isinstance(obj, dict) and isinstance(obj.get("type"), str):
            key = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
            if key not in seen:
                seen.add(key)
                out.append(obj)
    return out[:10]


def extract_code(txt: str) -> str | None:
    m = re.findall(r"```(?:python)?\s*(.*?)```", txt or "", flags=re.S)
    for c in m:
        if "def add_features" in c:
            return c
    if txt and "def add_features" in txt:
        return txt[txt.find("def add_features"):]
    return None
