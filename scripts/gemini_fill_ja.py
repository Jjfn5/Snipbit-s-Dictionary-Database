#!/usr/bin/env python3
"""Fills gaps in the Japanese example corpus with Gemini-written sentences.

Candidates = common JMdict headwords that have NO Tatoeba example and no
Gemini example yet. Words are sent in batches (many words per request, to
stay inside free-tier request quotas). Every sentence is validated before
it is stored; results are appended to examples/ja/gemini.jsonl (append-only:
the line number becomes the sentence id, so never reorder or delete lines).

Needs env GEMINI_API_KEY. GEMINI_MODEL picks the model (see README).
Without a key this exits 0 and does nothing, so the Tatoeba-only pipeline
keeps working.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

MAX_ATTEMPTS = 3
# Tried in this order; anything unavailable/overloaded/out of quota is
# skipped automatically. Override with the repo variable GEMINI_MODELS
# (comma-separated) -- see parse_models().
DEFAULT_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]
# Words that mark a listed model as not being a plain text model.
NOT_TEXT_MODEL = ("tts", "image", "live", "audio", "embed", "computer", "robotics",
                  "antigravity", "research", "lyria", "transcribe", "omni", "gemma",
                  "nano", "banana", "pro", "customtools", "argon")
API = "https://generativelanguage.googleapis.com/v1beta"
VERB_ADJ_TAGS = ("v1", "v5", "vk", "vz", "vs-i", "vs-s", "adj-i")
BAD_CHARS = set("[]{}()（）「」『』<>")

PROMPT_HEAD = """You write example sentences for a Japanese dictionary app used by learners.
For each numbered word below, write ONE natural Japanese sentence that uses the word in the meaning given.

Rules:
- 10 to 30 characters, everyday register, simple grammar and vocabulary apart from the target word.
- Use the target word itself (conjugated or inflected if that is natural). Put the exact form you used in "surface".
- No furigana, no romaji, no brackets or quotation marks in the sentence.
- "translation" is a natural English translation of the sentence.
- Return JSON only: an array with one object per word: {"index": n, "sentence": "...", "surface": "...", "translation": "..."}.

Words:
"""

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "index": {"type": "INTEGER"},
            "sentence": {"type": "STRING"},
            "surface": {"type": "STRING"},
            "translation": {"type": "STRING"},
        },
        "required": ["index", "sentence", "surface", "translation"],
    },
}


class QuotaExhausted(Exception):
    """No model is usable right now (all gone, rate-limited or overloaded)."""


class FatalError(Exception):
    """Problem no other model can fix (e.g. a bad API key)."""


class ModelGone(Exception):
    """This model doesn't exist / isn't allowed for this key: skip it for the whole run."""


class RateLimited(Exception):
    def __init__(self, msg, retry_after=None, per_day=False):
        super().__init__(msg)
        self.retry_after = retry_after
        self.per_day = per_day


class Overloaded(Exception):
    """Temporary server trouble (5xx, timeout, network): retry, then rest the model."""


# ------------------------------------------------------------------ helpers
def has_japanese(s: str) -> bool:
    return any(
        0x4E00 <= ord(c) <= 0x9FFF or 0x3040 <= ord(c) <= 0x309F or 0x30A0 <= ord(c) <= 0x30FF
        for c in s
    )


def word_stem(headword: str, pos: str) -> str:
    """Part of the headword that survives conjugation (食べる -> 食べ)."""
    tags = [t.strip() for t in (pos or "").split(",")]
    if len(headword) >= 2 and any(t.startswith(VERB_ADJ_TAGS) for t in tags):
        return headword[:-1]
    return headword


def first_glosses(meaning_json: str, limit: int = 3) -> str:
    try:
        groups = json.loads(meaning_json)
        return "; ".join(groups[0]["g"][:limit])
    except (ValueError, KeyError, IndexError, TypeError):
        return ""


def validate(entry: dict, item: dict) -> Optional[str]:
    """Returns None if the item is acceptable, else a short reject reason."""
    sentence = item.get("sentence")
    translation = item.get("translation")
    surface = item.get("surface")
    if not all(isinstance(x, str) for x in (sentence, translation, surface)):
        return "missing fields"
    sentence, translation, surface = sentence.strip(), translation.strip(), surface.strip()
    if not (4 <= len(sentence) <= 80):
        return "bad length"
    if not (3 <= len(translation) <= 250):
        return "bad translation length"
    if not has_japanese(sentence):
        return "not japanese"
    if any(c in BAD_CHARS for c in sentence):
        return "brackets/quotes in sentence"
    if not surface or surface not in sentence:
        return "surface not in sentence"
    if word_stem(entry["headword"], entry["pos"]) not in sentence:
        return "target word not in sentence"
    return None


def build_prompt(batch: list[dict]) -> str:
    lines = []
    for i, e in enumerate(batch, 1):
        reading = f" ({e['reading']})" if e["reading"] and e["reading"] != e["headword"] else ""
        pos = f" [{e['pos']}]" if e["pos"] else ""
        lines.append(f"{i}. {e['headword']}{reading}{pos} — {e['gloss']}")
    return PROMPT_HEAD + "\n".join(lines) + "\n"


# --------------------------------------------------------------- gemini call
def call_model(prompt: str, model: str, key: str) -> list[dict]:
    """One request to one model. Raises ModelGone / RateLimited / Overloaded /
    FatalError; returns [] when the model answered with nothing usable."""
    url = f"{API}/models/{model}:generateContent"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:600]
        low = detail.lower()
        if e.code in (400, 401, 403) and ("api key" in low or "api_key" in low):
            raise FatalError(f"Gemini rejected the API key (HTTP {e.code}): {detail}")
        if e.code == 429:
            m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', detail)
            raise RateLimited(detail, float(m.group(1)) if m else None, "PerDay" in detail)
        if e.code in (400, 403, 404):
            raise ModelGone(f"HTTP {e.code}: {detail[:200]}")
        raise Overloaded(f"HTTP {e.code}: {detail[:200]}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise Overloaded(str(e))
    try:
        cands = data.get("candidates") or []
        parts = (cands[0].get("content") or {}).get("parts") if cands else None
        if not parts:
            return []  # blocked / empty
        out = json.loads(parts[0].get("text", "[]"))
        return out if isinstance(out, list) else []
    except (ValueError, AttributeError, IndexError):
        return []


class ModelPool:
    """Tracks which models are usable: dead for the run, or resting until a time."""

    def __init__(self, models, now=time.monotonic):
        self.models = list(models)
        self.now = now
        self.dead: dict[str, str] = {}
        self.until: dict[str, float] = {}
        self.why: dict[str, str] = {}

    def usable(self):
        t = self.now()
        return [m for m in self.models if m not in self.dead and self.until.get(m, 0) <= t]

    def mark_dead(self, model, why):
        print(f"  model {model}: unavailable for this run ({why[:120]})")
        self.dead[model] = why

    def cooldown(self, model, seconds, why):
        print(f"  model {model}: resting {int(seconds)}s ({why[:120]})")
        self.until[model] = self.now() + seconds
        self.why[model] = why

    def next_wake(self):
        """Seconds until the soonest resting model is usable; None if none will be."""
        t = self.now()
        waits = [self.until[m] - t for m in self.models if m not in self.dead and m in self.until]
        waits = [w for w in waits if w > 0]
        return min(waits) if waits else None

    def summary(self):
        parts = []
        for m in self.models:
            if m in self.dead:
                parts.append(f"{m}: gone ({self.dead[m][:80]})")
            else:
                parts.append(f"{m}: resting ({self.why.get(m, 'n/a')[:80]})")
        return "; ".join(parts)


def call_with_pool(prompt: str, pool: ModelPool, call=None, sleep=time.sleep):
    """Sends [prompt] to the first usable model, switching automatically.

    - model missing / not allowed  -> skipped for the whole run
    - daily quota hit              -> rests 6h (i.e. the rest of the run)
    - per-minute limit             -> waits the server's retry delay once, else rests 2 min
    - overloaded / network error   -> retried twice (5s, 15s), then rests 5 min
    - empty answer                 -> next model gets a try
    Returns (items, model_used). Raises QuotaExhausted when nothing is usable.
    """
    call = call or (lambda pr, m: call_model(pr, m, os.environ["GEMINI_API_KEY"]))
    tried: set[str] = set()
    last = ""
    while True:
        cands = [m for m in pool.usable() if m not in tried]
        if not cands:
            if tried:
                return [], last  # every usable model answered with nothing
            wake = pool.next_wake()
            if wake is not None and wake <= 90:
                sleep(wake + 1)
                continue
            raise QuotaExhausted(pool.summary())
        model = cands[0]
        for attempt in range(3):
            try:
                items = call(prompt, model)
                if items:
                    return items, model
                tried.add(model)
                last = model
                break
            except ModelGone as e:
                pool.mark_dead(model, str(e))
                break
            except RateLimited as e:
                if e.per_day:
                    pool.cooldown(model, 6 * 3600, "daily quota")
                    break
                if e.retry_after is not None and e.retry_after <= 60 and attempt == 0:
                    sleep(e.retry_after + 1)
                    continue
                pool.cooldown(model, 120, "rate limited")
                break
            except Overloaded as e:
                if attempt < 2:
                    sleep(5 * 3 ** attempt)
                    continue
                pool.cooldown(model, 300, f"overloaded: {e}")
                break


def parse_models(*values: str) -> list[str]:
    """Comma/space separated model names -> clean ordered list (no 'models/' prefix)."""
    out: list[str] = []
    for v in values:
        for name in re.split(r"[,\s]+", v or ""):
            name = name.strip().removeprefix("models/")
            if name and name not in out:
                out.append(name)
    return out


def available_models(key: str):
    """Names of models the key can call generateContent on, or None if the lookup fails."""
    try:
        req = urllib.request.Request(f"{API}/models?pageSize=200", headers={"x-goog-api-key": key})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        return [m["name"].split("/", 1)[-1] for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])]
    except Exception as e:  # noqa: BLE001 - diagnostic only
        print(f"(could not list models: {e})")
        return None


def build_chain(configured: list[str], available) -> list[str]:
    """Configured models first (in order), then up to 4 other plain 'flash' models the key
    offers as last resorts. A model listed as available can still 404 at call time; the
    pool handles that, so this only drops names the key clearly doesn't have."""
    if available is None:
        return configured
    chain = []
    for m in configured:
        if m in available:
            chain.append(m)
        else:
            print(f"skipping {m}: not offered to this key")
    extras = [m for m in available
              if "flash" in m and not any(w in m for w in NOT_TEXT_MODEL) and m not in chain]
    # lite first, then stable before preview, then by name
    extras.sort(key=lambda m: ("lite" not in m, "preview" in m, m))
    return chain + extras[:4]


# ----------------------------------------------------------------- store/state
def load_store(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"attempts": {}}


def pick_candidates(jmdict_db: Path, covered: set[str], done: set[str],
                    attempts: dict, limit: int) -> list[dict]:
    con = sqlite3.connect(jmdict_db)
    seen: set[str] = set()
    pool: list[dict] = []
    for rowid, headword, reading, pos, meaning in con.execute(
        "SELECT rowid, headword, reading, part_of_speech, meaning FROM entries WHERE is_common = 1 ORDER BY rowid"
    ):
        if headword in seen:
            continue
        seen.add(headword)
        if headword in covered or headword in done or attempts.get(headword, 0) >= MAX_ATTEMPTS:
            continue
        gloss = first_glosses(meaning)
        if not gloss:
            continue
        pool.append({"rowid": rowid, "headword": headword, "reading": reading or "",
                     "pos": pos or "", "gloss": gloss})
    con.close()
    # Cheap proxy for "basic words first" until a real frequency list is wired in.
    pool.sort(key=lambda e: (len(e["headword"]), e["rowid"]))
    return pool[:limit]


# ------------------------------------------------------------------------ run
def run(args, generate: Callable[[str], tuple[list[dict], str]]) -> dict:
    store = load_store(args.store)
    state = load_state(args.state)
    attempts: dict = state.setdefault("attempts", {})
    done = {r["word"] for r in store}

    tcon = sqlite3.connect(args.tatoeba)
    covered = {r[0] for r in tcon.execute("SELECT DISTINCT word FROM sentence_words")}
    tcon.close()

    cands = pick_candidates(args.jmdict, covered, done, attempts, args.max_requests * args.batch_size)
    print(f"{len(cands)} candidate words this run (covered by Tatoeba: {len(covered)}, "
          f"already generated: {len(done)})")

    accepted = rejected = 0
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stopped = None
    with open(args.store, "a", encoding="utf-8") as out:
        for start in range(0, len(cands), args.batch_size):
            batch = cands[start:start + args.batch_size]
            try:
                items, used_model = generate(build_prompt(batch))
            except QuotaExhausted as e:
                stopped = f"quota exhausted: {e}"
                break
            by_index = {it.get("index"): it for it in items if isinstance(it, dict)}
            for i, entry in enumerate(batch, 1):
                item = by_index.get(i)
                reason = validate(entry, item) if item else "no item returned"
                if reason is None:
                    out.write(json.dumps({
                        "word": entry["headword"], "reading": entry["reading"],
                        "sentence": item["sentence"].strip(),
                        "translation": item["translation"].strip(),
                        "model": used_model, "added": today,
                    }, ensure_ascii=False) + "\n")
                    attempts.pop(entry["headword"], None)
                    accepted += 1
                else:
                    attempts[entry["headword"]] = attempts.get(entry["headword"], 0) + 1
                    rejected += 1
            out.flush()
            state_text = json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True)
            args.state.write_text(state_text + "\n", encoding="utf-8")
            if args.sleep:
                time.sleep(args.sleep)

    args.state.write_text(json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                          encoding="utf-8")
    print(f"accepted {accepted}, rejected {rejected}" + (f"; stopped early: {stopped}" if stopped else ""))
    return {"accepted": accepted, "rejected": rejected, "stopped": stopped}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jmdict", type=Path, required=True)
    ap.add_argument("--tatoeba", type=Path, required=True)
    ap.add_argument("--store", type=Path, default=Path("examples/ja/gemini.jsonl"))
    ap.add_argument("--state", type=Path, default=Path("examples/ja/state.json"))
    ap.add_argument("--max-requests", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--sleep", type=float, default=7.0, help="seconds between requests")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print("GEMINI_API_KEY not set -- skipping Gemini step.")
        return 0

    configured = parse_models(os.environ.get("GEMINI_MODELS", ""),
                              os.environ.get("GEMINI_MODEL", ""),
                              ",".join(DEFAULT_MODELS))
    chain = build_chain(configured, available_models(key))
    if not chain:
        print("::warning::No usable Gemini model found for this key -- skipping Gemini step.")
        return 0
    print("Model order:", ", ".join(chain))
    pool = ModelPool(chain)

    args.store.parent.mkdir(parents=True, exist_ok=True)
    args.store.touch(exist_ok=True)
    try:
        res = run(args, lambda prompt: call_with_pool(prompt, pool, lambda pr, m: call_model(pr, m, key)))
    except FatalError as e:
        print(f"::warning::Gemini step skipped: {e}")
        return 0
    if res["stopped"] and res["accepted"] == 0:
        print(f"::warning::Gemini unavailable this run: {res['stopped'][:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
