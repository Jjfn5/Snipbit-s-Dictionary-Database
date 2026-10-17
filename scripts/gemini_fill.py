#!/usr/bin/env python3
"""Fills gaps in a language's example corpus with Gemini-written sentences.

Fallback for words Tatoeba has no example for, for EVERY language (see
langs.py). Candidates = dictionary headwords with no Tatoeba example and no
Gemini example yet, common words first, then short ones. Words are sent in
batches (many per request, to stay inside free-tier request quotas). Every
sentence is validated (right script, target word really used, no brackets)
before it is stored; results are appended to examples/<lang>/gemini.jsonl
(append-only: the line number becomes the sentence id, so never reorder or
delete lines).

Needs env GEMINI_API_KEY. GEMINI_MODELS / GEMINI_MODEL pick the models (see
README). Without a key this exits 0 and does nothing, so the Tatoeba-only
pipeline keeps working.
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

import heapq
sys.path.insert(0, str(Path(__file__).parent))
from langs import (LANGS, Lang, has_japanese, index_key, script_ok,  # noqa: E402,F401
                   target_ok, word_stem, worth_generating)

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
BAD_CHARS = set("[]{}()（）「」『』<>")

PROMPT_HEAD = """You write example sentences for a {name} dictionary app used by learners.
For each numbered word below, write ONE natural {name} sentence that uses the word in the meaning given.

Rules:
- {length_rule}, everyday register, simple grammar and vocabulary apart from the target word.
- Use the target word itself (conjugated or inflected if that is natural). Put the exact form you used in "surface".
- {markup_rule}
- "translation" is a natural English translation of the sentence.
- Return JSON only: an array with one object per word: {{"index": n, "sentence": "...", "surface": "...", "translation": "..."}}.

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
def first_glosses(meaning_json: str, limit: int = 3) -> str:
    try:
        groups = json.loads(meaning_json)
        return "; ".join(groups[0]["g"][:limit])
    except (ValueError, KeyError, IndexError, TypeError):
        return ""


def validate(lang: Lang, entry: dict, item: dict) -> Optional[str]:
    """Returns None if the item is acceptable, else a short reject reason."""
    sentence = item.get("sentence")
    translation = item.get("translation")
    surface = item.get("surface")
    if not all(isinstance(x, str) for x in (sentence, translation, surface)):
        return "missing fields"
    sentence, translation, surface = sentence.strip(), translation.strip(), surface.strip()
    if not (lang.min_chars <= len(sentence) <= lang.max_chars):
        return "bad length"
    if not (3 <= len(translation) <= 250):
        return "bad translation length"
    bad_script = script_ok(lang, sentence)
    if bad_script:
        return bad_script
    if any(c in BAD_CHARS for c in sentence):
        return "brackets/quotes in sentence"
    if not surface or surface.lower() not in sentence.lower():
        return "surface not in sentence"
    if not target_ok(lang, entry["headword"], entry["pos"], sentence, surface):
        return "target word not in sentence"
    return None


def build_prompt(lang: Lang, batch: list[dict]) -> str:
    head = PROMPT_HEAD.format(name=lang.name, length_rule=lang.length_rule,
                              markup_rule=lang.markup_rule)
    lines = []
    for i, e in enumerate(batch, 1):
        reading = f" ({e['reading']})" if e["reading"] and e["reading"] != e["headword"] else ""
        pos = f" [{e['pos']}]" if e["pos"] else ""
        lines.append(f"{i}. {e['headword']}{reading}{pos} — {e['gloss']}")
    return head + "\n".join(lines) + "\n"


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


def pick_candidates(lang: Lang, dictionary_db: Path, covered: set[str], done: set[str],
                    attempts: dict, limit: int) -> list[dict]:
    """Common words first, then shorter headwords (a cheap proxy for "basic words first"
    until a real frequency list is wired in). Streams the table and keeps only the best
    `limit`, so a million-row Kaikki db stays cheap."""
    con = sqlite3.connect(dictionary_db)
    seen: set[str] = set()

    def candidates():
        for rowid, headword, reading, pos, meaning, common in con.execute(
            "SELECT rowid, headword, reading, part_of_speech, meaning, is_common "
            "FROM entries ORDER BY rowid"
        ):
            if headword in seen:
                continue
            seen.add(headword)
            if (headword in done or index_key(lang, headword) in covered
                    or attempts.get(headword, 0) >= MAX_ATTEMPTS):
                continue
            gloss = first_glosses(meaning)
            if not gloss or not worth_generating(lang, headword, pos or "", gloss):
                continue
            yield (0 if common else 1, len(headword), rowid), {
                "headword": headword, "reading": reading or "", "pos": pos or "", "gloss": gloss}

    best = heapq.nsmallest(limit, candidates(), key=lambda kv: kv[0])
    con.close()
    return [e for _, e in best]


# ------------------------------------------------------------------------ run
def run(args, generate: Callable[[str], tuple[list[dict], str]]) -> dict:
    store = load_store(args.store)
    state = load_state(args.state)
    attempts: dict = state.setdefault("attempts", {})
    done = {r["word"] for r in store}

    lang = LANGS[args.lang]
    tcon = sqlite3.connect(args.tatoeba)
    covered = {r[0] for r in tcon.execute("SELECT DISTINCT word FROM sentence_words")}
    tcon.close()

    cands = pick_candidates(lang, args.dictionary, covered, done, attempts,
                            args.max_requests * args.batch_size)
    print(f"{len(cands)} candidate words this run (covered by Tatoeba: {len(covered)}, "
          f"already generated: {len(done)})")

    accepted = rejected = 0
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stopped = None
    with open(args.store, "a", encoding="utf-8") as out:
        for start in range(0, len(cands), args.batch_size):
            batch = cands[start:start + args.batch_size]
            try:
                items, used_model = generate(build_prompt(lang, batch))
            except QuotaExhausted as e:
                stopped = f"quota exhausted: {e}"
                break
            by_index = {it.get("index"): it for it in items if isinstance(it, dict)}
            for i, entry in enumerate(batch, 1):
                item = by_index.get(i)
                reason = validate(lang, entry, item) if item else "no item returned"
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
    ap.add_argument("--lang", choices=sorted(LANGS), required=True)
    ap.add_argument("--dictionary", type=Path, required=True,
                    help="that language's published <source>.db (entries table)")
    ap.add_argument("--tatoeba", type=Path, required=True)
    ap.add_argument("--store", type=Path)
    ap.add_argument("--state", type=Path)
    ap.add_argument("--max-requests", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--sleep", type=float, default=7.0, help="seconds between requests")
    args = ap.parse_args()
    args.store = args.store or Path(f"examples/{args.lang}/gemini.jsonl")
    args.state = args.state or Path(f"examples/{args.lang}/state.json")

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
