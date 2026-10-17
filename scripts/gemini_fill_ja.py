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
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

MAX_ATTEMPTS = 3
DEFAULT_MODEL = "gemini-2.5-flash-lite"
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
    pass


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
def call_gemini(prompt: str, model: str, key: str) -> list[dict]:
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
    last_err: Optional[str] = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            cands = data.get("candidates") or []
            parts = (cands[0].get("content") or {}).get("parts") if cands else None
            if not parts:
                return []  # blocked / empty -> counts as a failed attempt for those words
            return json.loads(parts[0].get("text", "[]"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if e.code == 429:
                if attempt == 0:
                    print("429 from Gemini; waiting 65s once (may just be the per-minute limit)")
                    time.sleep(65)
                    continue
                raise QuotaExhausted(detail)
            if e.code >= 500:
                last_err = f"HTTP {e.code}: {detail}"
                time.sleep(5 * (attempt + 1))
                continue
            if e.code == 404:
                list_models(key)
                raise RuntimeError(
                    f"Model '{model}' not found. Pick one from the list above and set the "
                    f"repo variable GEMINI_MODEL (Settings > Secrets and variables > Actions > Variables)."
                )
            raise RuntimeError(f"Gemini HTTP {e.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = str(e)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Gemini request failed: {last_err}")


def list_models(key: str) -> None:
    try:
        req = urllib.request.Request(f"{API}/models?pageSize=200", headers={"x-goog-api-key": key})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        print("Models available to this key that support generateContent:")
        for m in data.get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print("  ", m["name"].split("/", 1)[-1])
    except Exception as e:  # noqa: BLE001 - diagnostic only
        print(f"(could not list models: {e})")


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
def run(args, generate: Callable[[str], list[dict]]) -> dict:
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
                items = generate(build_prompt(batch))
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
                        "model": args.model, "added": today,
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
    args.model = os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
    print(f"Using model {args.model}")
    args.store.parent.mkdir(parents=True, exist_ok=True)
    args.store.touch(exist_ok=True)
    run(args, lambda prompt: call_gemini(prompt, args.model, key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
