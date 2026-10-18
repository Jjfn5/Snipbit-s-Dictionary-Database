#!/usr/bin/env python3
"""Builds spa.db.gz / fra.db.gz from kaikki.org's Wiktionary JSONL export.

Port of KaikkiParser.kt. One JSON object per line = one word+POS; only the
LAST element of each sense's `glosses` breadcrumb is the real definition.

Shared by every Kaikki-based dictionary; the per-language entry points are
dictionaries/es/kaikki.py and dictionaries/fr/kaikki.py (they pass --lang).

  python dictionaries/es/kaikki.py
  python dictionaries/fr/kaikki.py --input some.jsonl --min-rows 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import dictlib  # noqa: E402
from translit import build_latin_key  # noqa: E402

BUILDER_VERSION = 1
LANGS = {
    "spa": "https://kaikki.org/dictionary/Spanish/kaikki.org-dictionary-Spanish.jsonl",
    "fra": "https://kaikki.org/dictionary/French/kaikki.org-dictionary-French.jsonl",
}


def line_row(source: str, line: str):
    line = line.strip()
    if not line:
        return None
    try:
        d = json.loads(line)
    except ValueError:
        return None  # malformed/partial line: skip, like the app parser did
    if not isinstance(d, dict):
        return None
    headword = (d.get("word") or "").strip()
    if not headword:
        return None
    pos = (d.get("pos") or "").strip() or None
    groups = []
    for sense in d.get("senses") or []:
        path = sense.get("glosses") if isinstance(sense, dict) else None
        if not path:
            continue
        gloss = (path[-1] or "").strip()
        if gloss:
            groups.append({"g": [gloss]})
    if not groups:
        return None
    return (source, headword, None, None, pos, dictlib.meaning_json(groups), 0,
            build_latin_key(source, headword))


def make_rows(source: str):
    def build_rows(files):
        with open(files[0], encoding="utf-8", errors="replace") as f:
            for line in f:
                row = line_row(source, line)
                if row:
                    yield row
    return build_rows


def main(argv=None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--lang", choices=sorted(LANGS), required=True)
    known, rest = pre.parse_known_args(argv)
    return dictlib.run(
        source=known.lang,
        builder_version=BUILDER_VERSION,
        urls=[LANGS[known.lang]],
        build_rows=make_rows(known.lang),
        default_min_rows=100_000,
        argv=rest,
    )


if __name__ == "__main__":
    sys.exit(main())
