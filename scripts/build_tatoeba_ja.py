#!/usr/bin/env python3
"""Builds the Tatoeba half of the Japanese example corpus.

Port of ExampleCorpusParser.parseSentencesWithTranslations (3-file join)
+ parseJapaneseIndicesFile (Tanaka "B-line" word index). Output tables
match the app's `sentences` / `sentence_words` (no indexes -- the app
owns those).

  python scripts/build_tatoeba_ja.py --out out/tatoeba_ja.db
  python scripts/build_tatoeba_ja.py --input-dir some/dir --out x.db   # offline/test
"""
from __future__ import annotations

import argparse
import bz2
import io
import re
import sqlite3
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterator, Optional

SOURCE_ID = "jmdict"
SCHEMA_VERSION = 1
BASE = "https://downloads.tatoeba.org/exports"
FILES = {
    "sentences": f"{BASE}/per_language/jpn/jpn_sentences.tsv.bz2",
    "links": f"{BASE}/per_language/jpn/jpn-eng_links.tsv.bz2",
    "english": f"{BASE}/per_language/eng/eng_sentences.tsv.bz2",
    "indices": f"{BASE}/jpn_indices.tar.bz2",
}
UA = "snipbit-dictionary-builder"
BATCH = 5000

_HEADWORD = re.compile(r"^([^\[{(~]+)")


def fetch(url: str, dest: Path) -> None:
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def bz2_lines(path: Path) -> Iterator[str]:
    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line.rstrip("\r\n")


def tar_bz2_lines(path: Path) -> Iterator[str]:
    """Lines of the first regular file inside a .tar.bz2 (jpn_indices)."""
    with tarfile.open(path, "r:bz2") as tar:
        for member in tar:
            if not member.isfile():
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            for line in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
                yield line.rstrip("\r\n")
            return
    raise RuntimeError(f"No file entry found in {path}")


def to_int(s: str) -> Optional[int]:
    try:
        return int(s.strip())
    except ValueError:
        return None


def has_cjk(s: str) -> bool:
    return any(
        0x4E00 <= ord(c) <= 0x9FFF or 0x3040 <= ord(c) <= 0x309F or 0x30A0 <= ord(c) <= 0x30FF
        for c in s
    )


def headword_from_token(token: str) -> Optional[str]:
    """'食べる[たべる]{食べた}' -> '食べる'; None if no CJK-script char."""
    m = _HEADWORD.match(token)
    word = m.group(1).strip() if m else ""
    if not word:
        return None
    return word if has_cjk(word) else None


def build(files: dict[str, Path], out: Path) -> tuple[int, int]:
    # Pass 1: links (jpnId \t engId) -> first translation per sentence.
    eng_for: dict[int, int] = {}
    needed: set[int] = set()
    for line in bz2_lines(files["links"]):
        cols = line.split("\t")
        if len(cols) >= 2:
            a, b = to_int(cols[0]), to_int(cols[1])
            if a is not None and b is not None:
                eng_for.setdefault(a, b)
                needed.add(b)
    print(f"links: {len(eng_for)} jpn sentences with a translation link")

    # Pass 2: English text, only for ids we need.
    eng_text: dict[int, str] = {}
    if needed:
        for line in bz2_lines(files["english"]):
            t1 = line.find("\t")
            if t1 <= 0:
                continue
            sid = to_int(line[:t1])
            if sid is None or sid not in needed:
                continue
            t2 = line.find("\t", t1 + 1)
            text = (line[t2 + 1:] if t2 >= 0 else line[t1 + 1:]).strip()
            if text:
                eng_text[sid] = text
    print(f"english: resolved {len(eng_text)} translations")

    if out.exists():
        out.unlink()
    con = sqlite3.connect(out)
    con.execute(
        "CREATE TABLE sentences (id INTEGER PRIMARY KEY, source TEXT NOT NULL, "
        "text TEXT NOT NULL, translation TEXT)"
    )
    con.execute(
        "CREATE TABLE sentence_words (word TEXT NOT NULL, sentence_id INTEGER NOT NULL, "
        "source TEXT NOT NULL)"
    )

    # Pass 3: Japanese sentences.
    rows = []
    n_sent = 0
    for line in bz2_lines(files["sentences"]):
        cols = line.split("\t")
        if len(cols) < 3:
            continue
        sid = to_int(cols[0])
        text = cols[2].strip()
        if sid is None or not text:
            continue
        eid = eng_for.get(sid)
        rows.append((sid, SOURCE_ID, text, eng_text.get(eid) if eid is not None else None))
        n_sent += 1
        if len(rows) >= BATCH:
            con.executemany("INSERT OR REPLACE INTO sentences VALUES (?,?,?,?)", rows)
            rows = []
    if rows:
        con.executemany("INSERT OR REPLACE INTO sentences VALUES (?,?,?,?)", rows)

    # Word index from the Tanaka-tagged indices file.
    wrows = []
    n_words = 0
    for line in tar_bz2_lines(files["indices"]):
        cols = line.split("\t")
        if len(cols) < 3:
            continue
        sid = to_int(cols[0])
        text = cols[2].strip()
        if sid is None or not text:
            continue
        seen: set[str] = set()
        for token in text.split():
            w = headword_from_token(token)
            if w and w not in seen:
                seen.add(w)
                wrows.append((w, sid, SOURCE_ID))
                n_words += 1
        if len(wrows) >= BATCH:
            con.executemany("INSERT INTO sentence_words VALUES (?,?,?)", wrows)
            wrows = []
    if wrows:
        con.executemany("INSERT INTO sentence_words VALUES (?,?,?)", wrows)

    con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    con.commit()
    con.close()
    print(f"wrote {n_sent} sentences, {n_words} word links -> {out}")
    return n_sent, n_words


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/tatoeba_ja.db"))
    ap.add_argument("--work-dir", type=Path, default=Path("out/tatoeba_work"))
    ap.add_argument("--input-dir", type=Path,
                    help="use already-downloaded files (named like the URLs' basenames)")
    ap.add_argument("--min-sentences", type=int, default=50_000,
                    help="sanity floor; refuse to produce a suspiciously small db")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    for key, url in FILES.items():
        name = url.rsplit("/", 1)[1]
        if args.input_dir:
            files[key] = args.input_dir / name
        else:
            args.work_dir.mkdir(parents=True, exist_ok=True)
            files[key] = args.work_dir / name
            fetch(url, files[key])

    n_sent, _ = build(files, args.out)
    if n_sent < args.min_sentences:
        print(f"ERROR: only {n_sent} sentences (< {args.min_sentences})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
