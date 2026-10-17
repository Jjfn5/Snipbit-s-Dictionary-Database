#!/usr/bin/env python3
"""Builds the Tatoeba example db for Chinese, Korean, Spanish and French.

Port of ExampleCorpusParser.parseSentencesWithTranslations: a 3-file join
(links -> English text -> sentences) plus a per-sentence word index.
Japanese has its own pipeline (build_tatoeba_ja.py + Gemini).

Word index: Korean/Spanish/French split on whitespace (apostrophe words are
indexed whole AND as pieces, so l'eau finds eau); Chinese is segmented with
jieba (pip install jieba). Every token is NFC-normalized, lowercased and
stripped of leading/trailing non-alphanumerics.

  python scripts/build_examples.py --lang spa --out-dir out
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

sys.path.insert(0, str(Path(__file__).parent))
import dictlib  # noqa: E402
from build_tatoeba_ja import bz2_lines, fetch, to_int  # noqa: E402

SCHEMA_VERSION = 1   # must equal PrebuiltImporter.EXAMPLES_SCHEMA_VERSION in the app
BUILDER_VERSION = 1
BASE = "https://downloads.tatoeba.org/exports"
ENG_URL = f"{BASE}/per_language/eng/eng_sentences.tsv.bz2"
BATCH = 5000

# lang -> (dictionary source id, Tatoeba code, release tag, sanity floor, tokenizer)
LANGS = {
    "zh": ("ccedict", "cmn", "examples-zh", 20_000, "jieba"),
    "ko": ("kor", "kor", "examples-ko", 3_000, "space"),
    "es": ("spa", "spa", "examples-es", 100_000, "space"),
    "fr": ("fra", "fra", "examples-fr", 100_000, "space"),
}

_EDGE = re.compile(r"^[\W_]+|[\W_]+$")
_WS = re.compile(r"\s+")
_ELISION = re.compile(r"['\u2019]")


def normalize_token(tok: str) -> Optional[str]:
    t = _EDGE.sub("", unicodedata.normalize("NFC", tok.strip()))
    return t.lower() or None


def tokenize_space(text: str) -> list[str]:
    out: list[str] = []
    for raw in _WS.split(text):
        whole = normalize_token(raw)
        if not whole:
            continue
        out.append(whole)
        if _ELISION.search(whole):
            for piece in _ELISION.split(whole):
                p = normalize_token(piece)
                if p:
                    out.append(p)
    return out


def make_jieba_tokenizer() -> Callable[[str], list[str]]:
    import jieba  # imported lazily: only the Chinese build needs it
    jieba.setLogLevel(60)

    def tokenize(text: str) -> list[str]:
        return [t for t in (normalize_token(w) for w in jieba.lcut(text)) if t]
    return tokenize


def build(files: dict[str, Path], out: Path, source: str,
          tokenize: Callable[[str], list[str]]) -> tuple[int, int]:
    eng_for: dict[int, int] = {}
    needed: set[int] = set()
    for line in bz2_lines(files["links"]):
        cols = line.split("\t")
        if len(cols) >= 2:
            a, b = to_int(cols[0]), to_int(cols[1])
            if a is not None and b is not None:
                eng_for.setdefault(a, b)   # first translation wins
                needed.add(b)

    eng_text: dict[int, str] = {}
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

    if out.exists():
        out.unlink()
    con = sqlite3.connect(out)
    con.execute("CREATE TABLE sentences (id INTEGER PRIMARY KEY, source TEXT NOT NULL, "
                "text TEXT NOT NULL, translation TEXT)")
    con.execute("CREATE TABLE sentence_words (word TEXT NOT NULL, "
                "sentence_id INTEGER NOT NULL, source TEXT NOT NULL)")

    srows, wrows = [], []
    n_sent = n_words = 0
    for line in bz2_lines(files["sentences"]):
        cols = line.split("\t")
        if len(cols) < 3:
            continue
        sid = to_int(cols[0])
        text = cols[2].strip()
        if sid is None or not text:
            continue
        eid = eng_for.get(sid)
        srows.append((sid, source, text, eng_text.get(eid) if eid is not None else None))
        n_sent += 1
        for w in dict.fromkeys(tokenize(text)):   # one row per distinct word
            wrows.append((w, sid, source))
            n_words += 1
        if len(srows) >= BATCH:
            con.executemany("INSERT OR REPLACE INTO sentences VALUES (?,?,?,?)", srows)
            srows = []
        if len(wrows) >= BATCH:
            con.executemany("INSERT INTO sentence_words VALUES (?,?,?)", wrows)
            wrows = []
    if srows:
        con.executemany("INSERT OR REPLACE INTO sentences VALUES (?,?,?,?)", srows)
    if wrows:
        con.executemany("INSERT INTO sentence_words VALUES (?,?,?)", wrows)
    con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    con.commit()
    con.execute("VACUUM")
    con.close()
    print(f"wrote {n_sent} sentences, {n_words} word links -> {out}")
    return n_sent, n_words


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=sorted(LANGS), required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("out"))
    ap.add_argument("--work-dir", type=Path, default=Path("out/tatoeba_work"))
    ap.add_argument("--input-dir", type=Path,
                    help="already-downloaded files named like the URLs' basenames (testing)")
    ap.add_argument("--prev-manifest", type=Path)
    ap.add_argument("--base-url")
    ap.add_argument("--min-sentences", type=int)
    args = ap.parse_args(argv)

    source, code, tag, floor, tok_kind = LANGS[args.lang]
    base_url = args.base_url or f"{dictlib.REPO_RELEASES}/{tag}"
    min_sent = floor if args.min_sentences is None else args.min_sentences
    urls = {
        "sentences": f"{BASE}/per_language/{code}/{code}_sentences.tsv.bz2",
        "links": f"{BASE}/per_language/{code}/{code}-eng_links.tsv.bz2",
        "english": ENG_URL,
    }
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    for key, url in urls.items():
        name = url.rsplit("/", 1)[1]
        if args.input_dir:
            files[key] = args.input_dir / name
        else:
            args.work_dir.mkdir(parents=True, exist_ok=True)
            files[key] = args.work_dir / name
            fetch(url, files[key])

    content_key = hashlib.sha256(
        ("|".join(dictlib.sha256_file(files[k]) for k in sorted(files))
         + f"|{BUILDER_VERSION}|{SCHEMA_VERSION}|{tok_kind}").encode()
    ).hexdigest()
    if args.prev_manifest and args.prev_manifest.exists():
        try:
            if json.loads(args.prev_manifest.read_text()).get("contentKey") == content_key:
                print("Examples unchanged -- nothing to publish")
                dictlib.set_output("changed", "false")
                return 0
        except (ValueError, OSError):
            pass

    tokenize = make_jieba_tokenizer() if tok_kind == "jieba" else tokenize_space
    db_name = f"{source}_examples"
    db_path = out / f"{db_name}.db"
    n_sent, n_words = build(files, db_path, source, tokenize)
    if n_sent < min_sent:
        print(f"ERROR: only {n_sent} sentences (< {min_sent}); refusing to publish",
              file=sys.stderr)
        return 1

    gz_path = out / f"{db_name}.db.gz"
    dictlib.gzip_file(db_path, gz_path)
    manifest = {
        "source": source,
        "kind": "examples",
        "schemaVersion": SCHEMA_VERSION,
        "builderVersion": BUILDER_VERSION,
        "contentKey": content_key,
        "builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rowCount": n_sent,
        "wordRowCount": n_words,
        "dbUrl": f"{base_url}/{db_name}.db.gz",
        "dbSha256": dictlib.sha256_file(gz_path),
        "dbGzBytes": gz_path.stat().st_size,
        "dbBytes": db_path.stat().st_size,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    dictlib.set_output("changed", "true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
