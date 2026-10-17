#!/usr/bin/env python3
"""Combines the Tatoeba db and the Gemini jsonl into the published
`jmdict_examples.db.gz` + `manifest.json`.

Gemini sentence ids = GEMINI_ID_BASE + line number (1-based). Tatoeba ids
are below ~15 million, so the ranges never collide, and the base stays
under 2^31 in case some code reads ids as 32-bit ints.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

SOURCE_ID = "jmdict"
SCHEMA_VERSION = 1
BUILDER_VERSION = 1
GEMINI_ID_BASE = 1_000_000_000
DEFAULT_BASE_URL = (
    "https://github.com/Jjfn5/Snipbit-s-Dictionary-Database/releases/download/examples-ja"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def set_output(key: str, value: str) -> None:
    p = os.environ.get("GITHUB_OUTPUT")
    if p:
        with open(p, "a") as f:
            f.write(f"{key}={value}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tatoeba", type=Path, required=True)
    ap.add_argument("--gemini", type=Path, default=Path("examples/ja/gemini.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("out"))
    ap.add_argument("--prev-manifest", type=Path)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    gemini_rows = []
    if args.gemini.exists():
        gemini_rows = [json.loads(l) for l in args.gemini.read_text(encoding="utf-8").splitlines() if l.strip()]
    gemini_sha = sha256_file(args.gemini) if args.gemini.exists() else "none"
    content_key = hashlib.sha256(
        f"{sha256_file(args.tatoeba)}|{gemini_sha}|{BUILDER_VERSION}|{SCHEMA_VERSION}".encode()
    ).hexdigest()

    if args.prev_manifest and args.prev_manifest.exists():
        try:
            if json.loads(args.prev_manifest.read_text()).get("contentKey") == content_key:
                print("Examples unchanged -- nothing to publish")
                set_output("changed", "false")
                return 0
        except (ValueError, OSError):
            pass

    db_path = out / "jmdict_examples.db"
    shutil.copyfile(args.tatoeba, db_path)
    con = sqlite3.connect(db_path)
    sent, words = [], []
    for n, r in enumerate(gemini_rows, 1):
        sid = GEMINI_ID_BASE + n
        sent.append((sid, SOURCE_ID, r["sentence"], r["translation"]))
        words.append((r["word"], sid, SOURCE_ID))
    con.executemany("INSERT OR REPLACE INTO sentences VALUES (?,?,?,?)", sent)
    con.executemany("INSERT INTO sentence_words VALUES (?,?,?)", words)
    con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    con.commit()
    n_sentences = con.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
    n_words = con.execute("SELECT COUNT(*) FROM sentence_words").fetchone()[0]
    con.execute("VACUUM")
    con.close()

    gz_path = out / "jmdict_examples.db.gz"
    with open(db_path, "rb") as fi, open(gz_path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as gz:
            while chunk := fi.read(1 << 20):
                gz.write(chunk)

    manifest = {
        "source": SOURCE_ID,
        "kind": "examples",
        "schemaVersion": SCHEMA_VERSION,
        "builderVersion": BUILDER_VERSION,
        "contentKey": content_key,
        "builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rowCount": n_sentences,
        "wordRowCount": n_words,
        "geminiCount": len(gemini_rows),
        "dbUrl": f"{args.base_url}/jmdict_examples.db.gz",
        "dbSha256": sha256_file(gz_path),
        "dbGzBytes": gz_path.stat().st_size,
        "dbBytes": db_path.stat().st_size,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    set_output("changed", "true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
