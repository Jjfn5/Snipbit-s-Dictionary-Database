#!/usr/bin/env python3
"""Builds a ready-to-copy `jmdict.db` from jmdict-simplified.

Port of JmdictParser.kt's addWordRow. Output schema matches the app's
`entries` table (minus id/indexes: the app inserts into its own table,
which owns ids and indexes).

Usage:
  python dictionaries/ja/jmdict.py                     # fetch latest upstream
  python dictionaries/ja/jmdict.py --input some.json.zip # local file (testing)
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sqlite3
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from translit import build_latin_key  # noqa: E402
import freqlib  # noqa: E402

RELEASE_API = "https://api.github.com/repos/scriptin/jmdict-simplified/releases/latest"
ASSET_RE = re.compile(r"^jmdict-eng-\d.*\.json\.zip$")
SCHEMA_VERSION = 2   # bump when the entries columns change (app must check); 2 = + freq_rank
BUILDER_VERSION = 1  # bump when parsing logic changes, to force a rebuild
DEFAULT_BASE_URL = (
    "https://github.com/Jjfn5/Snipbit-s-Dictionary-Database/releases/download/jmdict"
)
UA = "snipbit-dictionary-builder"

SCHEMA = """
CREATE TABLE entries (
    source TEXT NOT NULL,
    headword TEXT NOT NULL,
    alt_headword TEXT,
    reading TEXT,
    part_of_speech TEXT,
    meaning TEXT NOT NULL,
    is_common INTEGER NOT NULL DEFAULT 0,
    reading_latin TEXT,
    freq_rank INTEGER
)
"""


def _request(url: str) -> urllib.request.Request:
    headers = {"User-Agent": UA, "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token and "api.github.com" in url:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def resolve_latest() -> tuple[str, str]:
    with urllib.request.urlopen(_request(RELEASE_API), timeout=60) as r:
        data = json.load(r)
    for asset in data.get("assets", []):
        if ASSET_RE.match(asset.get("name", "")):
            return data["tag_name"], asset["browser_download_url"]
    raise RuntimeError("No asset matching jmdict-eng-*.json.zip in the latest release")


def download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(_request(url), timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def load_words(path: Path) -> list[dict]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = next(
                (n for n in zf.namelist() if n.endswith(".json") and not n.endswith("/")),
                None,
            )
            if name is None:
                raise RuntimeError(f"No .json entry in {path}")
            with zf.open(name) as f:
                return json.load(f)["words"]
    with open(path, encoding="utf-8") as f:
        return json.load(f)["words"]


def _str_list(v) -> list[str]:
    return [s for s in (v or []) if isinstance(s, str) and s]


def word_to_row(word: dict):
    """Returns the entries row tuple, or None if the entry is skipped."""
    kanji = word.get("kanji") or []
    kana = word.get("kana") or []

    first_kanji = kanji[0].get("text") if kanji else None
    first_kana = kana[0].get("text") if kana else None
    headword = first_kanji or first_kana
    if not headword:
        return None
    reading = first_kana

    is_common = any(k.get("common") for k in kanji) or any(k.get("common") for k in kana)

    groups: list[dict] = []
    all_pos: list[str] = []
    for sense in word.get("sense") or []:
        glosses = [
            g["text"]
            for g in (sense.get("gloss") or [])
            if g.get("lang") in (None, "eng") and g.get("text")
        ]
        if not glosses:
            continue
        pos = _str_list(sense.get("partOfSpeech"))
        for p in pos:
            if p not in all_pos:
                all_pos.append(p)
        field = _str_list(sense.get("field"))
        misc = _str_list(sense.get("misc"))
        info = _str_list(sense.get("info"))
        group: dict = {"g": glosses}  # key order g,p,f,m,i like SenseGroupJson
        if pos:
            group["p"] = ", ".join(pos)
        if field:
            group["f"] = ", ".join(field)
        if misc:
            group["m"] = ", ".join(misc)
        if info:
            group["i"] = "; ".join(info)
        groups.append(group)

    if not groups:
        return None

    alt = kanji[1].get("text") if len(kanji) > 1 else None
    return (
        "jmdict",
        headword,
        alt,
        reading,
        ", ".join(all_pos) if all_pos else None,
        json.dumps(groups, ensure_ascii=False, separators=(",", ":")),
        1 if is_common else 0,
        build_latin_key("jmdict", headword, reading),
    )


def build_db(words: list[dict], db_path: Path, ranker=None) -> int:
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.execute(SCHEMA)
    rows, n = [], 0
    insert = "INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?)"
    for w in words:
        row = word_to_row(w)
        if row is None:
            continue
        rows.append(ranker(row) if ranker else tuple(row) + (None,))
        n += 1
        if len(rows) >= 5000:
            con.executemany(insert, rows)
            rows = []
    if rows:
        con.executemany(insert, rows)
    con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    con.commit()
    con.execute("VACUUM")
    con.close()
    return n


def set_output(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"{key}={value}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, help="local .json/.json.zip instead of fetching")
    ap.add_argument("--out-dir", type=Path, default=Path("out"))
    ap.add_argument("--prev-manifest", type=Path, help="skip build if upstream tag unchanged")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--min-rows", type=int, default=150_000,
                    help="sanity floor; refuse to publish a suspiciously small db")
    ap.add_argument("--freq-file", type=Path,
                    help="frequency list (one word per line, most frequent first) "
                         "instead of the wordfreq package")
    ap.add_argument("--no-freq", action="store_true",
                    help="leave freq_rank empty (no frequency data at all)")
    ap.add_argument("--min-ranked", type=int, default=5_000,
                    help="sanity floor: refuse to publish if fewer entries got a freq_rank")
    args = ap.parse_args(argv)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    freq_label = "none" if args.no_freq else freqlib.source_label(args.freq_file)

    if args.input:
        src, tag = args.input, f"local-{args.input.name}"
    else:
        tag, url = resolve_latest()
        if args.prev_manifest and args.prev_manifest.exists():
            try:
                prev = json.loads(args.prev_manifest.read_text())
                if (prev.get("upstreamTag") == tag
                        and prev.get("builderVersion") == BUILDER_VERSION
                        and prev.get("schemaVersion") == SCHEMA_VERSION
                        and prev.get("frequencySource") == freq_label):
                    print(f"Upstream {tag} unchanged -- nothing to do")
                    set_output("changed", "false")
                    return 0
            except (ValueError, OSError):
                pass  # unreadable previous manifest -> just rebuild
        src = out / "jmdict-upstream.json.zip"
        print(f"Downloading {url}")
        download(url, src)

    ranker = None if args.no_freq else freqlib.make_ranker("jmdict", args.freq_file)
    print("Parsing...")
    words = load_words(src)
    db_path = out / "jmdict.db"
    n = build_db(words, db_path, ranker)
    ranked = ranker.ranked if ranker else 0
    print(f"{len(words)} upstream words -> {n} rows, {ranked} with a frequency rank ({freq_label})")
    if n < args.min_rows:
        print(f"ERROR: only {n} rows (< {args.min_rows}); refusing to publish", file=sys.stderr)
        return 1
    if not args.no_freq and ranked < args.min_ranked:
        print(f"ERROR: only {ranked} ranked entries (< {args.min_ranked}); refusing to publish",
              file=sys.stderr)
        return 1

    gz_path = out / "jmdict.db.gz"
    with open(db_path, "rb") as fi, open(gz_path, "wb") as raw:
        # mtime=0 -> deterministic bytes for identical input
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as gz:
            while chunk := fi.read(1 << 20):
                gz.write(chunk)

    sha = hashlib.sha256(gz_path.read_bytes()).hexdigest()
    manifest = {
        "source": "jmdict",
        "schemaVersion": SCHEMA_VERSION,
        "builderVersion": BUILDER_VERSION,
        "upstreamTag": tag,
        "builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rowCount": n,
        "frequencySource": freq_label,
        "rankedRows": ranked,
        "dbUrl": f"{args.base_url}/jmdict.db.gz",
        "dbSha256": sha,
        "dbGzBytes": gz_path.stat().st_size,
        "dbBytes": db_path.stat().st_size,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    set_output("changed", "true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
