"""Shared plumbing for the per-source entry builders (ccedict, kor, spa, fra).

dictionaries/ja/jmdict.py predates this module and keeps its own copy of the same
logic; the output format (table, manifest keys, gzip) is identical, so the
app treats every source the same way.

A builder supplies: its source id, a list of upstream URLs, and a function
turning the downloaded files into `entries` row tuples. `run()` does the
rest: change detection, download, sanity floor, db, gzip, manifest.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

SCHEMA_VERSION = 1  # bump when the entries columns change (the app checks it)
UA = "snipbit-dictionary-builder"
REPO_RELEASES = "https://github.com/Jjfn5/Snipbit-s-Dictionary-Database/releases/download"

SCHEMA = """
CREATE TABLE entries (
    source TEXT NOT NULL,
    headword TEXT NOT NULL,
    alt_headword TEXT,
    reading TEXT,
    part_of_speech TEXT,
    meaning TEXT NOT NULL,
    is_common INTEGER NOT NULL DEFAULT 0,
    reading_latin TEXT
)
"""

Row = tuple  # (source, headword, alt, reading, pos, meaning_json, is_common, reading_latin)


def meaning_json(groups: list[dict]) -> str:
    """Same shape as SenseGroupJson.encode: [{"g":[...], "p":..., ...}]."""
    return json.dumps(groups, ensure_ascii=False, separators=(",", ":"))


# ------------------------------------------------------------------ network
def _request(url: str, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(url, method=method, headers={"User-Agent": UA})


def stamp(urls: list[str]) -> str:
    """Cheap change detector from HEAD responses; '' if anything is unavailable."""
    parts = []
    for u in urls:
        try:
            with urllib.request.urlopen(_request(u, "HEAD"), timeout=60) as r:
                h = r.headers
                s = "|".join(h.get(k, "") for k in ("ETag", "Last-Modified", "Content-Length"))
        except Exception:
            return ""
        if not s.strip("|"):
            return ""
        parts.append(s)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def download(url: str, dest: Path) -> str:
    """Streams url to dest; returns its SHA-256."""
    print(f"Downloading {url}")
    h = hashlib.sha256()
    with urllib.request.urlopen(_request(url), timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            h.update(chunk)
            f.write(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ outputs
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def gzip_file(src: Path, dest: Path) -> None:
    with open(src, "rb") as fi, open(dest, "wb") as raw:
        # mtime=0 -> deterministic bytes for identical input
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as gz:
            while chunk := fi.read(1 << 20):
                gz.write(chunk)


def set_output(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"{key}={value}\n")


def write_entries_db(rows: Iterable[Row], db_path: Path) -> int:
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.execute(SCHEMA)
    insert = "INSERT INTO entries VALUES (?,?,?,?,?,?,?,?)"
    batch: list[Row] = []
    n = 0
    for row in rows:
        batch.append(row)
        n += 1
        if len(batch) >= 5000:
            con.executemany(insert, batch)
            batch = []
    if batch:
        con.executemany(insert, batch)
    con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    con.commit()
    con.execute("VACUUM")
    con.close()
    return n


# --------------------------------------------------------------------- main
def run(
    *,
    source: str,
    builder_version: int,
    urls: list[str],
    build_rows: Callable[[list[Path]], Iterable[Row]],
    default_min_rows: int,
    extra_args: Optional[Callable[[argparse.ArgumentParser], None]] = None,
    argv: Optional[list[str]] = None,
    out_name: Optional[str] = None,
    upstream_label: str = "",
) -> int:
    """Complete CLI for one entries builder. Files are named <out_name>.db(.gz)."""
    out_name = out_name or source
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, nargs="+",
                    help="local upstream file(s) instead of downloading (testing)")
    ap.add_argument("--out-dir", type=Path, default=Path("out"))
    ap.add_argument("--prev-manifest", type=Path, help="skip build if upstream unchanged")
    ap.add_argument("--base-url", default=f"{REPO_RELEASES}/{source}")
    ap.add_argument("--min-rows", type=int, default=default_min_rows,
                    help="sanity floor; refuse to publish a suspiciously small db")
    if extra_args:
        extra_args(ap)
    args = ap.parse_args(argv)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    prev: dict = {}
    if args.prev_manifest and args.prev_manifest.exists():
        try:
            prev = json.loads(args.prev_manifest.read_text())
        except (ValueError, OSError):
            prev = {}
    same_builder = (prev.get("builderVersion") == builder_version
                    and prev.get("schemaVersion") == SCHEMA_VERSION)

    if args.input:
        files = list(args.input)
        upstream_sha = hashlib.sha256(
            "|".join(sha256_file(p) for p in files).encode()).hexdigest()
        upstream_stamp = ""
    else:
        upstream_stamp = stamp(urls)
        if same_builder and upstream_stamp and prev.get("upstreamStamp") == upstream_stamp:
            print("Upstream unchanged (headers) -- nothing to do")
            set_output("changed", "false")
            return 0
        files, shas = [], []
        for i, u in enumerate(urls):
            dest = out / f"{out_name}-upstream-{i}"
            shas.append(download(u, dest))
            files.append(dest)
        upstream_sha = hashlib.sha256("|".join(shas).encode()).hexdigest()
        if same_builder and prev.get("upstreamSha256") == upstream_sha:
            print("Upstream content unchanged -- nothing to do")
            set_output("changed", "false")
            return 0

    print("Parsing...")
    db_path = out / f"{out_name}.db"
    n = write_entries_db(build_rows(files), db_path)
    print(f"{source}: {n} rows")
    if n < args.min_rows:
        print(f"ERROR: only {n} rows (< {args.min_rows}); refusing to publish", file=sys.stderr)
        return 1

    gz_path = out / f"{out_name}.db.gz"
    gzip_file(db_path, gz_path)

    manifest = {
        "source": source,
        "schemaVersion": SCHEMA_VERSION,
        "builderVersion": builder_version,
        "upstreamTag": upstream_label or f"sha-{upstream_sha[:12]}",
        "upstreamSha256": upstream_sha,
        "upstreamStamp": upstream_stamp,
        "builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rowCount": n,
        "dbUrl": f"{args.base_url}/{out_name}.db.gz",
        "dbSha256": sha256_file(gz_path),
        "dbGzBytes": gz_path.stat().st_size,
        "dbBytes": db_path.stat().st_size,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    set_output("changed", "true")
    return 0
