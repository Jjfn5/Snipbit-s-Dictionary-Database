"""Usage: python -m build.build_dictionary jmdict ccedict kor spa fra --out dist
Downloads each source, parses it, writes dist/<source>.db (same entries schema as the app)."""
import argparse
import datetime
import gzip
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile

from .catalog import SOURCES
from .parsers import parse_ccedict, parse_jmdict, parse_kaikki, parse_krdict
from .schema import insert_rows, open_db

BATCH = 5000


def _get(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "snipbit-dictionary-builder"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as out:
        shutil.copyfileobj(r, out)


def _resolve_jmdict(cfg):
    req = urllib.request.Request(cfg["release_api"], headers={"User-Agent": "snipbit-dictionary-builder"})
    if os.environ.get("GITHUB_TOKEN"):
        req.add_header("Authorization", "Bearer " + os.environ["GITHUB_TOKEN"])
    with urllib.request.urlopen(req, timeout=60) as r:
        rel = json.load(r)
    for a in rel["assets"]:
        if re.match(cfg["asset_pattern"], a["name"]):
            return a["browser_download_url"], rel.get("tag_name", "")
    raise RuntimeError("no jmdict asset matched " + cfg["asset_pattern"])


def _prepare(url, archive, tmp, name):
    raw = os.path.join(tmp, name + ".download")
    _get(url, raw)
    if archive == "zip":
        with zipfile.ZipFile(raw) as z:
            member = next(n for n in z.namelist() if n.endswith(".json"))
            out = os.path.join(tmp, name + ".json")
            with z.open(member) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return out
    if archive == "gzip":
        out = os.path.join(tmp, name + ".txt")
        with gzip.open(raw, "rb") as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return out
    return raw


def build(source, out_dir):
    cfg = SOURCES[source]
    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(out_dir, source + ".db")
    if os.path.exists(db_path):
        os.remove(db_path)
    db = open_db(db_path)
    upstream = ""
    with tempfile.TemporaryDirectory() as tmp:
        if source == "jmdict":
            url, upstream = _resolve_jmdict(cfg)
            rows = parse_jmdict(_prepare(url, cfg["archive"], tmp, "jmdict"))
            _insert(db, rows)
        elif source == "ccedict":
            upstream = cfg["urls"][0]
            _insert(db, parse_ccedict(_prepare(upstream, cfg["archive"], tmp, "ccedict")))
        elif source == "kor":
            upstream = f"{len(cfg['urls'])} files"
            for i, url in enumerate(cfg["urls"]):
                _insert(db, parse_krdict(_prepare(url, "none", tmp, f"kor{i}")))
        else:
            upstream = cfg["urls"][0]
            _insert(db, parse_kaikki(_prepare(upstream, "none", tmp, source), source))
    count = db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    db.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)",
                   [("source", source), ("entry_count", str(count)), ("built_at", now), ("upstream", upstream)])
    db.commit()
    db.execute("VACUUM")
    db.close()
    print(f"{source}: {count} entries -> {db_path}")
    if count == 0:
        sys.exit(f"{source}: 0 entries -- refusing to publish")


def _insert(db, rows):
    batch = []
    for r in rows:
        batch.append(r)
        if len(batch) >= BATCH:
            insert_rows(db, batch)
            batch = []
    if batch:
        insert_rows(db, batch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", choices=sorted(SOURCES))
    ap.add_argument("--out", default="dist")
    args = ap.parse_args()
    for s in args.sources:
        build(s, args.out)
