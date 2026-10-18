#!/usr/bin/env python3
"""Example-sentence databases for every language (Japanese included).

Two steps, run per language by .github/workflows/examples.yml:

  python scripts/build_examples.py tatoeba  --lang es --out out/tatoeba_es.db
  python scripts/build_examples.py assemble --lang es --tatoeba out/tatoeba_es.db \\
        --gemini examples/es/gemini.jsonl --out-dir out

`tatoeba` = port of ExampleCorpusParser's 3-file join (links -> English text ->
sentences) plus a word index. Japanese takes its index from Tatoeba's
Tanaka-tagged jpn_indices file; every other language tokenizes the sentence
itself (whitespace, or jieba for Chinese).

`assemble` appends the Gemini sentences (examples/<lang>/gemini.jsonl, the
fallback for words Tatoeba doesn't cover) and writes
<source>_examples.db.gz + examples-manifest.json. Gemini sentence ids are
GEMINI_ID_BASE + line number (Tatoeba ids stay below ~15 million, and the base
stays under 2^31 in case something reads ids as 32-bit ints), so the jsonl is
append-only: never reorder or delete lines.
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import io
import json
import re
import shutil
import sqlite3
import sys
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional

sys.path.insert(0, str(Path(__file__).parent))
import dictlib  # noqa: E402
from langs import (LANGS, Lang, index_key, normalize_token,  # noqa: E402,F401
                   tokenize_space, tokenizer_for)

SCHEMA_VERSION = 1   # must equal PrebuiltImporter.EXAMPLES_SCHEMA_VERSION in the app
BUILDER_VERSION = 2  # 2: Gemini fallback for every language, shared pipeline
GEMINI_ID_BASE = 1_000_000_000
BASE = "https://downloads.tatoeba.org/exports"
ENG_URL = f"{BASE}/per_language/eng/eng_sentences.tsv.bz2"
UA = "snipbit-dictionary-builder"
BATCH = 5000

_HEADWORD = re.compile(r"^([^\[{(~]+)")


# ------------------------------------------------------------------- helpers
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


def urls_for(lang: Lang) -> dict[str, str]:
    t = lang.tatoeba
    urls = {
        "sentences": f"{BASE}/per_language/{t}/{t}_sentences.tsv.bz2",
        "links": f"{BASE}/per_language/{t}/{t}-eng_links.tsv.bz2",
        "english": ENG_URL,
    }
    if lang.tokenizer == "tanaka":
        urls["indices"] = f"{BASE}/{t}_indices.tar.bz2"
    return urls


# ------------------------------------------------------------ tatoeba step
def build(files: dict[str, Path], out: Path, lang: Lang,
          tokenize: Optional[Callable[[str], list[str]]] = None) -> tuple[int, int]:
    """`tokenize` None => word index comes from files["indices"] (Japanese)."""
    source = lang.source
    eng_for: dict[int, int] = {}
    needed: set[int] = set()
    for line in bz2_lines(files["links"]):
        cols = line.split("\t")
        if len(cols) >= 2:
            a, b = to_int(cols[0]), to_int(cols[1])
            if a is not None and b is not None:
                eng_for.setdefault(a, b)   # first translation wins
                needed.add(b)
    print(f"links: {len(eng_for)} sentences with a translation link")

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
        if tokenize is not None:
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

    if tokenize is None:   # Tanaka-tagged word index (Japanese)
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
                    wrows.append((w, sid, source))
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


def cmd_tatoeba(args) -> int:
    lang = LANGS[args.lang]
    min_sent = lang.floor if args.min_sentences is None else args.min_sentences
    out = args.out or Path(f"out/tatoeba_{lang.code}.db")
    out.parent.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    for key, url in urls_for(lang).items():
        name = url.rsplit("/", 1)[1]
        if args.input_dir:
            files[key] = args.input_dir / name
        else:
            args.work_dir.mkdir(parents=True, exist_ok=True)
            files[key] = args.work_dir / name
            fetch(url, files[key])
    n_sent, _ = build(files, out, lang, tokenizer_for(lang))
    if n_sent < min_sent:
        print(f"ERROR: only {n_sent} sentences (< {min_sent})", file=sys.stderr)
        return 1
    return 0


# ----------------------------------------------------------- assemble step
def load_gemini(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def gemini_words(lang: Lang, row: dict,
                 tokenize: Optional[Callable[[str], list[str]]]) -> list[str]:
    """Index words for one Gemini sentence: the headword it was written for, plus
    (like Tatoeba sentences) every word in it when we can tokenize the language."""
    words = [index_key(lang, row["word"])]
    if tokenize is not None:
        words += tokenize(row["sentence"])
    return list(dict.fromkeys(words))


def cmd_assemble(args) -> int:
    lang = LANGS[args.lang]
    base_url = args.base_url or f"{dictlib.REPO_RELEASES}/{lang.source}"
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    gemini_rows = load_gemini(args.gemini)
    gemini_sha = dictlib.sha256_file(args.gemini) if args.gemini.exists() else "none"
    content_key = hashlib.sha256(
        f"{dictlib.sha256_file(args.tatoeba)}|{gemini_sha}|{BUILDER_VERSION}|{SCHEMA_VERSION}".encode()
    ).hexdigest()

    if args.prev_manifest and args.prev_manifest.exists():
        try:
            if json.loads(args.prev_manifest.read_text()).get("contentKey") == content_key:
                print("Examples unchanged -- nothing to publish")
                dictlib.set_output("changed", "false")
                return 0
        except (ValueError, OSError):
            pass

    name = f"{lang.source}_examples"
    db_path = out / f"{name}.db"
    shutil.copyfile(args.tatoeba, db_path)
    tokenize = tokenizer_for(lang)
    con = sqlite3.connect(db_path)
    sent, words = [], []
    for n, r in enumerate(gemini_rows, 1):
        sid = GEMINI_ID_BASE + n
        sent.append((sid, lang.source, r["sentence"], r["translation"]))
        words.extend((w, sid, lang.source) for w in gemini_words(lang, r, tokenize))
    con.executemany("INSERT OR REPLACE INTO sentences VALUES (?,?,?,?)", sent)
    con.executemany("INSERT INTO sentence_words VALUES (?,?,?)", words)
    con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    con.commit()
    n_sentences = con.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
    n_words = con.execute("SELECT COUNT(*) FROM sentence_words").fetchone()[0]
    con.execute("VACUUM")
    con.close()

    gz_path = out / f"{name}.db.gz"
    dictlib.gzip_file(db_path, gz_path)
    manifest = {
        "source": lang.source,
        "kind": "examples",
        "schemaVersion": SCHEMA_VERSION,
        "builderVersion": BUILDER_VERSION,
        "contentKey": content_key,
        "builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rowCount": n_sentences,
        "wordRowCount": n_words,
        "geminiCount": len(gemini_rows),
        "dbUrl": f"{base_url}/{name}.db.gz",
        "dbSha256": dictlib.sha256_file(gz_path),
        "dbGzBytes": gz_path.stat().st_size,
        "dbBytes": db_path.stat().st_size,
    }
    (out / "examples-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    dictlib.set_output("changed", "true")
    return 0


# ----------------------------------------------------------------------- cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tatoeba", help="build the Tatoeba half")
    t.add_argument("--lang", choices=sorted(LANGS), required=True)
    t.add_argument("--out", type=Path)
    t.add_argument("--work-dir", type=Path, default=Path("out/tatoeba_work"))
    t.add_argument("--input-dir", type=Path,
                   help="use already-downloaded files named like the URLs' basenames (testing)")
    t.add_argument("--min-sentences", type=int, help="sanity floor override")
    t.set_defaults(fn=cmd_tatoeba)

    a = sub.add_parser("assemble", help="add Gemini sentences, gzip, write the manifest")
    a.add_argument("--lang", choices=sorted(LANGS), required=True)
    a.add_argument("--tatoeba", type=Path, required=True)
    a.add_argument("--gemini", type=Path)
    a.add_argument("--out-dir", type=Path, default=Path("out"))
    a.add_argument("--prev-manifest", type=Path)
    a.add_argument("--base-url")
    a.set_defaults(fn=cmd_assemble)

    args = ap.parse_args(argv)
    if args.cmd == "assemble" and args.gemini is None:
        args.gemini = Path(f"examples/{args.lang}/gemini.jsonl")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
