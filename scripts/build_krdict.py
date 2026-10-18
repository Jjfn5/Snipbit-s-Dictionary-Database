#!/usr/bin/env python3
"""Builds kor.db.gz from the KrDict LMF XML mirror. Port of KrdictParser.kt.

Every real value hangs off a `<feat att="..." val="..."/>` child; only direct
children are searched (never descendants).
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import dictlib  # noqa: E402
from translit import build_latin_key  # noqa: E402

BASE = "https://raw.githubusercontent.com/spellcheck-ko/korean-dict-nikl-krdict/master"
URLS = [f"{BASE}/{n}.xml" for n in
        (5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 51947)]
BUILDER_VERSION = 1

ENGLISH = "영어"
PRONUNCIATION = "발음"
BEGINNER = "초급"


def children(el, tag):
    return [c for c in el if c.tag == tag]


def feat_val(el, att):
    for c in children(el, "feat"):
        if c.get("att") == att:
            return c.get("val") or ""
    return None


def entry_row(entry):
    lemmas = children(entry, "Lemma")
    headword = ((feat_val(lemmas[0], "writtenForm") or "").strip()) if lemmas else ""
    if not headword:
        return None

    groups = []
    for sense in children(entry, "Sense"):
        lemma_en = definition_en = None
        for eq in children(sense, "Equivalent"):
            if feat_val(eq, "language") == ENGLISH:
                v = feat_val(eq, "lemma")
                lemma_en = v.strip() if v is not None else None
                v = feat_val(eq, "definition")
                definition_en = v.strip() if v is not None else None
                break
        glosses, info = [], None
        if lemma_en:
            glosses.append(lemma_en)
            if definition_en:
                info = definition_en
        elif definition_en:
            glosses.append(definition_en)
        if glosses:
            g = {"g": glosses}
            if info:
                g["i"] = info
            groups.append(g)
    if not groups:
        return None

    pronunciation = None
    for wf in children(entry, "WordForm"):
        if feat_val(wf, "type") == PRONUNCIATION:
            v = feat_val(wf, "pronunciation")
            pronunciation = v.strip() if v is not None else None
            break
    reading = pronunciation if pronunciation and pronunciation != headword else None

    origin = (feat_val(entry, "origin") or "").strip()
    alt = origin if origin and origin != headword else None
    pos = (feat_val(entry, "partOfSpeech") or "").strip() or None
    common = 1 if feat_val(entry, "vocabularyLevel") == BEGINNER else 0

    return ("kor", headword, alt, reading, pos, dictlib.meaning_json(groups),
            common, build_latin_key("kor", headword))


# --------------------------------------------------------------- robust XML
# The upstream mirror is machine-generated and occasionally contains things a
# strict XML parser rejects ("not well-formed (invalid token)"): control
# characters, bare `&`, or a stray `<` / `"` inside a feat value. We repair
# those textually before parsing instead of failing the whole weekly build.
_ILLEGAL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
_BARE_AMP = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|#[0-9]+|#x[0-9a-fA-F]+);)")
_FEAT = re.compile(r'(<feat att=")([^"]*)(" val=")(.*?)("\s*/>)')


def _fix_feat(m):
    val = _BARE_AMP.sub("&amp;", m.group(4)).replace("<", "&lt;").replace('"', "&quot;")
    return m.group(1) + m.group(2) + m.group(3) + val + m.group(5)


def sanitize_xml(text: str) -> str:
    text = text.lstrip("\ufeff")
    text = _ILLEGAL.sub("", text)
    text = _FEAT.sub(_fix_feat, text)
    return _BARE_AMP.sub("&amp;", text)


_ENTRY = re.compile(r"<LexicalEntry\b.*?</LexicalEntry>", re.S)


def parse_entries(path):
    """Yields every <LexicalEntry> element in the file, tolerating bad markup."""
    text = sanitize_xml(Path(path).read_bytes().decode("utf-8", errors="replace"))
    try:
        parser = ET.XMLParser(encoding="utf-8")
        parser.feed(text.encode("utf-8"))
        root = parser.close()
    except ET.ParseError as e:
        line = e.position[0] if getattr(e, "position", None) else 0
        lines = text.splitlines()
        print(f"WARNING: {path}: {e}; falling back to per-entry parsing", file=sys.stderr)
        for i in range(max(0, line - 3), min(len(lines), line + 2)):
            print(f"  {i + 1}: {lines[i][:200]}", file=sys.stderr)
        skipped = 0
        for m in _ENTRY.finditer(text):
            try:
                yield ET.fromstring(m.group(0))
            except ET.ParseError:
                skipped += 1
        print(f"  skipped {skipped} unparseable entries", file=sys.stderr)
        return
    yield from root.iter("LexicalEntry")


def build_rows(files):
    for path in files:
        for entry in parse_entries(path):
            row = entry_row(entry)
            if row:
                yield row


def main(argv=None) -> int:
    return dictlib.run(
        source="kor",
        builder_version=BUILDER_VERSION,
        urls=URLS,
        build_rows=build_rows,
        default_min_rows=30_000,
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
