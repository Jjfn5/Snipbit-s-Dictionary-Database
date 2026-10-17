#!/usr/bin/env python3
"""Builds ccedict.db.gz from CC-CEDICT. Port of CcCedictParser.kt.

Line format: `Traditional Simplified [pin1 yin1] /def 1/def 2/` (v1) or
`[[pin1yin1]]` (v2) -- both occur in the "Mixed" download.
"""
from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import dictlib  # noqa: E402
from translit import build_latin_key  # noqa: E402

URL = "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz"
BUILDER_VERSION = 1

LINE = re.compile(r"^(\S+)\s+(\S+)\s+(?:\[\[([^\]]*)\]\]|\[([^\]]*)\])\s+/(.*)/$")
TONE_BOUNDARY = re.compile(r"([1-5])(?=[A-Za-z])")


def normalize_pinyin_spacing(raw: str) -> str:
    return TONE_BOUNDARY.sub(lambda m: m.group(1) + " ", raw)


def parse_line(raw_line: str):
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    m = LINE.match(line)
    if not m:
        return None
    traditional, simplified, g3, g4, defs = m.groups()
    pinyin_raw = g3 if g3 is not None else (g4 if g4 is not None else "")
    definitions = [d.strip() for d in defs.split("/") if d.strip()]
    if not definitions:
        return None
    pinyin = normalize_pinyin_spacing(pinyin_raw)
    return (
        "ccedict",
        simplified,
        None if traditional == simplified else traditional,
        pinyin,
        None,
        dictlib.meaning_json([{"g": [d]} for d in definitions]),
        0,
        build_latin_key("ccedict", simplified, pinyin),
    )


def build_rows(files: list[Path]):
    path = files[0]
    opener = gzip.open if path.suffix == ".gz" or _is_gzip(path) else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            row = parse_line(raw)
            if row:
                yield row


def _is_gzip(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def main(argv=None) -> int:
    return dictlib.run(
        source="ccedict",
        builder_version=BUILDER_VERSION,
        urls=[URL],
        build_rows=build_rows,
        default_min_rows=100_000,
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
