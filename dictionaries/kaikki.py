#!/usr/bin/env python3
"""Spanish -- Kaikki (Wiktionary) dictionary -> spa.db.gz.

The parser is shared with every other Kaikki language (scripts/kaikki.py);
this file only picks the language.

  python dictionaries/es/kaikki.py
  python dictionaries/es/kaikki.py --input some.jsonl --min-rows 0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import kaikki  # noqa: E402

if __name__ == "__main__":
    sys.exit(kaikki.main(["--lang", "spa", *sys.argv[1:]]))
