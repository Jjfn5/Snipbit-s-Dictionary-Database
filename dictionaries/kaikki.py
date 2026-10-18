#!/usr/bin/env python3
"""French -- Kaikki (Wiktionary) dictionary -> fra.db.gz.

The parser is shared with every other Kaikki language (scripts/kaikki.py);
this file only picks the language.

  python dictionaries/fr/kaikki.py
  python dictionaries/fr/kaikki.py --input some.jsonl --min-rows 0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import kaikki  # noqa: E402

if __name__ == "__main__":
    sys.exit(kaikki.main(["--lang", "fra", *sys.argv[1:]]))
