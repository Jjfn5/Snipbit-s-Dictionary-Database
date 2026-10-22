"""Word-frequency ranks for the entries builders (dictlib.run and jmdict.py).

Every entries row gets a `freq_rank` (1 = most frequent word of that language,
NULL = not in the list). The rank is looked up from a per-language frequency
list and attached while the entries db is written, so the app only has to
download the one db it already downloads.

Where the lists come from
  * default: the `wordfreq` package (pip install wordfreq), which ships one
    ranked list per language -- ja, zh, ko, es, fr here. Its data is
    CC BY-SA 4.0 (see README for the attribution).
  * `--freq-file FILE`: any plain-text list instead (one word per line, most
    frequent first; an optional count after a tab/space is ignored). Used by
    the tests and handy for swapping in a better list for one language.

Matching is deliberately conservative -- see `candidates()`. A missing rank is
better than a wrong one, so the app should sort NULL ranks last.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).parent))
from langs import BY_SOURCE, index_key  # noqa: E402

WORDFREQ_LIMIT = 1_000_000  # more than any wordfreq list holds: "give me all of it"


# ------------------------------------------------------------------ loading
def source_label(freq_file: Optional[Path]) -> str:
    """Cheap id of the frequency data, stored in the manifest so a new
    wordfreq release (or a different --freq-file) triggers a rebuild."""
    if freq_file:
        import hashlib
        h = hashlib.sha256(Path(freq_file).read_bytes()).hexdigest()
        return f"file-{h[:12]}"
    try:
        from importlib.metadata import version
        return f"wordfreq-{version('wordfreq')}"
    except Exception:
        return "wordfreq-unknown"


def _read_word_file(path: Path) -> Iterable[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line.split(None, 1)[0]


def _read_wordfreq(lang_code: str) -> Iterable[str]:
    try:
        import wordfreq
    except ImportError as e:
        raise RuntimeError(
            "The `wordfreq` package is needed for frequency ranks "
            "(pip install wordfreq), or pass --freq-file / --no-freq."
        ) from e
    return wordfreq.top_n_list(lang_code, WORDFREQ_LIMIT, wordlist="best")


def _usable(token: str) -> bool:
    """Drops punctuation and bare numbers, which can never be a headword."""
    return any(c.isalpha() for c in token)


def load_ranks(source: str, freq_file: Optional[Path] = None) -> dict[str, int]:
    """word -> rank (1 = most frequent) for the language of `source`."""
    lang = BY_SOURCE[source]
    words = _read_word_file(freq_file) if freq_file else _read_wordfreq(lang.code)
    ranks: dict[str, int] = {}
    for w in words:
        if not _usable(w):
            continue
        key = index_key(lang, w)   # es/fr: lowercase + NFC, same as the headword side
        if key not in ranks:
            ranks[key] = len(ranks) + 1
    if not ranks:
        raise RuntimeError(f"Frequency list for {source} is empty")
    return ranks


# ----------------------------------------------------------------- matching
def _has_han(s: str) -> bool:
    return any(0x3400 <= ord(c) <= 0x9FFF for c in s)


def _usually_kana(meaning_json: str) -> bool:
    """JMdict `uk` misc tag: the word is normally written in kana even though
    it has a kanji form (e.g. 事 is usually こと)."""
    try:
        groups = json.loads(meaning_json)
    except ValueError:
        return False
    for g in groups:
        m = g.get("m") or ""
        if "usually written using kana alone" in m or "uk" in [t.strip() for t in m.split(",")]:
            return True
    return False


def candidates(source: str, headword: str, alt: Optional[str],
               reading: Optional[str], meaning_json: str) -> list[str]:
    """Spellings of an entry to look up, best first (the lowest rank wins).

    ja: headword, second kanji spelling, and the kana reading ONLY for
        usually-kana words -- otherwise 橋 and 箸 would both inherit the rank
        of はし.
    zh: simplified headword (wordfreq's Chinese list is simplified).
    ko: headword. wordfreq splits Korean into morphemes, so conjugating
        verbs/adjectives (먹다) mostly stay unranked rather than get a wrong rank.
    es/fra: lowercase NFC headword; multi-word entries simply don't match.
    """
    lang = BY_SOURCE[source]
    out = [index_key(lang, headword)]
    if source == "jmdict":
        if alt:
            out.append(alt)
        if reading and reading != headword and _has_han(headword) and _usually_kana(meaning_json):
            out.append(reading)
    return out


class Ranker:
    """Callable: 8-column entries row -> 9-column row with freq_rank appended."""

    def __init__(self, source: str, ranks: dict[str, int]):
        self.source = source
        self.ranks = ranks
        self.total = 0
        self.ranked = 0

    def rank_of(self, row: tuple) -> Optional[int]:
        _src, headword, alt, reading, _pos, meaning = row[:6]
        found = [self.ranks[c] for c in candidates(self.source, headword, alt, reading, meaning)
                 if c in self.ranks]
        return min(found) if found else None

    def __call__(self, row: tuple) -> tuple:
        rank = self.rank_of(row)
        self.total += 1
        if rank is not None:
            self.ranked += 1
        return tuple(row[:8]) + (rank,)


def make_ranker(source: str, freq_file: Optional[Path] = None) -> Ranker:
    return Ranker(source, load_ranks(source, freq_file))
