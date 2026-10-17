"""Per-language configuration shared by the examples builder and the Gemini
filler: Tatoeba codes, word-index tokenizers, and the language-specific parts
of Gemini validation (script check, "is the target word in the sentence").

Release tag == dictionary source id, so each language is one GitHub release.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class Lang:
    code: str          # short id used on the command line / in examples/<code>/
    source: str        # dictionary source id == release tag == `source` column
    name: str          # English name, used in release titles and Gemini prompts
    tatoeba: str       # Tatoeba language code
    floor: int         # sanity floor: refuse to publish fewer Tatoeba sentences
    tokenizer: str     # "tanaka" (pre-tagged index file), "jieba" or "space"
    length_rule: str   # Gemini prompt: how long the sentence should be
    markup_rule: str   # Gemini prompt: what must NOT appear in the sentence
    min_chars: int     # validation bounds on the sentence length (characters)
    max_chars: int


LANGS: dict[str, Lang] = {l.code: l for l in [
    Lang("ja", "jmdict", "Japanese", "jpn", 50_000, "tanaka",
         "10 to 30 characters", "No furigana, no romaji, no brackets or quotation marks in the sentence.",
         4, 80),
    Lang("zh", "ccedict", "Chinese", "cmn", 20_000, "jieba",
         "6 to 25 characters, Simplified Chinese", "No pinyin, no brackets or quotation marks in the sentence.",
         4, 80),
    Lang("ko", "kor", "Korean", "kor", 3_000, "space",
         "8 to 30 characters, polite speech style", "No romanization, no brackets or quotation marks in the sentence.",
         4, 90),
    Lang("es", "spa", "Spanish", "spa", 100_000, "space",
         "6 to 14 words", "No brackets or quotation marks in the sentence.",
         10, 140),
    Lang("fr", "fra", "French", "fra", 100_000, "space",
         "6 to 14 words", "No brackets or quotation marks in the sentence.",
         10, 140),
]}
BY_SOURCE = {l.source: l for l in LANGS.values()}


# ------------------------------------------------------------------ tokenizers
_EDGE = re.compile(r"^[\W_]+|[\W_]+$")
_WS = re.compile(r"\s+")
_ELISION = re.compile(r"['\u2019]")


def normalize_token(tok: str) -> Optional[str]:
    """NFC, lowercase, leading/trailing non-alphanumerics stripped (None if empty)."""
    t = _EDGE.sub("", unicodedata.normalize("NFC", tok.strip()))
    return t.lower() or None


def tokenize_space(text: str) -> list[str]:
    """Whitespace words; l'eau is indexed whole AND as l + eau."""
    out: list[str] = []
    for raw in _WS.split(text):
        whole = normalize_token(raw)
        if not whole:
            continue
        out.append(whole)
        if _ELISION.search(whole):
            for piece in _ELISION.split(whole):
                p = normalize_token(piece)
                if p:
                    out.append(p)
    return out


def make_jieba_tokenizer() -> Callable[[str], list[str]]:
    import jieba  # imported lazily: only the Chinese build needs it
    jieba.setLogLevel(60)

    def tokenize(text: str) -> list[str]:
        return [t for t in (normalize_token(w) for w in jieba.lcut(text)) if t]
    return tokenize


def tokenizer_for(lang: Lang) -> Optional[Callable[[str], list[str]]]:
    """None for Japanese: its word index comes from Tatoeba's tagged indices file."""
    if lang.tokenizer == "jieba":
        return make_jieba_tokenizer()
    if lang.tokenizer == "space":
        return tokenize_space
    return None


def index_key(lang: Lang, headword: str) -> str:
    """How a dictionary headword is spelled in `sentence_words`."""
    if lang.code in ("es", "fr"):
        return normalize_token(headword) or headword
    return headword


# --------------------------------------------------------- Gemini validation
def _is_kana(c: str) -> bool:
    return 0x3040 <= ord(c) <= 0x30FF


def _is_han(c: str) -> bool:
    return 0x4E00 <= ord(c) <= 0x9FFF


def _is_hangul(c: str) -> bool:
    return 0xAC00 <= ord(c) <= 0xD7A3 or 0x1100 <= ord(c) <= 0x11FF


def has_japanese(s: str) -> bool:
    return any(_is_han(c) or _is_kana(c) for c in s)


VERB_ADJ_TAGS = ("v1", "v5", "vk", "vz", "vs-i", "vs-s", "adj-i")


def word_stem(headword: str, pos: str) -> str:
    """Japanese: part of the headword that survives conjugation (食べる -> 食べ)."""
    tags = [t.strip() for t in (pos or "").split(",")]
    if len(headword) >= 2 and any(t.startswith(VERB_ADJ_TAGS) for t in tags):
        return headword[:-1]
    return headword


def script_ok(lang: Lang, sentence: str) -> Optional[str]:
    """None if the sentence is in the right script, else a reject reason."""
    if lang.code == "ja":
        return None if has_japanese(sentence) else "not japanese"
    if lang.code == "zh":
        if not any(_is_han(c) for c in sentence):
            return "not chinese"
        if any(_is_kana(c) or _is_hangul(c) for c in sentence):
            return "kana/hangul in chinese sentence"
        return None
    if lang.code == "ko":
        return None if any(_is_hangul(c) for c in sentence) else "not korean"
    letters = [c for c in sentence if c.isalpha()]
    if not letters:
        return "no letters"
    if any(_is_han(c) or _is_kana(c) or _is_hangul(c) for c in letters):
        return "wrong script"
    latin = sum(1 for c in letters if ord(c) < 0x250)
    return None if latin / len(letters) >= 0.9 else "not latin script"


def target_ok(lang: Lang, headword: str, pos: str, sentence: str, surface: str) -> bool:
    """Does the sentence really use the target word (possibly conjugated)?"""
    if lang.code == "ja":
        return word_stem(headword, pos) in sentence
    if lang.code == "zh":
        return headword in sentence
    if lang.code == "ko":
        stem = headword[:-1] if len(headword) >= 2 and headword.endswith("다") else headword
        return stem in sentence
    # es / fr: irregular verbs can look nothing like the infinitive, so verbs
    # rely on `surface` being in the sentence; other words must share their stem.
    if (pos or "").lower() == "verb":
        return True
    stem = headword.lower()
    if len(stem) >= 4:
        stem = stem[:-1]   # casa -> casas, bonito -> bonita
    return stem in sentence.lower()


# ------------------------------------------------- Gemini candidate filtering
_FORM_OF = re.compile(
    r"\b(inflection|inflected|plural|singular|participle|person|tense|subjunctive|imperative|"
    r"gerund|indicative|feminine|masculine|diminutive|augmentative|superlative|comparative|"
    r"alternative|obsolete|misspelling|abbreviation|apocopic|elided|contraction|variant|"
    r"archaic|dated|nonstandard|rare)\b.*\bof\b", re.I)
_SKIP_POS = {"name", "prefix", "suffix", "infix", "symbol", "character", "phrase", "proverb",
             "contraction", "abbrev", "romanization", "prep_phrase"}
_ZH_JUNK = re.compile(r"^(surname |variant of|old variant|abbr\. for|see |same as|radical|"
                      r"Kangxi|erhua variant|archaic variant|CL:)", re.I)


def worth_generating(lang: Lang, headword: str, pos: str, gloss: str) -> bool:
    """Skips entries that make poor examples (inflected forms, names, radicals...).
    Heuristics only: a few misses are fine, a wasted Gemini slot is cheap."""
    if lang.code in ("es", "fr"):
        if " " in headword or any(ch.isdigit() for ch in headword):
            return False
        if (pos or "").lower() in _SKIP_POS:
            return False
        return not _FORM_OF.search(gloss)
    if lang.code == "zh":
        return not _ZH_JUNK.match(gloss) and "Kangxi radical" not in gloss
    return True
