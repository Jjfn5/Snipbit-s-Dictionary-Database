"""Python port of Transliteration.kt / dictionary_transliteration.dart.

Keep in sync by hand with the Dart/Kotlin versions: the app's search
matches against `reading_latin`, so every table and rule here must
produce byte-identical output to the app's own implementation.
"""
from __future__ import annotations

import re
from typing import Optional


def _pairs(s: str) -> dict[str, str]:
    t = s.split()
    return dict(zip(t[0::2], t[1::2]))


# ---------------------------------------------------------------- diacritics
_FOLD_GROUPS = {
    "a": "àáâãäåāăą", "A": "ÀÁÂÃÄÅĀĂĄ",
    "c": "çćĉċč", "C": "ÇĆĈĊČ",
    "e": "èéêëēĕėęě", "E": "ÈÉÊËĒĔĖĘĚ",
    "i": "ìíîïĩīĭįı", "I": "ÌÍÎÏĨĪĬĮ",
    "n": "ñńņň", "N": "ÑŃŅŇ",
    "o": "òóôõöøōŏő", "O": "ÒÓÔÕÖØŌŎŐ",
    "u": "ùúûüũūŭůűų", "U": "ÙÚÛÜŨŪŬŮŰŲ",
    "y": "ýÿŷ", "Y": "ÝŸŶ",
}
_DIACRITIC_FOLD: dict[str, str] = {
    ch: base for base, chars in _FOLD_GROUPS.items() for ch in chars
}
_DIACRITIC_FOLD.update({"ß": "ss", "œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE"})


def fold_diacritics(s: str) -> str:
    return "".join(_DIACRITIC_FOLD.get(c, c) for c in s)


# -------------------------------------------------------------------- pinyin
_PINYIN_NOISE = re.compile(r"[0-9\s\u00b7]")


def strip_pinyin_tones(s: str) -> str:
    return _PINYIN_NOISE.sub("", s).lower()


# ---------------------------------------------------------------- kana->romaji
_MONO = _pairs("""
あ a い i う u え e お o
か ka き ki く ku け ke こ ko
が ga ぎ gi ぐ gu げ ge ご go
さ sa し shi す su せ se そ so
ざ za じ ji ず zu ぜ ze ぞ zo
た ta ち chi つ tsu て te と to
だ da ぢ ji づ zu で de ど do
な na に ni ぬ nu ね ne の no
は ha ひ hi ふ fu へ he ほ ho
ば ba び bi ぶ bu べ be ぼ bo
ぱ pa ぴ pi ぷ pu ぺ pe ぽ po
ま ma み mi む mu め me も mo
や ya ゆ yu よ yo
ら ra り ri る ru れ re ろ ro
わ wa ゐ wi ゑ we を wo
ん n
ぁ a ぃ i ぅ u ぇ e ぉ o
""")

_DIGRAPH = _pairs("""
きゃ kya きゅ kyu きょ kyo
ぎゃ gya ぎゅ gyu ぎょ gyo
しゃ sha しゅ shu しょ sho
じゃ ja じゅ ju じょ jo
ちゃ cha ちゅ chu ちょ cho
ぢゃ ja ぢゅ ju ぢょ jo
にゃ nya にゅ nyu にょ nyo
ひゃ hya ひゅ hyu ひょ hyo
びゃ bya びゅ byu びょ byo
ぴゃ pya ぴゅ pyu ぴょ pyo
みゃ mya みゅ myu みょ myo
りゃ rya りゅ ryu りょ ryo
ふぁ fa ふぃ fi ふぇ fe ふぉ fo
ゔぁ va ゔぃ vi ゔぇ ve ゔぉ vo
てぃ ti でぃ di とぅ tu どぅ du
うぃ wi うぇ we うぉ wo
ちぇ che じぇ je しぇ she
つぁ tsa つぇ tse つぉ tso
""")

_KATA_START, _KATA_END, _KATA_OFFSET = 0x30A1, 0x30FA, 0x60
_VOWELS = "aeiou"


def _fold_katakana(s: str) -> str:
    return "".join(
        chr(ord(c) - _KATA_OFFSET) if _KATA_START <= ord(c) <= _KATA_END else c
        for c in s
    )


def kana_to_romaji(text: str) -> str:
    chars = _fold_katakana(text)
    buf: list[str] = []
    pending_sokuon = False
    i = 0
    n = len(chars)
    while i < n:
        c = chars[i]
        if c == "っ":
            pending_sokuon = True
            i += 1
            continue
        if c == "ー":
            for ch in reversed(buf):
                if ch in _VOWELS:
                    buf.append(ch)
                    break
            i += 1
            continue

        romaji: Optional[str] = None
        if i + 1 < n:
            d = _DIGRAPH.get(c + chars[i + 1])
            if d is not None:
                romaji = d
                i += 1
        if romaji is None:
            romaji = _MONO.get(c)

        if romaji is None:  # stray kanji/punctuation: skip, drop pending sokuon
            pending_sokuon = False
            i += 1
            continue

        if pending_sokuon and romaji and romaji[0] not in _VOWELS and romaji[0] != "n":
            buf.append(romaji[0])
        pending_sokuon = False
        buf.extend(romaji)
        i += 1
    return "".join(buf)


# -------------------------------------------------------------- hangul->romaja
_H_LEADS = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
_H_VOWELS = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe",
             "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"]
_H_TAILS = ["", "g", "kk", "gs", "n", "nj", "nh", "d", "l", "lg", "lm", "lb", "ls", "lt", "lp", "lh",
            "m", "b", "bs", "s", "ss", "ng", "j", "ch", "k", "t", "p", "h"]


def hangul_to_romaja(text: str) -> str:
    out = []
    for ch in text:
        cp = ord(ch)
        if cp < 0xAC00 or cp > 0xD7A3:
            continue
        off = cp - 0xAC00
        out.append(_H_LEADS[off // (28 * 21)])
        out.append(_H_VOWELS[(off // 28) % 21])
        out.append(_H_TAILS[off % 28])
    return "".join(out)


# ------------------------------------------------------------------ entry point
def build_latin_key(source_id: str, headword: str, reading: Optional[str] = None) -> Optional[str]:
    if source_id == "jmdict":
        return kana_to_romaji(reading if reading else headword) or None
    if source_id == "ccedict":
        return strip_pinyin_tones(reading if reading else headword) or None
    if source_id == "kor":
        return hangul_to_romaja(headword) or None
    if source_id in ("spa", "fra"):
        return fold_diacritics(headword).lower() or None
    return None
