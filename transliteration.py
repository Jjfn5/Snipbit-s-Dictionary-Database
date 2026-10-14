"""Port of dictionary_transliteration.dart / Transliteration.kt. Must stay identical:
the app searches reading_latin with keys produced by this logic."""
import re

_FOLD_SRC = {
    "a": "àáâãäåāăą", "c": "çćĉċč", "e": "èéêëēĕėęě", "i": "ìíîïĩīĭįı",
    "n": "ñńņň", "o": "òóôõöøōŏő", "u": "ùúûüũūŭůűų", "y": "ýÿŷ",
}
_DIACRITIC_FOLD = {}
for _base, _chars in _FOLD_SRC.items():
    for _ch in _chars:
        _DIACRITIC_FOLD[_ch] = _base
        if _ch != "ı":  # dotless i has no uppercase entry in the table
            _DIACRITIC_FOLD[_ch.upper()] = _base.upper()
_DIACRITIC_FOLD.update({"ß": "ss", "œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE"})


def _fold_diacritics(s):
    return "".join(_DIACRITIC_FOLD.get(ch, ch) for ch in s)


_PINYIN_NOISE = re.compile(r"[0-9\s\u00b7]")


def _strip_pinyin_tones(p):
    return _PINYIN_NOISE.sub("", p).lower()


_MONO = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "ゐ": "wi", "ゑ": "we", "を": "wo", "ん": "n",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
}

_DIGRAPH = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo", "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho", "じゃ": "ja", "じゅ": "ju", "じょ": "jo",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho", "ぢゃ": "ja", "ぢゅ": "ju", "ぢょ": "jo",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo", "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo", "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo", "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
    "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo",
    "ゔぁ": "va", "ゔぃ": "vi", "ゔぇ": "ve", "ゔぉ": "vo",
    "てぃ": "ti", "でぃ": "di", "とぅ": "tu", "どぅ": "du",
    "うぃ": "wi", "うぇ": "we", "うぉ": "wo",
    "ちぇ": "che", "じぇ": "je", "しぇ": "she", "つぁ": "tsa", "つぇ": "tse", "つぉ": "tso",
}

_VOWELS = "aeiou"


def _fold_katakana(s):
    return "".join(
        chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30FA else c for c in s
    )


def _kana_to_romaji(text):
    chars = list(_fold_katakana(text))
    out = []
    pending = False
    i = 0
    while i < len(chars):
        c = chars[i]
        if c == "っ":
            pending = True
            i += 1
            continue
        if c == "ー":
            joined = "".join(out)
            for j in range(len(joined) - 1, -1, -1):
                if joined[j] in _VOWELS:
                    out.append(joined[j])
                    break
            i += 1
            continue
        romaji = None
        if i + 1 < len(chars):
            d = _DIGRAPH.get(c + chars[i + 1])
            if d is not None:
                romaji = d
                i += 1
        if romaji is None:
            romaji = _MONO.get(c)
        if romaji is None:
            pending = False
            i += 1
            continue
        if pending and romaji and romaji[0] not in _VOWELS and romaji[0] != "n":
            out.append(romaji[0])
        pending = False
        out.append(romaji)
        i += 1
    return "".join(out)


_LEADS = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
_HV = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe",
       "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"]
_TAILS = ["", "g", "kk", "gs", "n", "nj", "nh", "d", "l", "lg", "lm", "lb", "ls", "lt", "lp", "lh",
          "m", "b", "bs", "s", "ss", "ng", "j", "ch", "k", "t", "p", "h"]


def _hangul_to_romaja(text):
    out = []
    for ch in text:
        cp = ord(ch)
        if cp < 0xAC00 or cp > 0xD7A3:
            continue
        off = cp - 0xAC00
        out.append(_LEADS[off // (28 * 21)])
        out.append(_HV[(off // 28) % 21])
        out.append(_TAILS[off % 28])
    return "".join(out)


def build_latin_key(source_id, headword, reading=None):
    if source_id == "jmdict":
        key = _kana_to_romaji(reading if reading else headword)
    elif source_id == "ccedict":
        key = _strip_pinyin_tones(reading if reading else headword)
    elif source_id == "kor":
        key = _hangul_to_romaja(headword)
    elif source_id in ("spa", "fra"):
        key = _fold_diacritics(headword).lower()
    else:
        return None
    return key or None
