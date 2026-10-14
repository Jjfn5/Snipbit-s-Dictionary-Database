"""Ports of JmdictParser / CcCedictParser / KaikkiParser / KrdictParser (.kt).
Each parser yields entry-row dicts in file order, same filtering rules as the app."""
import json
import re
import xml.etree.ElementTree as ET

from .schema import meaning_json, sense_group
from .transliteration import build_latin_key


# ---------------------------------------------------------------- JMdict
def _str_list(v):
    return [x for x in (v or []) if isinstance(x, str) and x]


def parse_jmdict(path):
    """jmdict-simplified JSON: { words: [ {kanji, kana, sense} ] }."""
    with open(path, encoding="utf-8") as f:
        words = json.load(f).get("words", [])
    for w in words:
        kanji = w.get("kanji") or []
        kana = w.get("kana") or []
        first_kanji = (kanji[0].get("text") if kanji else None) or None
        first_kana = (kana[0].get("text") if kana else None) or None
        headword = first_kanji or first_kana
        if not headword:
            continue
        reading = first_kana
        is_common = any(k.get("common") for k in kanji) or any(k.get("common") for k in kana)

        groups, all_pos = [], []
        for sense in w.get("sense") or []:
            glosses = [
                g.get("text", "")
                for g in sense.get("gloss") or []
                if g.get("lang") in (None, "eng") and g.get("text")
            ]
            if not glosses:
                continue
            pos = _str_list(sense.get("partOfSpeech"))
            for p in pos:
                if p not in all_pos:
                    all_pos.append(p)
            field = _str_list(sense.get("field"))
            misc = _str_list(sense.get("misc"))
            info = _str_list(sense.get("info"))
            groups.append(sense_group(
                glosses, ", ".join(pos), ", ".join(field), ", ".join(misc), "; ".join(info)))
        if not groups:
            continue
        alt = (kanji[1].get("text") if len(kanji) > 1 else None) or None
        yield {
            "source": "jmdict", "headword": headword, "alt_headword": alt, "reading": reading,
            "part_of_speech": ", ".join(all_pos) or None, "meaning": meaning_json(groups),
            "is_common": is_common,
            "reading_latin": build_latin_key("jmdict", headword, reading),
        }


# ---------------------------------------------------------------- CC-CEDICT
_LINE = re.compile(r"^(\S+)\s+(\S+)\s+(?:\[\[([^\]]*)\]\]|\[([^\]]*)\])\s+/(.*)/$")
_TONE_BOUNDARY = re.compile(r"([1-5])(?=[A-Za-z])")


def parse_ccedict(path):
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = _LINE.match(line)
            if not m:
                continue
            trad, simp = m.group(1), m.group(2)
            pinyin_raw = m.group(3) if m.group(3) is not None else (m.group(4) or "")
            defs = [d.strip() for d in m.group(5).split("/") if d.strip()]
            if not defs:
                continue
            pinyin = _TONE_BOUNDARY.sub(r"\1 ", pinyin_raw)
            yield {
                "source": "ccedict", "headword": simp,
                "alt_headword": None if trad == simp else trad, "reading": pinyin,
                "part_of_speech": None,
                "meaning": meaning_json([sense_group([d]) for d in defs]),
                "is_common": False,
                "reading_latin": build_latin_key("ccedict", simp, pinyin),
            }


# ---------------------------------------------------------------- Kaikki (spa / fra)
def parse_kaikki(path, source_id):
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            headword = (d.get("word") or "").strip()
            if not headword:
                continue
            pos = (d.get("pos") or "").strip() or None
            groups = []
            for sense in d.get("senses") or []:
                path_ = sense.get("glosses") or []
                if not path_:
                    continue
                gloss = (path_[-1] or "").strip()  # leaf of the breadcrumb only
                if gloss:
                    groups.append(sense_group([gloss]))
            if not groups:
                continue
            yield {
                "source": source_id, "headword": headword, "alt_headword": None, "reading": None,
                "part_of_speech": pos, "meaning": meaning_json(groups), "is_common": False,
                "reading_latin": build_latin_key(source_id, headword),
            }


# ---------------------------------------------------------------- KrDict (LMF XML)
def _children(el, tag):
    return [c for c in el if c.tag == tag]


def _feat(el, att):
    for c in _children(el, "feat"):
        if c.get("att") == att:
            return c.get("val", "")
    return None


def parse_krdict(path):
    root = ET.parse(path).getroot()
    for entry in root.iter("LexicalEntry"):
        lemmas = _children(entry, "Lemma")
        headword = ((_feat(lemmas[0], "writtenForm") or "").strip()) if lemmas else ""
        if not headword:
            continue
        groups = []
        for sense in _children(entry, "Sense"):
            lemma = definition = None
            for eq in _children(sense, "Equivalent"):
                if _feat(eq, "language") == "영어":
                    lemma = (_feat(eq, "lemma") or "").strip()
                    definition = (_feat(eq, "definition") or "").strip()
                    break
            if lemma:
                groups.append(sense_group([lemma], info=definition or None))
            elif definition:
                groups.append(sense_group([definition]))
        if not groups:
            continue
        pron = None
        for wf in _children(entry, "WordForm"):
            if _feat(wf, "type") == "발음":
                pron = (_feat(wf, "pronunciation") or "").strip()
                break
        reading = pron if pron and pron != headword else None
        origin = (_feat(entry, "origin") or "").strip()
        alt = origin if origin and origin != headword else None
        pos = (_feat(entry, "partOfSpeech") or "").strip() or None
        yield {
            "source": "kor", "headword": headword, "alt_headword": alt, "reading": reading,
            "part_of_speech": pos, "meaning": meaning_json(groups),
            "is_common": _feat(entry, "vocabularyLevel") == "초급",
            "reading_latin": build_latin_key("kor", headword),
        }
