"""Offline tests for the ccedict / krdict / kaikki entry builders (dictionaries/<lang>/) (no network, tiny fixtures)."""
import gzip
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "dictionaries" / "zh"))
sys.path.insert(0, str(ROOT / "dictionaries" / "ko"))
import ccedict as cc  # noqa: E402
import kaikki as kk  # noqa: E402  (shared parser in scripts/)
import krdict as kr  # noqa: E402
import dictlib  # noqa: E402

COLS = ["source", "headword", "alt_headword", "reading", "part_of_speech",
        "meaning", "is_common", "reading_latin"]


def rows_of(db: Path):
    con = sqlite3.connect(db)
    assert con.execute("PRAGMA user_version").fetchone()[0] == dictlib.SCHEMA_VERSION
    names = [r[1] for r in con.execute("PRAGMA table_info(entries)")]
    assert names == COLS, names   # exactly what DictionaryDb.replaceEntriesFromPrebuilt selects
    return con.execute("SELECT * FROM entries").fetchall()


def test_ccedict():
    text = "\n".join([
        "# comment",
        "傳統 传统 [chuan2 tong3] /tradition/traditional/",
        "你好 你好 [ni3 hao3] /hello/",
        "二次方程 二次方程 [[er4ci4fang1cheng2]] /quadratic equation/",
        "bad line",
        "空 空 [kong1] //",
    ])
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src = d / "c.txt.gz"
        src.write_bytes(gzip.compress(text.encode()))
        out = d / "out"
        assert cc.main(["--input", str(src), "--out-dir", str(out), "--min-rows", "0"]) == 0
        rows = rows_of(out / "ccedict.db")
        assert len(rows) == 3
        trad = rows[0]
        assert trad[1:5] == ("传统", "傳統", "chuan2 tong3", None)
        assert json.loads(trad[5]) == [{"g": ["tradition"]}, {"g": ["traditional"]}]
        assert trad[7] == "chuantong"
        assert rows[1][2] is None                       # same script -> no alt
        assert rows[2][3] == "er4 ci4 fang1 cheng2"     # v2 spacing normalised
        m = json.loads((out / "manifest.json").read_text())
        assert m["source"] == "ccedict" and m["rowCount"] == 3
        assert m["dbSha256"] == dictlib.sha256_file(out / "ccedict.db.gz")


KR_XML = """<?xml version="1.0" encoding="UTF-8"?>
<LexicalResource><Lexicon>
<LexicalEntry>
 <feat att="partOfSpeech" val="명사"/><feat att="origin" val="學校"/><feat att="vocabularyLevel" val="초급"/>
 <Lemma><feat att="writtenForm" val="학교"/></Lemma>
 <WordForm><feat att="type" val="발음"/><feat att="pronunciation" val="학꾜"/></WordForm>
 <Sense><Equivalent><feat att="language" val="영어"/><feat att="lemma" val="school"/>
   <feat att="definition" val="a place for education"/></Equivalent></Sense>
</LexicalEntry>
<LexicalEntry>
 <Lemma><feat att="writtenForm" val="없음"/></Lemma>
 <Sense><Equivalent><feat att="language" val="프랑스어"/><feat att="lemma" val="rien"/></Equivalent></Sense>
</LexicalEntry>
<LexicalEntry>
 <Lemma><feat att="writtenForm" val="물"/></Lemma>
 <Sense><Equivalent><feat att="language" val="영어"/><feat att="lemma" val=""/>
   <feat att="definition" val="a clear liquid"/></Equivalent></Sense>
</LexicalEntry>
</Lexicon></LexicalResource>"""


def test_krdict():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src = d / "k.xml"
        src.write_text(KR_XML, encoding="utf-8")
        out = d / "out"
        assert kr.main(["--input", str(src), "--out-dir", str(out), "--min-rows", "0"]) == 0
        rows = rows_of(out / "kor.db")
        assert len(rows) == 2   # the French-only entry is dropped
        r = rows[0]
        assert r[1:5] == ("학교", "學校", "학꾜", "명사") and r[6] == 1
        assert json.loads(r[5]) == [{"g": ["school"], "i": "a place for education"}]
        assert r[7] == "haggyo"   # app's simplified romaja (final ㄱ -> g)
        assert json.loads(rows[1][5]) == [{"g": ["a clear liquid"]}]  # definition-only sense


def test_krdict_malformed_xml():
    """Bare &, stray < and ", and control chars must not abort the build."""
    bad = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<LexicalResource><Lexicon>\n'
        '<LexicalEntry>\n'
        ' <Lemma><feat att="writtenForm" val="굴"/></Lemma>\n'
        ' <Sense><Equivalent><feat att="language" val="영어"/>\n'
        '   <feat att="lemma" val="oyster"/>\n'
        '   <feat att="definition" val="salt & pepper < 5" or "x"/></Equivalent></Sense>\n'
        '</LexicalEntry>\n'
        '<LexicalEntry>\n'
        ' <Lemma><feat att="writtenForm" val="물"/></Lemma>\n'
        ' <Sense><Equivalent><feat att="language" val="영어"/>\n'
        '   <feat att="lemma" val="water\x0b"/></Equivalent></Sense>\n'
        '</LexicalEntry>\n'
        '</Lexicon></LexicalResource>'
    )
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src = d / "k.xml"
        src.write_text(bad, encoding="utf-8")
        out = d / "out"
        assert kr.main(["--input", str(src), "--out-dir", str(out), "--min-rows", "0"]) == 0
        rows = rows_of(out / "kor.db")
        assert [r[1] for r in rows] == ["굴", "물"]
        assert json.loads(rows[0][5])[0]["i"] == 'salt & pepper < 5" or "x'


def test_kaikki():
    lines = [
        json.dumps({"word": "Café", "pos": "noun",
                    "senses": [{"glosses": ["drink", "coffee"]}, {"glosses": []}, {"x": 1}]}),
        "not json",
        json.dumps({"word": "", "senses": [{"glosses": ["x"]}]}),
        json.dumps({"word": "vacío", "pos": "adj", "senses": [{"glosses": ["empty"]}]}),
        json.dumps({"word": "nada", "senses": []}),
    ]
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src = d / "k.jsonl"
        src.write_text("\n".join(lines), encoding="utf-8")
        out = d / "out"
        assert kk.main(["--lang", "spa", "--input", str(src), "--out-dir", str(out),
                        "--min-rows", "0"]) == 0
        rows = rows_of(out / "spa.db")
        assert [r[1] for r in rows] == ["Café", "vacío"]
        assert json.loads(rows[0][5]) == [{"g": ["coffee"]}]   # leaf of the breadcrumb only
        assert rows[0][7] == "cafe" and rows[1][7] == "vacio"
        assert (out / "spa.db.gz").exists()


def test_unchanged_upstream_skips():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src = d / "c.txt"
        src.write_text("你好 你好 [ni3 hao3] /hello/\n", encoding="utf-8")
        out = d / "out"
        assert cc.main(["--input", str(src), "--out-dir", str(out), "--min-rows", "0"]) == 0
        before = (out / "manifest.json").read_text()
        # --input has no cheap stamp but the content hash still matches -> no rebuild
        prev = d / "prev.json"
        prev.write_text(before)
        assert cc.main(["--input", str(src), "--out-dir", str(out), "--min-rows", "0",
                        "--prev-manifest", str(prev)]) == 0
        assert (out / "manifest.json").read_text() == before


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
