import json, os, sqlite3, tempfile, unittest
from build.parsers import parse_jmdict, parse_ccedict, parse_kaikki, parse_krdict
from build.schema import open_db, insert_rows
from build.transliteration import build_latin_key as k


def tmp(name, text):
    p = os.path.join(tempfile.mkdtemp(), name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


class Translit(unittest.TestCase):
    def test_kana(self):
        self.assertEqual(k("jmdict", "食べる", "たべる"), "taberu")
        self.assertEqual(k("jmdict", "x", "がっこう"), "gakkou")
        self.assertEqual(k("jmdict", "x", "きょう"), "kyou")
        self.assertEqual(k("jmdict", "x", "コーヒー"), "koohii")
        self.assertEqual(k("jmdict", "x", "ファイル"), "fairu")
        self.assertEqual(k("jmdict", "x", "ちょっと"), "chotto")

    def test_others(self):
        self.assertEqual(k("ccedict", "你好", "ni3 hao3"), "nihao")
        self.assertEqual(k("kor", "한국"), "hangug")
        self.assertEqual(k("spa", "Ñandú"), "nandu")
        self.assertEqual(k("fra", "Café"), "cafe")
        self.assertIsNone(k("jmdict", "漢字", None))


class Parsers(unittest.TestCase):
    def test_jmdict(self):
        data = {"words": [
            {"kanji": [{"text": "食べる", "common": True}, {"text": "喰べる"}],
             "kana": [{"text": "たべる", "common": True}],
             "sense": [{"partOfSpeech": ["v1", "vt"], "field": [], "misc": [], "info": [],
                        "gloss": [{"lang": "eng", "text": "to eat"}, {"lang": "ger", "text": "essen"}]},
                       {"partOfSpeech": [], "gloss": [{"lang": "fre", "text": "manger"}]}]},
            {"kanji": [], "kana": [{"text": "あ"}], "sense": [{"gloss": []}]}]}
        rows = list(parse_jmdict(tmp("j.json", json.dumps(data))))
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r["headword"], r["alt_headword"], r["reading"], r["reading_latin"]),
                         ("食べる", "喰べる", "たべる", "taberu"))
        self.assertTrue(r["is_common"])
        self.assertEqual(r["part_of_speech"], "v1, vt")
        self.assertEqual(json.loads(r["meaning"]), [{"g": ["to eat"], "p": "v1, vt"}])

    def test_ccedict_v1_v2(self):
        txt = ("# comment\n"
               "你好 你好 [ni3 hao3] /hello/hi/\n"
               "漢字 汉字 [[han4zi4]] /Chinese character/\n"
               "bad line\n")
        rows = list(parse_ccedict(tmp("c.txt", txt)))
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[0]["alt_headword"])
        self.assertEqual(rows[0]["reading_latin"], "nihao")
        self.assertEqual((rows[1]["headword"], rows[1]["alt_headword"]), ("汉字", "漢字"))
        self.assertEqual(rows[1]["reading"], "han4 zi4")
        self.assertEqual(len(json.loads(rows[0]["meaning"])), 2)

    def test_kaikki(self):
        lines = [json.dumps({"word": "café", "pos": "noun",
                             "senses": [{"glosses": ["drink", "coffee"]}, {"glosses": []}]}),
                 "not json", json.dumps({"word": "", "senses": []})]
        rows = list(parse_kaikki(tmp("k.jsonl", "\n".join(lines)), "spa"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0]["meaning"]), [{"g": ["coffee"]}])
        self.assertEqual(rows[0]["reading_latin"], "cafe")

    def test_krdict(self):
        xml = """<Lexicon><LexicalEntry>
          <feat att="partOfSpeech" val="명사"/><feat att="origin" val="學校"/><feat att="vocabularyLevel" val="초급"/>
          <Lemma><feat att="writtenForm" val="학교"/></Lemma>
          <WordForm><feat att="type" val="발음"/><feat att="pronunciation" val="학꾜"/></WordForm>
          <Sense><Equivalent><feat att="language" val="영어"/><feat att="lemma" val="school"/>
            <feat att="definition" val="a place for study"/></Equivalent></Sense>
        </LexicalEntry></Lexicon>"""
        rows = list(parse_krdict(tmp("kr.xml", xml)))
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r["headword"], r["reading"], r["alt_headword"], r["is_common"]),
                         ("학교", "학꾜", "學校", True))
        self.assertEqual(json.loads(r["meaning"]), [{"g": ["school"], "i": "a place for study"}])


class Db(unittest.TestCase):
    def test_insert(self):
        db = open_db(":memory:")
        insert_rows(db, [{"source": "x", "headword": "a", "meaning": "[]"}])
        self.assertEqual(db.execute("SELECT COUNT(*) FROM entries").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
