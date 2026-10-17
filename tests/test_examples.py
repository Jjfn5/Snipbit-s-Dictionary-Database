"""Offline end-to-end test of the examples pipeline with tiny fixture files."""
import bz2
import io
import json
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_tatoeba_ja as tat  # noqa: E402
import gemini_fill_ja as gem  # noqa: E402


def _bz2(path: Path, text: str):
    path.write_bytes(bz2.compress(text.encode("utf-8")))


def make_fixtures(d: Path):
    _bz2(d / "jpn_sentences.tsv.bz2",
         "100\tjpn\t私は毎日りんごを食べる。\n101\tjpn\t猫がいる。\n102\tjpn\t\n")
    _bz2(d / "jpn-eng_links.tsv.bz2", "100\t900\n100\t901\n101\t902\n")
    _bz2(d / "eng_sentences.tsv.bz2",
         "900\teng\tI eat an apple every day.\n901\teng\tSecond translation.\n902\teng\tThere is a cat.\n")
    data = "100\t1\t私[1] は[2] 毎日[3] 林檎{りんご}[4] を 食べる[5]{食べる}\n101\t2\t猫[1] が いる[2]{いる}\n".encode()
    with tarfile.open(d / "jpn_indices.tar.bz2", "w:bz2") as tar:
        info = tarfile.TarInfo("jpn_indices.csv")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


def test_headword_from_token():
    assert tat.headword_from_token("食べる[たべる]{食べた}") == "食べる"
    assert tat.headword_from_token("abc") is None
    assert tat.headword_from_token("{x}") is None


def test_tatoeba_build():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        make_fixtures(d)
        files = {k: d / u.rsplit("/", 1)[1] for k, u in tat.FILES.items()}
        out = d / "t.db"
        n_sent, n_words = tat.build(files, out)
        assert n_sent == 2  # blank-text row skipped
        con = sqlite3.connect(out)
        assert con.execute("SELECT translation FROM sentences WHERE id=100").fetchone()[0] == \
            "I eat an apple every day."  # first link wins
        words = {r[0] for r in con.execute("SELECT word FROM sentence_words WHERE sentence_id=100")}
        assert "食べる" in words and "私" in words
        assert con.execute("PRAGMA user_version").fetchone()[0] == 1


def test_validate():
    e = {"headword": "食べる", "pos": "v1, vt"}
    ok = {"sentence": "朝ご飯を食べた。", "surface": "食べた", "translation": "I ate breakfast."}
    assert gem.validate(e, ok) is None
    assert gem.validate(e, {**ok, "sentence": "朝ご飯を飲んだ。"}) is not None  # surface missing
    assert gem.validate(e, {**ok, "surface": "飲んだ", "sentence": "朝ご飯を飲んだ。"}) is not None  # wrong word
    assert gem.validate(e, {**ok, "sentence": "朝ご飯を食べた（笑）。"}) is not None  # brackets
    assert gem.validate({"headword": "勉強", "pos": "n, vs"}, {
        "sentence": "毎日勉強します。", "surface": "勉強", "translation": "I study every day."}) is None
    assert gem.word_stem("勉強", "n, vs") == "勉強"  # 'vs' (suru-noun) must NOT strip
    assert gem.word_stem("高い", "adj-i") == "高"


def test_gemini_and_assemble(monkeypatch=None):
    import assemble_examples_ja as asm
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        make_fixtures(d)
        files = {k: d / u.rsplit("/", 1)[1] for k, u in tat.FILES.items()}
        tat.build(files, d / "t.db")

        # tiny jmdict.db: 食べる is covered by Tatoeba; 水 and 山 are not
        jm = sqlite3.connect(d / "jm.db")
        jm.execute("CREATE TABLE entries (source, headword, alt_headword, reading, part_of_speech, "
                   "meaning, is_common, reading_latin)")
        for hw, rd, pos, g in [("食べる", "たべる", "v1", "to eat"), ("水", "みず", "n", "water"),
                               ("山", "やま", "n", "mountain"), ("珍", "めずらしい", "adj-i", "rare")]:
            jm.execute("INSERT INTO entries VALUES ('jmdict',?,NULL,?,?,?,1,NULL)",
                       (hw, rd, pos, json.dumps([{"g": [g]}], ensure_ascii=False)))
        jm.commit()
        jm.close()

        calls = []

        def fake(prompt):
            calls.append(prompt)
            assert "食べる" not in prompt  # already covered, must not be requested
            return [
                {"index": 1, "sentence": "水を飲みます。", "surface": "水", "translation": "I drink water."},
                {"index": 2, "sentence": "山に登る。", "surface": "山", "translation": "I climb the mountain."},
                {"index": 3, "sentence": "これは何ですか。", "surface": "珍", "translation": "What is this?"},  # fails stem
            ]

        args = SimpleNamespace(jmdict=d / "jm.db", tatoeba=d / "t.db", store=d / "g.jsonl",
                               state=d / "s.json", max_requests=5, batch_size=25, sleep=0, model="test")
        args.store.write_text("")
        res = gem.run(args, fake)
        assert res["accepted"] == 2 and res["rejected"] == 1 and len(calls) == 1
        state = json.loads(args.state.read_text())
        assert state["attempts"] == {"珍": 1}

        # second run: nothing new for accepted words; 珍 retried, still bad
        gem.run(args, fake)
        assert len(args.store.read_text().splitlines()) == 2

        sys.argv = ["assemble", "--tatoeba", str(d / "t.db"), "--gemini", str(args.store),
                    "--out-dir", str(d / "out")]
        assert asm.main() == 0
        con = sqlite3.connect(d / "out" / "jmdict_examples.db")
        assert con.execute("SELECT COUNT(*) FROM sentences").fetchone()[0] == 4
        assert con.execute("SELECT text FROM sentences WHERE id=?", (asm.GEMINI_ID_BASE + 1,)).fetchone()
        assert con.execute("SELECT sentence_id FROM sentence_words WHERE word='水'").fetchone()[0] == asm.GEMINI_ID_BASE + 1
        man = json.loads((d / "out" / "manifest.json").read_text())
        assert man["rowCount"] == 4 and man["geminiCount"] == 2 and len(man["dbSha256"]) == 64

        # unchanged inputs + previous manifest -> skip
        sys.argv += ["--prev-manifest", str(d / "out" / "manifest.json")]
        assert asm.main() == 0


if __name__ == "__main__":
    test_headword_from_token()
    test_tatoeba_build()
    test_validate()
    test_gemini_and_assemble()
    print("ok")
