"""Offline tests for the examples pipeline: Tatoeba build, Gemini fallback and
assembly, for every language, with tiny fixture files."""
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
import build_examples as ex  # noqa: E402
import gemini_fill as gem  # noqa: E402
from langs import LANGS, tokenize_space  # noqa: E402

JA, ZH, KO, ES, FR = (LANGS[c] for c in ("ja", "zh", "ko", "es", "fr"))


def _bz2(path: Path, text: str):
    path.write_bytes(bz2.compress(text.encode("utf-8")))


def ja_fixtures(d: Path) -> dict:
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
    return {k: d / u.rsplit("/", 1)[1] for k, u in ex.urls_for(JA).items()}


def es_fixtures(d: Path) -> dict:
    _bz2(d / "spa_sentences.tsv.bz2", "1\tspa\tMe gusta el agua.\n2\tspa\tHola, mundo.\n3\tspa\t\n")
    _bz2(d / "spa-eng_links.tsv.bz2", "1\t10\n1\t11\n2\t12\n")
    _bz2(d / "eng_sentences.tsv.bz2", "10\teng\tI like water.\n11\teng\tSecond.\n12\teng\tHello, world.\n")
    return {k: d / u.rsplit("/", 1)[1] for k, u in ex.urls_for(ES).items()}


def test_urls():
    assert "indices" in ex.urls_for(JA) and "indices" not in ex.urls_for(ES)
    assert ex.urls_for(ZH)["sentences"].endswith("cmn/cmn_sentences.tsv.bz2")


def test_headword_from_token():
    assert ex.headword_from_token("食べる[たべる]{食べた}") == "食べる"
    assert ex.headword_from_token("abc") is None
    assert ex.headword_from_token("{x}") is None


def test_tokenize_space():
    assert tokenize_space("L'eau, c'est bon !") == ["l'eau", "l", "eau", "c'est", "c", "est", "bon"]
    assert tokenize_space("¿Dónde está?") == ["dónde", "está"]
    assert tokenize_space("나는 학교에 갑니다.") == ["나는", "학교에", "갑니다"]


def test_tatoeba_ja():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        out = d / "t.db"
        n_sent, n_words = ex.build(ja_fixtures(d), out, JA, None)
        assert n_sent == 2  # blank-text row skipped
        con = sqlite3.connect(out)
        assert con.execute("SELECT translation FROM sentences WHERE id=100").fetchone()[0] == \
            "I eat an apple every day."  # first link wins
        words = {r[0] for r in con.execute("SELECT word FROM sentence_words WHERE sentence_id=100")}
        assert "食べる" in words and "私" in words
        assert con.execute("PRAGMA user_version").fetchone()[0] == 1


def test_tatoeba_es():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        out = d / "e.db"
        n_s, _ = ex.build(es_fixtures(d), out, ES, tokenize_space)
        assert n_s == 2
        con = sqlite3.connect(out)
        assert con.execute("SELECT translation FROM sentences WHERE id=1").fetchone()[0] == "I like water."
        words = {r[0] for r in con.execute("SELECT word FROM sentence_words WHERE sentence_id=1")}
        assert words == {"me", "gusta", "el", "agua"}
        assert con.execute("SELECT DISTINCT source FROM sentences").fetchall() == [("spa",)]


def test_validate_ja():
    e = {"headword": "食べる", "pos": "v1, vt"}
    ok = {"sentence": "朝ご飯を食べた。", "surface": "食べた", "translation": "I ate breakfast."}
    assert gem.validate(JA, e, ok) is None
    assert gem.validate(JA, e, {**ok, "sentence": "朝ご飯を飲んだ。"}) is not None  # surface missing
    assert gem.validate(JA, e, {**ok, "surface": "飲んだ", "sentence": "朝ご飯を飲んだ。"}) is not None  # wrong word
    assert gem.validate(JA, e, {**ok, "sentence": "朝ご飯を食べた（笑）。"}) is not None  # brackets
    assert gem.validate(JA, {"headword": "勉強", "pos": "n, vs"}, {
        "sentence": "毎日勉強します。", "surface": "勉強", "translation": "I study every day."}) is None
    assert gem.word_stem("勉強", "n, vs") == "勉強"  # 'vs' (suru-noun) must NOT strip
    assert gem.word_stem("高い", "adj-i") == "高"


def test_validate_other_languages():
    tr = "A translation."
    # Chinese
    z = {"headword": "学校", "pos": ""}
    assert gem.validate(ZH, z, {"sentence": "我每天去学校。", "surface": "学校", "translation": tr}) is None
    assert gem.validate(ZH, z, {"sentence": "私は学校へ行く。", "surface": "学校", "translation": tr}) is not None
    assert gem.validate(ZH, z, {"sentence": "我每天去大学。", "surface": "大学", "translation": tr}) is not None
    # Korean: stem of a 다-verb survives conjugation
    k = {"headword": "먹다", "pos": "동사"}
    assert gem.validate(KO, k, {"sentence": "저는 밥을 먹어요.", "surface": "먹어요", "translation": tr}) is None
    assert gem.validate(KO, k, {"sentence": "I eat rice every day.", "surface": "eat", "translation": tr}) is not None
    # Spanish: verbs trust `surface` (irregulars), nouns need their stem, wrong script rejected
    v = {"headword": "ir", "pos": "verb"}
    assert gem.validate(ES, v, {"sentence": "Mañana voy al mercado.", "surface": "voy", "translation": tr}) is None
    n = {"headword": "casa", "pos": "noun"}
    assert gem.validate(ES, n, {"sentence": "Mi casa es muy grande.", "surface": "casa", "translation": tr}) is None
    assert gem.validate(ES, n, {"sentence": "Las casas son nuevas.", "surface": "casas", "translation": tr}) is None
    assert gem.validate(ES, n, {"sentence": "El perro come mucho.", "surface": "perro", "translation": tr}) is not None
    assert gem.validate(ES, n, {"sentence": "我的家很大，在这里。", "surface": "casa", "translation": tr}) is not None
    # French
    f = {"headword": "pomme", "pos": "noun"}
    assert gem.validate(FR, f, {"sentence": "Je mange une pomme rouge.", "surface": "pomme", "translation": tr}) is None
    assert gem.validate(FR, f, {"sentence": "Court.", "surface": "Court", "translation": tr}) is not None  # too short


def test_candidate_filter():
    from langs import worth_generating as w
    assert not w(ES, "comí", "verb", "first-person singular preterite indicative of comer")
    assert not w(ES, "buenos días", "phrase", "good morning")
    assert not w(ES, "Madrid", "name", "capital of Spain")
    assert w(ES, "casa", "noun", "house")
    assert not w(ZH, "李", "", "surname Li")
    assert w(ZH, "学校", "", "school")


def _entries_db(path: Path, source: str, rows):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE entries (source, headword, alt_headword, reading, part_of_speech, "
                "meaning, is_common, reading_latin)")
    for hw, rd, pos, g, common in rows:
        con.execute("INSERT INTO entries VALUES (?,?,NULL,?,?,?,?,NULL)",
                    (source, hw, rd, pos, json.dumps([{"g": [g]}], ensure_ascii=False), common))
    con.commit()
    con.close()


def test_gemini_and_assemble_ja():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        ex.build(ja_fixtures(d), d / "t.db", JA, None)
        # 食べる is covered by Tatoeba; 水 and 山 are not
        _entries_db(d / "jm.db", "jmdict", [
            ("食べる", "たべる", "v1", "to eat", 1), ("水", "みず", "n", "water", 1),
            ("山", "やま", "n", "mountain", 1), ("珍", "めずらしい", "adj-i", "rare", 1)])
        calls = []

        def fake(prompt):
            calls.append(prompt)
            assert "食べる" not in prompt  # already covered, must not be requested
            assert "Japanese" in prompt
            return [
                {"index": 1, "sentence": "水を飲みます。", "surface": "水", "translation": "I drink water."},
                {"index": 2, "sentence": "山に登る。", "surface": "山", "translation": "I climb the mountain."},
                {"index": 3, "sentence": "これは何ですか。", "surface": "珍", "translation": "What is this?"},
            ], "test"

        args = SimpleNamespace(lang="ja", dictionary=d / "jm.db", tatoeba=d / "t.db",
                               store=d / "g.jsonl", state=d / "s.json",
                               max_requests=5, batch_size=25, sleep=0)
        args.store.write_text("")
        res = gem.run(args, fake)
        assert res["accepted"] == 2 and res["rejected"] == 1 and len(calls) == 1
        assert json.loads(args.state.read_text())["attempts"] == {"珍": 1}
        gem.run(args, fake)   # accepted words are not repeated; 珍 retried, still bad
        assert len(args.store.read_text().splitlines()) == 2

        argv = ["assemble", "--lang", "ja", "--tatoeba", str(d / "t.db"),
                "--gemini", str(args.store), "--out-dir", str(d / "out")]
        assert ex.main(argv) == 0
        con = sqlite3.connect(d / "out" / "jmdict_examples.db")
        assert con.execute("SELECT COUNT(*) FROM sentences").fetchone()[0] == 4
        assert con.execute("SELECT text FROM sentences WHERE id=?", (ex.GEMINI_ID_BASE + 1,)).fetchone()
        assert con.execute("SELECT sentence_id FROM sentence_words WHERE word='水'").fetchone()[0] == ex.GEMINI_ID_BASE + 1
        man = json.loads((d / "out" / "examples-manifest.json").read_text())
        assert man["rowCount"] == 4 and man["geminiCount"] == 2 and len(man["dbSha256"]) == 64
        assert man["dbUrl"].endswith("/releases/download/jmdict/jmdict_examples.db.gz")
        assert ex.main(argv + ["--prev-manifest", str(d / "out" / "examples-manifest.json")]) == 0  # skip


def test_gemini_and_assemble_es():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        ex.build(es_fixtures(d), d / "t.db", ES, tokenize_space)
        # Kaikki has no common flag: "agua" is covered (Tatoeba, via normalisation of "Agua");
        # a form-of entry and a phrase are filtered; "perro" and "gato" need Gemini
        _entries_db(d / "spa.db", "spa", [
            ("Agua", "", "noun", "water", 0), ("perro", "", "noun", "dog", 0),
            ("gato", "", "noun", "cat", 0), ("comí", "", "verb", "first-person singular preterite of comer", 0),
            ("por favor", "", "phrase", "please", 0)])

        def fake(prompt):
            assert "Spanish" in prompt and "Agua" not in prompt and "comí" not in prompt
            assert "por favor" not in prompt
            assert prompt.index("gato") < prompt.index("perro")   # shorter headword first
            return [
                {"index": 1, "sentence": "El gato duerme en el sofá.", "surface": "gato", "translation": "The cat sleeps on the sofa."},
                {"index": 2, "sentence": "Mi perro corre mucho.", "surface": "perro", "translation": "My dog runs a lot."},
            ], "test"

        args = SimpleNamespace(lang="es", dictionary=d / "spa.db", tatoeba=d / "t.db",
                               store=d / "g.jsonl", state=d / "s.json",
                               max_requests=5, batch_size=25, sleep=0)
        args.store.write_text("")
        res = gem.run(args, fake)
        assert res["accepted"] == 2 and res["rejected"] == 0

        assert ex.main(["assemble", "--lang", "es", "--tatoeba", str(d / "t.db"),
                        "--gemini", str(args.store), "--out-dir", str(d / "out")]) == 0
        con = sqlite3.connect(d / "out" / "spa_examples.db")
        # headword AND every other word of the Gemini sentence are indexed
        words = {r[0] for r in con.execute(
            "SELECT word FROM sentence_words WHERE sentence_id > ?", (ex.GEMINI_ID_BASE,))}
        assert {"gato", "perro", "sofá", "duerme", "corre"} <= words
        assert con.execute("SELECT DISTINCT source FROM sentences").fetchall() == [("spa",)]
        man = json.loads((d / "out" / "examples-manifest.json").read_text())
        assert man["source"] == "spa" and man["geminiCount"] == 2


def test_model_pool():
    clock = [0.0]
    pool = gem.ModelPool(["A", "B", "C"], now=lambda: clock[0])
    slept = []
    sleep = lambda s: (slept.append(s), clock.__setitem__(0, clock[0] + s))  # noqa: E731
    calls = []

    def fake_call(prompt, model):
        calls.append(model)
        if model == "A":
            raise gem.ModelGone("404 not found")
        if model == "B":
            raise gem.RateLimited("quota", None, per_day=True)
        return [{"index": 1}]

    items, used = gem.call_with_pool("p", pool, fake_call, sleep)
    assert used == "C" and calls == ["A", "B", "C"]
    calls.clear()
    gem.call_with_pool("p", pool, fake_call, sleep)
    assert calls == ["C"]  # A is gone, B is resting: straight to C

    # overloaded: retried twice with backoff, then succeeds on the same model
    pool = gem.ModelPool(["X", "Y"], now=lambda: clock[0])
    n = {"x": 0}

    def flaky(prompt, model):
        if model == "X":
            n["x"] += 1
            if n["x"] < 3:
                raise gem.Overloaded("503")
        return [{"index": 1}]

    slept.clear()
    assert gem.call_with_pool("p", pool, flaky, sleep)[1] == "X" and slept == [5, 15]

    # short per-minute limit: waits the server's delay once, then same model works
    pool = gem.ModelPool(["X", "Y"], now=lambda: clock[0])
    state = {"first": True}

    def rpm(prompt, model):
        if model == "X" and state["first"]:
            state["first"] = False
            raise gem.RateLimited("slow down", 20.0, per_day=False)
        return [{"index": 1}]

    slept.clear()
    assert gem.call_with_pool("p", pool, rpm, sleep)[1] == "X" and slept == [21.0]

    # everything dead/resting for long -> QuotaExhausted
    pool = gem.ModelPool(["X", "Y"], now=lambda: clock[0])

    def nope(prompt, model):
        raise gem.ModelGone("gone") if model == "X" else gem.RateLimited("q", None, True)

    try:
        gem.call_with_pool("p", pool, nope, sleep)
        raise AssertionError("should have raised")
    except gem.QuotaExhausted as e:
        assert "X" in str(e) and "Y" in str(e)

    # empty answers fall through to the next model, then give up quietly
    pool = gem.ModelPool(["X", "Y"], now=lambda: clock[0])
    seen = []
    items, used = gem.call_with_pool("p", pool, lambda pr, m: (seen.append(m), [])[1], sleep)
    assert items == [] and seen == ["X", "Y"]


def test_chain_building():
    assert gem.parse_models("a, b", "models/c", "a") == ["a", "b", "c"]
    avail = ["gemini-2.5-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite-tts",
             "gemini-3.5-flash", "gemini-2.5-pro", "gemma-4-31b-it", "gemini-3.1-flash-lite-preview"]
    chain = gem.build_chain(["gemini-2.5-flash-lite", "gemini-3.1-flash-lite"], avail)
    assert chain[0] == "gemini-3.1-flash-lite"           # unlisted name dropped, listed kept
    assert "gemini-3.5-flash-lite-tts" not in chain and "gemini-2.5-pro" not in chain
    assert "gemma-4-31b-it" not in chain
    assert chain[1] == "gemini-3.1-flash-lite-preview"   # extras: lite first
    assert gem.build_chain(["x"], None) == ["x"]         # lookup failed -> try configured as-is


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
