import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from translit import build_latin_key as k  # noqa: E402


def test_japanese():
    assert k("jmdict", "食べる", "たべる") == "taberu"
    assert k("jmdict", "x", "コーヒー") == "koohii"
    assert k("jmdict", "x", "がっこう") == "gakkou"
    assert k("jmdict", "x", "にっぽん") == "nippon"
    assert k("jmdict", "x", "ファイル") == "fairu"
    assert k("jmdict", "x", "しゃしん") == "shashin"
    assert k("jmdict", "漢字", None) is None  # no kana at all -> null, like the app


def test_other_sources():
    assert k("ccedict", "你好", "ni3 hao3") == "nihao"
    assert k("kor", "한국어") == "hangugeo"
    assert k("spa", "Ñandú") == "nandu"
    assert k("fra", "café") == "cafe"
    assert k("xyz", "a") is None


if __name__ == "__main__":
    test_japanese()
    test_other_sources()
    print("ok")
