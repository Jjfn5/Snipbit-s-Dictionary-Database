"""Mirror of dictionary_catalog.dart (definition sources only)."""
KRDICT_BASE = "https://raw.githubusercontent.com/spellcheck-ko/korean-dict-nikl-krdict/master/"
KRDICT_FILES = [f"{n}.xml" for n in (5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 51947)]

SOURCES = {
    "jmdict": {"title": "JMdict (Japanese–English)",
               "release_api": "https://api.github.com/repos/scriptin/jmdict-simplified/releases/latest",
               "asset_pattern": r"^jmdict-eng-\d.*\.json\.zip$", "archive": "zip"},
    "ccedict": {"title": "CC-CEDICT (Chinese–English)",
                "urls": ["https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz"],
                "archive": "gzip"},
    "kor": {"title": "KrDict (Korean–English)", "urls": [KRDICT_BASE + f for f in KRDICT_FILES], "archive": "none"},
    "spa": {"title": "Kaikki (Spanish–English)",
            "urls": ["https://kaikki.org/dictionary/Spanish/kaikki.org-dictionary-Spanish.jsonl"], "archive": "none"},
    "fra": {"title": "Kaikki (French–English)",
            "urls": ["https://kaikki.org/dictionary/French/kaikki.org-dictionary-French.jsonl"], "archive": "none"},
}
