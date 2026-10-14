"""SQLite schema -- identical to DictionaryDb.ensureSchema (entries table) in the app."""
import json
import sqlite3

ENTRIES_SQL = """
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    headword TEXT NOT NULL,
    alt_headword TEXT,
    reading TEXT,
    part_of_speech TEXT,
    meaning TEXT NOT NULL,
    is_common INTEGER NOT NULL DEFAULT 0,
    reading_latin TEXT
);
CREATE INDEX IF NOT EXISTS idx_entries_headword ON entries(headword);
CREATE INDEX IF NOT EXISTS idx_entries_alt_headword ON entries(alt_headword);
CREATE INDEX IF NOT EXISTS idx_entries_reading ON entries(reading);
CREATE INDEX IF NOT EXISTS idx_entries_reading_latin ON entries(reading_latin);
CREATE INDEX IF NOT EXISTS idx_entries_source ON entries(source);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def sense_group(glosses, pos=None, field=None, misc=None, info=None):
    """Mirrors SenseGroupJson / SenseGroup.toJson(): keys g,p,f,m,i."""
    g = {"g": list(glosses)}
    if pos:
        g["p"] = pos
    if field:
        g["f"] = field
    if misc:
        g["m"] = misc
    if info:
        g["i"] = info
    return g


def meaning_json(groups):
    return json.dumps(groups, ensure_ascii=False, separators=(",", ":"))


def open_db(path):
    db = sqlite3.connect(path)
    db.executescript(ENTRIES_SQL)
    return db


INSERT_SQL = (
    "INSERT INTO entries (source, headword, alt_headword, reading, part_of_speech,"
    " meaning, is_common, reading_latin) VALUES (?,?,?,?,?,?,?,?)"
)


def insert_rows(db, rows):
    """rows: iterable of dicts with the entries columns (is_common bool)."""
    db.executemany(
        INSERT_SQL,
        (
            (r["source"], r["headword"], r.get("alt_headword"), r.get("reading"),
             r.get("part_of_speech"), r["meaning"], 1 if r.get("is_common") else 0,
             r.get("reading_latin"))
            for r in rows
        ),
    )
    db.commit()
