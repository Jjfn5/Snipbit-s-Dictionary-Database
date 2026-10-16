# Snipbit's Dictionary Database

GitHub Actions builds the dictionaries so phones don't have to parse them.

## Phase 1 (current): JMdict only
`scripts/build_jmdict.py` fetches the latest jmdict-simplified release, ports
`JmdictParser.addWordRow`, and publishes to the rolling release tag `jmdict`:

- `jmdict.db.gz` -- SQLite, one table `entries` (no id, no indexes)
- `manifest.json` -- `schemaVersion`, `upstreamTag`, `rowCount`, `dbUrl`, `dbSha256`, ...

Stable URL for the app:
`https://github.com/Jjfn5/Snipbit-s-Dictionary-Database/releases/download/jmdict/manifest.json`

Run locally: `python scripts/build_jmdict.py --input tests/sample_jmdict.json --min-rows 0`

## App side (copy into the app's own db)
Columns match `DictionaryDb.ensureSchema`. `ATTACH` cannot run inside a
transaction, so attach first:

```sql
ATTACH DATABASE '<path>/jmdict.db' AS prebuilt;
BEGIN;
DELETE FROM entries WHERE source = 'jmdict';
INSERT INTO entries
  (source, headword, alt_headword, reading, part_of_speech, meaning, is_common, reading_latin)
SELECT source, headword, alt_headword, reading, part_of_speech, meaning, is_common, reading_latin
FROM prebuilt.entries;
COMMIT;
DETACH DATABASE prebuilt;
```
Check `schemaVersion` == 1 and the gz's SHA-256 before importing.

## Keeping in sync
`scripts/translit.py` is a third hand-maintained copy of the romaji logic
(Dart, Kotlin, Python). `tests/test_translit.py` pins known outputs.
