# Snipbit dictionary database

Builds prebuilt SQLite dictionaries (same `entries` schema as the app) for jmdict, ccedict, kor, spa, fra.

    python -m build.build_dictionary jmdict ccedict kor spa fra --out dist
    python -m unittest discover -s tests -t .

Python 3.11+, standard library only. Each `dist/<source>.db` also has a `meta` table
(source, entry_count, built_at, upstream). The build fails if a source yields 0 entries.

Planned: example-sentence cache (Gemini), release workflow + manifest.json, LICENSES.md.
