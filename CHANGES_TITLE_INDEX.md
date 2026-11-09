# Aria: General Wikipedia Title Index Update

This revision extends the existing one-folder Aria application and **does not**
replace its GUI or Crutch-first routing architecture.

- Optional offline title catalogue (`aria_title_index.py`), stored as
  `data/wiki_titles.sqlite3` after a one-time build.
- One-time build launcher: `.venv/bin/python build_title_index.py`.
- Normalised exact-title and conservative FTS5 title search with source ZIM
  signature validation. The original 1.09 GB ZIM remains untouched.
- Broader query-subject extraction, including bare proper names, author and
  birth queries, with fallback to native ZIM suggestion and full-text search.
- Existing historical-officeholder timeline extraction retained as a distinct
  evidence check, not a hard-coded prime minister answer.
- Fixed `libzim.Archive` context-manager compatibility in the RAG backend and
  archive validation script.
- Additional automated tests for all-title indexing, internal listing fallback,
  stale source detection, bare names and historical-officeholder parsing.

## Limitations

The one-time catalogue build requires the installed `libzim` binding to expose
an internal accessor (`_get_entry_by_id`). If unavailable, the builder exits
with an explanation and normal ZIM retrieval remains intact. The build is not
run automatically on startup. No 400,061-title scan occurs per message.

The original ZIM and local Qwen model server are not included. Full live
index-building, article retrieval, and GUI streaming are not validated here.
Streaming support is a separate planned change, not part of this revision.
