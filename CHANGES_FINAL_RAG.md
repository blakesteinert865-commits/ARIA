# Aria final RAG consolidation (October 2026)

This release consolidates the most recent one-folder Aria application, broad title catalogue, improved question parsing, deterministic historical timeline lookups, live-ZIM diagnostics and unfinished-response cleaner.

## Improvements since the general-title release

- Added generalized subject parsing for indirect questions, including `Explain what an atom is in detail`, date-related predicates, people, names, and common article-topic relationships.
- Restored topic-phrase resolution for `How stars form` and other variants.
- Corrected Python `libzim` suggestion result compatibility (`getResults` versus `get_results`).
- Preserved conservative title matching and fallback rather than accepting a vaguely related page.
- Kept one-shot historical officeholder table extraction with full term dates (including transition years). Historical questions remain a regression test, not the scope of the whole retrieval system.
- Versioned the SQLite passage cache key so improvements do not silently reuse outdated reference passages.
- Added opt-in `ARIA_RAG_DEBUG=1` timing and article information on the bridge's standard error.
- Added `rag_audit.py`, which records retrieval evidence, timings and the matched article in JSONL for upcoming fine-tuning evaluation.
- Integrated the response-cleaner module **into `crutch.py`**. You do not have to run its old installer after extracting this release.
- Added new regression tests and expanded `check_wiki.py` on-device checks.

## Not changed

The order is still original user message → Crutch → Hassil → Wikipedia RAG (if eligible) → existing Qwen model. No new model, cloud API, embeddings database or full Wikipedia extraction. The GUI and artwork have not been restyled. The server still uses non-streaming responses; streaming is a separate feature, not part of this RAG-only stabilization release.

## Important limitations

- An offline ZIM is a historical snapshot and cannot answer current questions reliably.
- Title indexes retrieve relevant **sources**, not answers to every arbitrary relationship question. Some complex questions need later evidence extractors; no retriever can guarantee correctness for all possible prompts.
- `build_title_index.py` uses a private binding method to read ZIM title metadata. This worked with test doubles and matches documented libzim internals, but the entire real-title build **must still be verified on the user's Linux installation**. Without it, Aria continues to use native ZIM lookups.
- Local end-to-end model responses and the 1.09 GB archive cannot be verified in the packaging sandbox (which lacks `libzim` runtime). The pre-release real-ZIM check from the user verified live extraction before these changes.
- Crutch or Hassil may answer certain questions before RAG is consulted, by design.

## Recommended smoke tests on CachyOS / Fish

```fish
cd ~/ARIA
.venv/bin/python -m unittest discover -v
.venv/bin/python crutch.py --test
.venv/bin/python check_wiki.py
.venv/bin/python rag_audit.py
# Optional one-time title metadata catalogue if not built already:
.venv/bin/python build_title_index.py
```

Keep `data/wiki-eng-simple.zim`, `data/wiki_cache.sqlite3`, `data/wiki_titles.sqlite3`, `.venv/`, and your personal data when updating. If the ZIM changes, rebuild `wiki_titles.sqlite3` once.
