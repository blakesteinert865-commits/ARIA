# Aria: one-folder GUI, Crutch, Hassil and offline Wikipedia RAG

This is the **complete updated Aria application**, packaged to be extracted into:

```
/home/blake/ARIA/
├── app.py                   # Original PyQt6 GUI and artwork (design unchanged)
├── Arialogobase.png
├── AriaGlow.png
├── AriaAppIcon.png
├── crutch.py                # Deterministic tools and Qwen-model API client
├── aria_hassil_router.py    # Intent matching
├── aria_backend_bridge.py  # GUI → tools → Wikipedia → existing model
├── aria_wiki_rag.py         # Bounded offline Wikipedia retriever
├── aria_title_index.py     # Optional indexed whole-title catalogue
├── build_title_index.py    # One-time title-index builder
├── vendor/                  # Bundled Hassil
├── data/                    # Notes/lists/timetable, Wikipedia archive, RAG cache
│   ├── README.txt
│   └── wiki-eng-simple.zim  # YOU MOVE THIS HERE, not included in ZIP
├── .venv/                   # You create this in setup, not included in ZIP
├── run_aria.sh
├── requirements.txt
├── migrate_old_data.py     # Optional safe migration helper
├── check_wiki.py           # Run on your machine to check the real ZIM archive
└── test_*.py               # Headless regression tests
```

No application code refers to `~/Documents/Aria tools` anymore. The Crutch
storage default is now **`/home/blake/ARIA/data`**, not `~/.crutch`. The RAG
cache is **`/home/blake/ARIA/data/wiki_cache.sqlite3`**. All paths are derived
from the location of the source files, so the folder can be renamed or moved
later as a unit. Exceptions are explicit environment-variable overrides.

## 1. Install on CachyOS / Arch Linux

1. Create `/home/blake/ARIA`, then extract the ZIP contents directly into that directory. The ZIP has the application files at its root, not inside an extra `ARIA/` folder.
2. In a terminal:

   ```bash
   cd /home/blake/ARIA
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   ```

   If `venv` or `pip` is missing, install the corresponding Python packages
   using your distro package manager first. The previous separate Python
   environment at `~/Documents/Aria-data/.venv` is not needed.

3. Migrate your existing files if desired:

   ```bash
   .venv/bin/python migrate_old_data.py --copy-notes --move-zim
   ```

   This **copies** files from the old `~/.crutch` data folder without
   overwriting existing files, and **moves** the full archive from
   `/home/blake/Documents/Aria-data/wiki-eng-simple.zim` into
   `/home/blake/ARIA/data/wiki-eng-simple.zim`. The archive is not extracted,
   rebuilt or duplicated. The command is optional. You can also move the ZIM
   manually from its current location into `data/`, making sure its filename is
   `wiki-eng-simple.zim`. **Keep a backup before reorganising your files.**

4. Start your **existing local OpenAI-compatible model server**, e.g. LM
   Studio or llama-server, with your existing Qwen 0.8B model. The Aria GUI
   uses `http://localhost:1234/v1/chat/completions` by default. If needed:

   ```bash
   export CRUTCH_URL='http://localhost:8080/v1/chat/completions'
   export CRUTCH_MODEL='local-model'
   ```

   The model server and its model weights are **not bundled** and still need
   to be available. The one-folder layout concerns Aria's application,
   dependencies and local storage; it does not silently install or change
   your inference server.

5. Run:

   ```bash
   ./run_aria.sh
   ```

   If you prefer: `.venv/bin/python app.py`.

## 2. How Wikipedia RAG works

The exact flow is:

1. The GUI sends your original prompt unchanged to the backend bridge.
2. Crutch's deterministic tools run first, then bundled Hassil intent matching.
3. If a tool matches, Aria replies directly. **Wikipedia is not queried.**
4. Only unanswered, appropriate general-knowledge queries get a ZIM lookup.
5. `libzim` checks exact titles first, then the optional local title catalogue,
   built-in suggestions, and indexed full-text results. Article titles must
   match the inferred subject conservatively; unrelated results are rejected.
6. The selected article's HTML is cleaned of templates, infoboxes, citations,
   navigation and references. Nothing is bulk-extracted.
7. **Standard mode:** short introductory passage, at most ~760 characters,
   preserving Qwen's existing **120-output-token** limit.
8. **Detailed mode:** introduction and relevant sections, at most ~4,400
   characters (roughly within a 1,500-token context budget, depending on the
   tokenizer), preserving the existing **512-output-token** limit.
9. The reference is attached to the existing **system** instructions as
   labelled, untrusted source material, **not as a conversation message**.
   The existing three prior messages still have distinct user/assistant roles.
10. The model produces an ordinary reply. If the model server is unreachable
    but Wikipedia found a relevant article, Aria displays the retrieved offline
    passage directly instead of failing. If no suitable article is found, Aria
    displays the normal model-unavailable message.

Use `/debug` in the GUI after any request to inspect the latest routing, RAG
eligibility/result, selected article, cache status, whether the reference was
attached to the model system prompt, model endpoint/status, timings and errors.
The full latest-request trace is stored in `data/aria_debug.json` (override with
`ARIA_DEBUG_FILE`). It is replaced atomically on each request and does not store
the full conversation or model response.

Crutch already knows some simple science and geography facts. For instance,
`What is photosynthesis?` may be answered by Crutch itself, in which case
retrieval is correctly skipped. Use other factual questions to test Wikipedia.

No extra LLM, embedding model, vector database, cloud API, always-on indexing
job or network request is introduced by RAG.

### SQLite cache

The local database stores the normalised query, article title/path, short
passage, standard/detailed mode, archive UUID, archive file signature and last
access time. It contains at most 512 saved passages; access time is updated on
hits. If the ZIM changes size or modification time, old entries are ignored
and cleared automatically when a new passage is stored. A missing database is
created on demand. `/clear` resets only temporary GUI chat context, **not**
Crutch notes or the Wikipedia cache.

### Optional configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `ARIA_ZIM_PATH` | `./data/wiki-eng-simple.zim` | Alternative archive path |
| `ARIA_RAG_CACHE` | `./data/wiki_cache.sqlite3` | Alternative SQLite cache path |
| `ARIA_RAG_DEBUG=1` | Disabled | Print compact RAG retrieval details to stderr |
| `ARIA_DEBUG_FILE` | `./data/aria_debug.json` | Latest request trace shown by `/debug` |
| `CRUTCH_HOME` | `./data` | Override Crutch notes/lists storage |
| `CRUTCH_URL` | `http://localhost:1234/v1/chat/completions` | Existing model endpoint |
| `CRUTCH_MAX_TOKENS` | `120` | Standard Qwen output limit |
| `CRUTCH_DETAIL_MAX_TOKENS` | `512` | Detailed Qwen output limit |

The `./data` paths above mean the `data/` directory alongside these scripts,
not the terminal's current working directory.

## Optional whole-Wikipedia title catalogue

The original Simple English Wikipedia ZIM already has a fast native title index.
Aria now also has an **optional** local SQLite FTS5 catalogue containing only
article titles, ZIM paths and search keys. It enables more flexible question
phrasing (names, aliases, and topic extraction) without scanning all 400,061
articles on every question. **It does not contain article text or replace the
existing ZIM.** An additional compressed ZIM would add unnecessary reader
complexity, and SQLite's B-tree/FTS indexes support direct lookups efficiently.

Build the catalogue **once** on your Linux computer, after installing `libzim`:

```fish
cd ~/ARIA
.venv/bin/python build_title_index.py
```

The resulting `data/wiki_titles.sqlite3` remains in the same ARIA folder.
Index creation reads the existing ZIM title listing, *not* each article body.
The builder uses a private python-libzim accessor because public Python bindings
lack whole-archive iteration. Some library versions may not expose it; if so,
the command reports an explicit error and Aria continues using native ZIM search.
Actual file size and build time depend on the archive and machine. Run
`ls -lh ~/ARIA/data/wiki_titles.sqlite3` to see the size.

On each message, Aria uses Crutch/Hassil first. If an encyclopedic request
is unmatched, it checks the passage cache, attempts fast native title matching,
consults the optional SQLite catalogue when useful, and only then uses ZIM
suggestion/full-text fallback. **It does not load or iterate all titles per
message.** Questions asking for unknown people, dates or other relationships
may also require specialised evidence extraction: the all-title index improves
*finding the source article* but cannot itself prove an answer from a table.
Historical officeholder questions are one example of this distinction; Aria
checks the corresponding timeline instead of asking Qwen to guess the officeholder.

The optional catalogue is invalidated when the ZIM size, modification timestamp,
or UUID changes. Re-run the builder if you replace Wikipedia. Deleting the
SQLite index is safe and does not remove your notes or encyclopedia.

**Streaming GUI output is not yet implemented in this release.** This package
focuses on retrieval and indexing; the existing Qwen server request still
uses non-streaming completions.

## 3. Verification

With the Python virtual environment activated or using `.venv/bin/python`:

```bash
cd /home/blake/ARIA
.venv/bin/python -m py_compile app.py crutch.py aria_backend_bridge.py aria_hassil_router.py aria_wiki_rag.py aria_title_index.py build_title_index.py
.venv/bin/python -m unittest -v test_context_bridge test_gui_context test_hassil_integration test_wiki_rag test_title_index
.venv/bin/python crutch.py --test
.venv/bin/python check_wiki.py
```

`check_wiki.py` uses your **real, intact Wikipedia ZIM**. It reports archive
index diagnostics and tests multiple short and detailed lookups. An error here
is useful: it means the live installation needs inspection, not that a mock
unit test somehow proved full end-to-end correctness.

### What has been verified in the packaging environment

- Python syntax compilation for all updated application components.
- Existing Crutch self-tests: **465 passed, zero failed** (including 252
  exhaustive capital checks).
- Existing GUI/bridge/Hassil tests: **19 passed**.
- Wikipedia, historical and title-catalogue regression tests included.
- Previous release headless suite: **40 passed**; final consolidation suite: **53 passed**.
- HTML cleaning, query routing, conservative article-title selection, redirects,
  caching, archive-change invalidation, missing archive/library fallback, and
  role-separated model history tested headlessly.

**Not verified in this packaging environment:** the real one-time all-title
catalogue build, a `libzim` import, an interactive PyQt6 GUI window, or a
Qwen response. The full ZIM archive is not bundled. Your earlier on-device
`check_wiki.py` results confirmed five live article lookups, but this new
catalogue build must still be checked on your Linux computer. The original
GUI graphics are unchanged.

## Final RAG consolidation and fine-tuning handoff

See `CHANGES_FINAL_RAG.md` for the latest source selection, alias handling,
response-cleaner integration, known limitations and upgrade notes.

The `aria_response_cleaner.py` module is already installed in `crutch.py` in
this release. **Do not run `install_response_cleaner.py` again.**

Run `.venv/bin/python rag_audit.py` to capture a JSONL list of retrieved
articles, reference passages, and retrieval times. To evaluate your own
questions, create a UTF-8 text file with one question per line and run:

```fish
cd ~/ARIA
.venv/bin/python rag_audit.py --questions prompts.txt --output local_rag_audit.jsonl
```

To see RAG match and timing messages while debugging the bridge, run:

```fish
cd ~/ARIA
set -x ARIA_RAG_DEBUG 1
.venv/bin/python aria_backend_bridge.py --prompt 'Who was the Australian prime minister in 1986?'
set -e ARIA_RAG_DEBUG
```

The GUI graphics are unchanged; this update adds the `/debug` local command.
When the model server is offline, a matched Wikipedia passage is now shown
directly. The standalone data audit tests retrieval only, not how well Qwen
uses the retrieved text. Use a separately reviewed held-out set when training
an adapter later.

## Diagnostics and future model lifecycle

`/debug` is available for the latest request trace. See
[`SERVER_LIFECYCLE_PLAN.md`](SERVER_LIFECYCLE_PLAN.md) for the separately
planned (not yet implemented) `Start the server` / `Stop the server` workflow.
That future feature must only unload or stop a server instance Aria explicitly
owns, while keeping Crutch and offline Wikipedia available without a model.

The source snapshot's regression suite currently contains 55 unittest cases.
