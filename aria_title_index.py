"""Compact, optional all-title resolver for Aria's existing Wikipedia ZIM.

Index metadata only: titles, paths, normalised lookup keys. Article text is
NEVER copied; the original ZIM remains the single source of factual passages.
This is a convenience index, not a second encyclopedia or language model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import sqlite3
import struct
import unicodedata

APP_DIR = Path(__file__).resolve().parent
INDEX_PATH = APP_DIR / 'data' / 'wiki_titles.sqlite3'
VERSION = '1'


def normalise(s: str) -> str:
    s = ''.join(c for c in unicodedata.normalize('NFKD', s.casefold()) if not unicodedata.combining(c))
    s = s.replace('_', ' ').replace('&', ' and ').replace('’', "'")
    s = re.sub(r"[^\w\s]", ' ', s, flags=re.UNICODE)
    return ' '.join(s.split())


def _country(text: str) -> str | None:
    """Reuse canonical names and aliases from existing deterministic Crutch."""
    try:
        import crutch
        candidates = {'australian': 'australia', 'american': 'united states',
                      'british': 'united kingdom', 'canadian': 'canada',
                      'french': 'france', 'german': 'germany', 'indian': 'india',
                      'japanese': 'japan', 'irish': 'ireland', 'italian': 'italy',
                      'new zealand': 'new zealand', 'new zealander': 'new zealand'}
        key = normalise(text)
        key = candidates.get(key, key)
        key = crutch._geo_entity(key)
        return key if key in crutch.COUNTRY_CAPITALS else None
    except (ImportError, AttributeError):
        return None


def subjects(question: str) -> list[str]:
    """Extract plausible encyclopedia titles, not guessed answers.

    Deterministic intent reduction works on one request at a time. Query forms
    are converted to *source topics*; a conservative ZIM-title match must still
    approve every candidate. Crutch supplies the geographic alias vocabulary.
    """
    q = re.sub(r"\s+", " ", question.strip().replace("’", "'"))
    q = q.strip(" ?!.,")
    q = re.sub(r"\s+(?:in detail|thoroughly|in depth|comprehensively|fully|step by step)\s*$", "", q, flags=re.I)
    if not q:
        return []
    candidates: list[str] = []

    def add(value: str) -> None:
        value = value.strip(" \t.!?;,:\"'")
        value = re.sub(r"^(?:a|an|the)\s+", "", value, flags=re.I)
        if value and len(value) <= 150 and normalise(value) not in {normalise(x) for x in candidates}:
            candidates.append(value)

    # '[office] of [country]' is an example of an entity relationship, not a
    # hard-coded historical answer. The date is handled by the evidence reader.
    m = re.search(r"\b(prime\s+minister|president|monarch|king|queen|governor|capital)\b", q, re.I)
    if m:
        office = m.group(1).lower()
        before, after = q[:m.start()], q[m.end():]
        country = None
        of_match = re.search(r"\b(?:of|for)\s+(?:the\s+)?(.+)", after, re.I)
        if of_match:
            place = re.sub(r"\b(?:in|during|on|at|as of)\s+(?:the year\s+)?\d{4}\b.*$", "", of_match.group(1), flags=re.I).strip()
            country = _country(place)
        if country is None:
            phrase = re.sub(r"^(?:who|what|which)\s+(?:was|is|were|are)\s+(?:the\s+)?", "", before, flags=re.I)
            country = _country(phrase.strip())
        if country:
            title_country = {"united states": "the United States",
                             "united kingdom": "the United Kingdom"}.get(country, country.title())
            add(f"{office.title()} of {title_country}")

    candidate = re.sub(r"\b(?:in|during|as of|throughout|on)\s+(?:the year\s+)?(?:1[5-9]\d\d|20\d\d)\b.*$", "", q, flags=re.I)

    # Explicit embedded questions are common speech: "Explain what an atom is".
    candidate = re.sub(r"^(?:please\s+)?(?:explain|describe|elaborate on|define|tell me about)\s+", "", candidate, flags=re.I)
    embedded = re.fullmatch(r"(?:what|who)\s+(?:is|are|was|were)\s+(.+?)\s+(?:is|are|was|were)", candidate, re.I)
    if embedded:
        candidate = embedded.group(1)
    candidate = re.sub(r"^(?:what|who|where|when|which|why|how)\s+(?:is|are|was|were|does|did|do|has|had)\s+", "", candidate, flags=re.I)
    candidate = re.sub(r"^(?:what\s+causes|who\s+(?:wrote|invented|discovered|created|founded|designed|painted)|when\s+(?:was|were|did))\s+", "", candidate, flags=re.I)
    candidate = re.sub(r"^(?:what\s+causes|who\s+(?:wrote|invented|discovered|created|founded|designed|painted))\s+", "", candidate, flags=re.I)
    candidate = re.sub(r"\s+(?:born|created|invented|founded|discovered|located|written|made|built|published|died|sink|sank|end|ended|begin|began)\s*$", "", candidate, flags=re.I)
    candidate = re.sub(r"\s+(?:work|works|function|functions)\s*$", "", candidate, flags=re.I)
    # "What an atom is" is a second useful embedded form.
    candidate = re.sub(r"^what\s+(?:an?|the)\s+(.+?)\s+is$", r"\1", candidate, flags=re.I)
    # Common relationship-question surface forms. The result remains a title
    # candidate, never an assumed answer to the relationship itself.
    candidate = re.sub(r"^happens\s+(?:during|in)\s+", "", candidate, flags=re.I)
    candidate = re.sub(r"\s+(?:erupt|erupts|happen|happens|occur|occurs)\s*$", "", candidate, flags=re.I)
    # Transparent topic-to-article noun phrases, not arbitrary name guesses.
    if re.fullmatch(r"(?:how\s+)?stars?\s+(?:are\s+)?form(?:ed|ing)?", candidate, re.I):
        candidate = "star formation"
    if re.fullmatch(r"photosynthesis\s+(?:works|happens)", candidate, re.I):
        candidate = "photosynthesis"
    # Discard empty/generic source candidates.
    if normalise(candidate) not in {"what", "who", "which", "why", "how", "it", "this", "that"}:
        add(candidate)
    return candidates[:5]


def _signature(zim_path: Path, uuid: str) -> str:
    s = zim_path.stat()
    return f'{uuid}|{s.st_size}|{s.st_mtime_ns}'


def _title_ids(zim):
    """Enumerate every FRONT_ARTICLE via ZIM's stored title-ordered IDs.

    python-libzim doesn't expose general article iteration. The original ZIM
    contains its own compact title-order listing; access by internal ID is an
    implementation detail, so we validate the entry and fail clearly if the
    binding changes. Do not decompress individual articles during indexing.
    """
    getter = getattr(zim, '_get_entry_by_id', None)
    if not callable(getter):
        raise RuntimeError('Installed libzim lacks _get_entry_by_id; cannot build full catalogue')
    listing = None
    try:
        entry = zim.get_entry_by_path('listing/titleOrdered/v1')
        if str(getattr(entry, 'path', 'listing/titleOrdered/v1')).endswith('listing/titleOrdered/v1'):
            listing = entry
    except (KeyError, ValueError, RuntimeError):
        pass
    if listing is None:
        # The listing is an internal ZIM entry, typically very close to the
        # end of the internal ID range and inaccessible through public paths.
        # Consult only a bounded tail; never scan whole articles to locate it.
        for idx in range(int(zim.all_entry_count) - 1,
                         max(-1, int(zim.all_entry_count) - 129), -1):
            try:
                entry = getter(idx)
                if str(entry.path).endswith('listing/titleOrdered/v1'):
                    listing = entry
                    break
            except (KeyError, ValueError, RuntimeError, IndexError):
                continue
    if listing is None:
        raise RuntimeError('ZIM internal titleOrdered/v1 listing not accessible with installed libzim')
    content = bytes(listing.get_item().content)
    if len(content) % 4:
        raise ValueError('Invalid titleOrdered/v1 index length')
    count = len(content) // 4
    if not (0 < count <= zim.all_entry_count):
        raise ValueError('Invalid number of indexed ZIM titles')
    for (idx,) in struct.iter_unpack('<I', content):
        if idx >= zim.all_entry_count:
            raise ValueError('Invalid titleOrdered/v1 entry ID')
        yield idx


def build(zim_path: Path, index_path: Path = INDEX_PATH, *, progress=None) -> int:
    """One-time metadata index build, atomically replacing an old index.

    A few tens of MB of SQLite data is expected, depending on title lengths.
    Can be deleted at any time; the original Wikipedia ZIM is never modified.
    """
    from libzim.reader import Archive
    zim_path = Path(zim_path)
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_name(index_path.name + '.building')
    if tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    try:
        db.execute('PRAGMA journal_mode=DELETE')
        db.execute('PRAGMA synchronous=OFF')
        db.execute('CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        db.execute('CREATE TABLE titles (id INTEGER PRIMARY KEY, title TEXT NOT NULL, path TEXT NOT NULL, norm TEXT NOT NULL)')
        db.execute('CREATE INDEX titles_norm ON titles(norm)')
        db.execute('CREATE VIRTUAL TABLE title_search USING fts5(norm, tokenize="unicode61 remove_diacritics 2")')
        zim = Archive(str(zim_path))
        try:
            count = 0
            batch = []
            for idx in _title_ids(zim):
                entry = zim._get_entry_by_id(idx)
                title, path = str(entry.title), str(entry.path)
                if not title or not path:
                    continue
                batch.append((title, path, normalise(title)))
                if len(batch) >= 2000:
                    _insert_batch(db, batch)
                    count += len(batch)
                    batch.clear()
                    if progress and count % 20000 == 0:
                        progress(count)
            if batch:
                _insert_batch(db, batch)
                count += len(batch)
            sig = _signature(zim_path, str(zim.uuid))
            db.executemany('INSERT INTO meta VALUES (?, ?)', [('version', VERSION), ('signature', sig), ('count', str(count))])
            db.commit()
        finally:
            del zim
    except BaseException:
        db.close()
        tmp.unlink(missing_ok=True)
        raise
    else:
        db.close()
        os.replace(tmp, index_path)
        return count


def _insert_batch(db, batch) -> None:
    # rowids are kept identical in both tables for an inexpensive JOIN.
    db.executemany('INSERT INTO titles (title, path, norm) VALUES (?, ?, ?)', batch)
    ids = list(db.execute('SELECT id, norm FROM titles ORDER BY id DESC LIMIT ?', (len(batch),)))
    db.executemany('INSERT INTO title_search (rowid, norm) VALUES (?, ?)', ids)


@dataclass(frozen=True)
class TitleHit:
    title: str
    path: str
    confidence: float


def lookup(question: str, source_path: Path, archive_uuid: str, index_path: Path = INDEX_PATH) -> TitleHit | None:
    """Return a well-supported title match, or None; no unrestricted guessing."""
    index_path = Path(index_path)
    if not index_path.is_file():
        return None
    try:
        db = sqlite3.connect(f'file:{index_path.as_posix()}?mode=ro', uri=True, timeout=2)
        try:
            meta = dict(db.execute('SELECT key, value FROM meta'))
            if meta.get('version') != VERSION or meta.get('signature') != _signature(Path(source_path), archive_uuid):
                return None
            for candidate in subjects(question):
                norm = normalise(candidate)
                if not norm:
                    continue
                exacts = db.execute('SELECT title, path FROM titles WHERE norm=? LIMIT 2', (norm,)).fetchall()
                if len(exacts) == 1:
                    return TitleHit(exacts[0][0], exacts[0][1], 1.0)
                # FTS title-token overlap only: refuse loosely related results.
                terms = [t for t in norm.split() if len(t) > 2 and t not in {'the','of','in','and','for'}]
                if not terms:
                    continue
                expr = ' AND '.join('"' + t.replace('"', '') + '"' for t in terms[:8])
                rows = db.execute('''SELECT t.title, t.path, t.norm FROM title_search f
                     JOIN titles t ON t.id=f.rowid WHERE title_search MATCH ? LIMIT 12''', (expr,)).fetchall()
                best = []
                want = set(terms)
                for title, path, title_norm in rows:
                    got = set(title_norm.split()) - {'the', 'of', 'in', 'and'}
                    overlap = len(want & got)
                    # Require all query terms and no more than one extra title word.
                    if overlap == len(want) and len(got - want) <= 1:
                        best.append(TitleHit(title, path, 0.85))
                if len(best) == 1:
                    return best[0]
            return None
        finally:
            db.close()
    except (OSError, ValueError, sqlite3.Error):
        return None
