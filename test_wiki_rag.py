"""Offline retrieval regression tests with small in-memory libzim test doubles.

These do not claim a live ZIM lookup or an installed libzim binding. Run the
optional check_wiki.py on the target machine with the real 1.09 GB archive.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import aria_backend_bridge as bridge
import aria_wiki_rag as rag
import crutch


HTML = b'''<html><body><div class="mw-parser-output">
<table class="infobox"><tr><td>Bad metadata</td></tr></table>
<p>An atom is the smallest unit of ordinary matter that forms a chemical element.
Atoms consist of a nucleus and surrounding electrons.</p>
<p>Atoms can connect to form molecules and larger chemical structures.</p>
<h2>Structure <span class="mw-editsection">[edit]</span></h2>
<p>The nucleus contains protons and usually neutrons, surrounded by electrons.</p>
<p>Additional structure information is given here to allow a detailed answer.</p>
<h2>References</h2><p>This should never appear in a retrieved text reference.</p>
</div></body></html>'''


class FakeItem:
    content = memoryview(HTML)
    size = len(HTML)
    mimetype = "text/html"


class FakeEntry:
    is_redirect = False
    def __init__(self, title, path=None, data=None):
        self.title = title
        self.path = path or title.replace(" ", "_")
        self._item = data or FakeItem()

    def get_item(self):
        return self._item


class FakeRedirect(FakeEntry):
    is_redirect = True
    def __init__(self, title, target):
        super().__init__(title)
        self.target = target
    def get_redirect_entry(self):
        return self.target


class FakeArchive:
    open_count = 0
    uuid = "a954ad0f-6aa2-b0c7-40bb-5494f33f1a07"
    articles = {
        "Atom": FakeEntry("Atom"),
        "Albert Einstein": FakeEntry("Albert Einstein"),
        "Eiffel Tower": FakeEntry("Eiffel Tower"),
        "Star formation": FakeEntry("Star formation"),
        "Photosynthesis": FakeEntry("Photosynthesis"),
    }

    def __init__(self, path):
        self.path = path
        type(self).open_count += 1
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def get_entry_by_title(self, title):
        if title not in self.articles:
            raise KeyError(title)
        return self.articles[title]
    def get_entry_by_path(self, path):
        for e in self.articles.values():
            if e.path == path:
                return e
        raise KeyError(path)


class FakeQuery:
    def set_query(self, query):
        self.query = query
        return self


class FakeSearcher:
    paths = ["Unrelated_article"]
    def __init__(self, zim):
        self.zim = zim
    def search(self, query):
        return self
    def getResults(self, start, count):
        return self.paths[start:start + count]


class WikiRagTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.zimfile = Path(self.tmp.name) / "wiki-eng-simple.zim"
        self.zimfile.write_bytes(b"fake signature; no encyclopedia included")
        self.cachefile = Path(self.tmp.name) / "wiki_cache.sqlite3"
        self.env = patch.dict(os.environ, {
            "ARIA_ZIM_PATH": str(self.zimfile),
            "ARIA_RAG_CACHE": str(self.cachefile),
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        FakeArchive.open_count = 0
        self.libzim = types.ModuleType("libzim")
        self.reader = types.ModuleType("libzim.reader")
        self.search = types.ModuleType("libzim.search")
        self.reader.Archive = FakeArchive
        self.search.Query = FakeQuery
        self.search.Searcher = FakeSearcher
        self.modules = patch.dict(sys.modules, {
            "libzim": self.libzim,
            "libzim.reader": self.reader,
            "libzim.search": self.search,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_factual_and_not_factual(self):
        eligible = ["What is an atom?", "What is photosynthesis?", "Who was Albert Einstein?",
                    "Tell me about the Eiffel Tower.", "Explain atoms in detail.",
                    "Describe photosynthesis thoroughly.", "Elaborate on how stars form."]
        excluded = ["Tell me a joke.", "How are you?", "What's the weather right now?",
                    "What lesson do I have next?", "Add milk to my shopping list.",
                    "What is it?", "What do you think?", "What's the latest news?",
                    "Who's the current prime minister of Australia?"]
        for q in eligible:
            with self.subTest(eligible=q):
                self.assertTrue(rag.is_knowledge_question(q))
        for q in excluded:
            with self.subTest(excluded=q):
                self.assertFalse(rag.is_knowledge_question(q))

    def test_unmatched_question_types_and_bare_name(self):
        for q in ("Bob Hawke", "Who wrote Pride and Prejudice?", "When was Albert Einstein born?"):
            with self.subTest(q=q):
                self.assertTrue(rag.is_knowledge_question(q))
        with patch.dict(FakeArchive.articles, {"Bob Hawke": FakeEntry("Bob Hawke")}):
            result = rag.retrieve("Bob Hawke")
        self.assertIsNotNone(result)
        self.assertEqual(result.title, "Bob Hawke")

    def test_historical_officeholder_parser(self):
        intent = rag.historical_intent("Who was the Australian prime minister in 1986?")
        self.assertIsNotNone(intent)
        self.assertEqual((intent.country, intent.office, intent.year),
                         ("australia", "Prime Minister", 1986))

    def test_officeholder_table_extraction(self):
        html = b"""<table><tr><td>22</td><td>Malcolm Fraser</td><td>Liberal</td><td>11 November 1975</td><td>11 March 1983</td></tr>
        <tr><td>23</td><td>Bob Hawke</td><td>Labor</td><td>11 March 1983</td><td>20 December 1991</td></tr></table>"""
        matches = rag._officeholder_rows(html, 1986)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], "Bob Hawke")

    def test_subject_variations(self):
        pairs = {"What is an atom?": "atom", "Explain atoms in detail.": "atoms",
                 "Who was Albert Einstein?": "Albert Einstein",
                 "Tell me about the Eiffel Tower.": "Eiffel Tower",
                 "Elaborate on how stars form.": "star formation"}
        for q, topic in pairs.items():
            self.assertEqual(rag.subject(q), topic)

    def test_paragraph_cleaning_and_modes(self):
        paragraphs = rag.parse_html(HTML)
        intro = rag._standard_passage(paragraphs, rag.NORMAL_CHAR_BUDGET)
        detail = rag.expanded_passage(paragraphs, "Explain atoms in detail.")
        self.assertIn("smallest unit", intro)
        self.assertNotIn("Bad metadata", intro)
        self.assertNotIn("References", detail)
        self.assertNotIn("This should never", detail)
        self.assertIn("protons", detail)
        self.assertGreater(len(detail), len(intro))

    def test_read_from_title_index_and_cache(self):
        first = rag.retrieve("What is an atom?")
        self.assertEqual(first.title, "Atom")
        self.assertFalse(first.from_cache)
        second = rag.retrieve("What is an atom?")
        self.assertTrue(second.from_cache)
        self.assertEqual(first.passage, second.passage)
        self.assertEqual(FakeArchive.open_count, 1)
        with rag.cache_connection() as db:
            row = db.execute("SELECT archive_uuid, mode FROM wiki_passages").fetchone()
        self.assertEqual(row, (FakeArchive.uuid, "standard"))

    def test_detailed_and_standard_separate_cache(self):
        short = rag.retrieve("Explain atoms in detail.", detailed=False)
        long = rag.retrieve("Explain atoms in detail.", detailed=True)
        self.assertGreater(len(long.passage), len(short.passage))
        self.assertEqual(FakeArchive.open_count, 2)
        rag.retrieve("Explain atoms in detail.", detailed=False)
        rag.retrieve("Explain atoms in detail.", detailed=True)
        self.assertEqual(FakeArchive.open_count, 2)

    def test_archive_update_invalidates_cache(self):
        rag.retrieve("What is an atom?")
        self.zimfile.write_bytes(b"archive has changed, but ZIM is still fake")
        new = rag.retrieve("What is an atom?")
        self.assertFalse(new.from_cache)
        self.assertEqual(FakeArchive.open_count, 2)
        with rag.cache_connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM wiki_passages").fetchone()[0], 1)

    def test_redirect_resolution(self):
        with patch.dict(FakeArchive.articles, {"Atom": FakeRedirect("Atom", FakeEntry("Atom"))}):
            result = rag.retrieve("What is an atom?")
        self.assertEqual(result.title, "Atom")

    def test_fulltext_irrelevant_match_rejected(self):
        with patch.dict(FakeArchive.articles, {"Irrelevant article": FakeEntry("Irrelevant article")}, clear=True):
            FakeSearcher.paths = ["Irrelevant_article"]
            self.assertIsNone(rag.retrieve("Who was Albert Einstein?"))

    def test_missing_archive_or_libzim_does_not_crash(self):
        self.zimfile.unlink()
        self.assertIsNone(rag.retrieve("What is an atom?"))
        self.zimfile.write_bytes(b"fake ZIM")
        with patch.dict(sys.modules, {"libzim.reader": None}):
            self.assertIsNone(rag.retrieve("What is an atom?"))

    def test_cache_initialization_without_database(self):
        self.assertFalse(self.cachefile.exists())
        self.assertIsNotNone(rag.retrieve("What is an atom?"))
        self.assertTrue(self.cachefile.is_file())

    def test_cache_bounded_to_512(self):
        signature = rag._source_signature(self.zimfile)
        ref = rag.RagReference("Atom", "Atom", "Reference text." * 5)
        for n in range(518):
            rag.cache_write(signature, FakeArchive.uuid, f"What is item {n}?", False, ref)
        with rag.cache_connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM wiki_passages").fetchone()[0], 512)

    def test_ref_keeps_original_model_prompts(self):
        ref = rag.retrieve("What is an atom?")
        system = rag.model_reference_context(crutch, ref, detailed=False)
        detailed_system = rag.model_reference_context(crutch, ref, detailed=True)
        self.assertTrue(system.startswith(crutch.SYSTEM_PROMPT))
        self.assertTrue(detailed_system.startswith(crutch.DETAILED_SYSTEM_PROMPT))
        self.assertIn("not a new user turn", system)
        self.assertIn("Atom", system)

    def test_model_bridge_gets_reference_as_system_not_user(self):
        payloads = []
        def fake_model(prompt, system=None, detailed=False, history=None):
            payloads.append((prompt, system, detailed, history))
            return "An atom is a basic unit of matter."
        class Backend:
            ModelUnavailable = crutch.ModelUnavailable
            SYSTEM_PROMPT = crutch.SYSTEM_PROMPT
            DETAILED_SYSTEM_PROMPT = crutch.DETAILED_SYSTEM_PROMPT
            route = staticmethod(lambda prompt: None)
            respond = staticmethod(lambda prompt, use_model=True: ("", "none"))
            wants_detailed_reply = staticmethod(crutch.wants_detailed_reply)
            ask_model = staticmethod(fake_model)
            plain = staticmethod(crutch.plain)
        context = json.dumps([{"role":"assistant","content":"An earlier response."}])
        with patch.object(bridge, "load_backend", return_value=Backend), \
             patch.object(sys, "argv", ["bridge.py", "--prompt", "What is an atom?", "--context", context]):
            with redirect_stdout(StringIO()) as out:
                self.assertEqual(bridge.main(), 0)
        self.assertIn("basic unit", out.getvalue())
        self.assertEqual(payloads[0][0], "What is an atom?")
        self.assertIn("OFFLINE WIKIPEDIA REFERENCE", payloads[0][1])
        self.assertEqual(payloads[0][3], [{"role":"assistant","content":"An earlier response."}])

    def test_crutch_match_short_circuits_wikipedia(self):
        class Backend:
            route = staticmethod(lambda prompt: ("Already handled by Crutch.", "knowledge"))
        with patch.object(bridge, "load_backend", return_value=Backend), \
             patch.object(rag, "retrieve", side_effect=AssertionError("should not run")), \
             patch.object(sys, "argv", ["bridge.py", "--prompt", "What is photosynthesis?"]):
            with redirect_stdout(StringIO()) as out:
                self.assertEqual(bridge.main(), 0)
        self.assertIn("Already handled", out.getvalue())


if __name__ == "__main__":
    unittest.main()
