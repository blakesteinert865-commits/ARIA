"""Extra source-selection regressions for broad offline title retrieval.

These tests do not replace actual ZIM verification on the target machine.
"""
from __future__ import annotations
import unittest
from unittest.mock import patch
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import aria_title_index as titles
import aria_wiki_rag as rag


class RetrievalRefinementTests(unittest.TestCase):
    def test_nested_and_indirect_subjects(self):
        examples = {
            'Explain what an atom is in detail': 'atom',
            'Describe what an atom is': 'atom',
            'What is an atom?': 'atom',
            'Who wrote Pride and Prejudice?': 'Pride and Prejudice',
            'Who invented the telephone?': 'telephone',
            'What causes earthquakes?': 'earthquakes',
            'When was Albert Einstein born?': 'Albert Einstein',
            'When did the Titanic sink?': 'Titanic',
            'Explain how stars form': 'star formation',
            'Bob Hawke': 'Bob Hawke',
        }
        for question, candidate in examples.items():
            with self.subTest(question=question):
                self.assertIn(titles.normalise(candidate),
                              [titles.normalise(x) for x in titles.subjects(question)])
                self.assertTrue(rag.is_knowledge_question(question))

    def test_title_phrases_and_non_knowledge(self):
        self.assertTrue(rag.is_knowledge_question('Pride and Prejudice'))
        self.assertTrue(rag.is_knowledge_question('World War II'))
        for text in ('Hello there', 'How are you?', 'What is it?',
                     'Set a timer for 10 minutes', 'What is the weather today?'):
            with self.subTest(text=text):
                self.assertFalse(rag.is_knowledge_question(text))

    def test_no_vague_matches(self):
        self.assertLess(rag._confidence('Albert Einstein', 'Albert Park'), 0.84)
        self.assertLess(rag._confidence('Prime Minister of Australia', 'Prime Minister of Canada'), .84)

    def test_suggestion_getResults_compatibility(self):
        class Entry:
            title = 'Atom'
            path = 'Atom'
            is_redirect = False
        class Zim:
            uuid = 'fake'
            def get_entry_by_title(self, title):
                raise KeyError(title)
            def get_entry_by_path(self, path):
                if path == 'Atom':
                    return Entry()
                raise KeyError(path)
        class Sugg:
            def __init__(self, zim):
                pass
            def suggest(self, text):
                return SimpleNamespace(getResults=lambda a, b: ['Atom'] if text == 'atom' else [])
        from types import ModuleType
        import sys
        mod = ModuleType('libzim.suggestion')
        mod.SuggestionSearcher = Sugg
        with patch.dict(sys.modules, {'libzim.suggestion': mod}):
            with patch('aria_title_index.lookup', return_value=None):
                self.assertEqual(rag.find_article(Zim(), 'What is an atom?').title, 'Atom')

    def test_historical_year_transition(self):
        html = b'<table>' + b''.join([
            b'<tr><td>22</td><td>Malcolm Fraser</td><td>Liberal</td><td>11 November 1975</td><td>11 March 1983</td></tr>',
            b'<tr><td>23</td><td>Bob Hawke</td><td>Labor</td><td>11 March 1983</td><td>20 December 1991</td></tr>',
        ]) + b'</table>'
        self.assertEqual([p[0] for p in rag._officeholder_rows(html, 1986)], ['Bob Hawke'])
        self.assertEqual({p[0] for p in rag._officeholder_rows(html, 1983)},
                         {'Malcolm Fraser', 'Bob Hawke'})

    def test_cache_parser_versioning(self):
        with patch.object(rag, 'RETRIEVER_VERSION', 'A'):
            old = rag._source_signature(Path(__file__))
        with patch.object(rag, 'RETRIEVER_VERSION', 'B'):
            new = rag._source_signature(Path(__file__))
        self.assertNotEqual(old, new)


if __name__ == '__main__':
    unittest.main()
