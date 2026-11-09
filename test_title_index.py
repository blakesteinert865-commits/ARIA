"""Synthetic metadata-level tests for the optional all-title catalogue."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import struct
import sys
import unittest

import aria_title_index as titles


class MockArchive:
    uuid = 'fake-archive'
    all_entry_count = 6
    data = [
        ('Atom', 'Atom'),
        ('Bob Hawke', 'Bob_Hawke'),
        ('Prime Minister of Australia', 'Prime_Minister_of_Australia'),
        ('Pride and Prejudice', 'Pride_and_Prejudice'),
        ('Eiffel Tower', 'Eiffel_Tower'),
    ]

    def __init__(self, path):
        self.path = path

    def get_entry_by_path(self, path):
        if path != 'listing/titleOrdered/v1':
            raise KeyError(path)
        indexes = struct.pack('<' + 'I' * len(self.data), *range(len(self.data)))
        return SimpleNamespace(get_item=lambda: SimpleNamespace(content=memoryview(indexes)))

    def _get_entry_by_id(self, idx):
        name, path = self.data[idx]
        return SimpleNamespace(title=name, path=path)


class TitleIndexTests(unittest.TestCase):
    def test_query_intent_is_general(self):
        items = {
            'What is an atom?': 'atom',
            'Bob Hawke': 'Bob Hawke',
            'Who was the Australian prime minister in 1986?': 'Prime Minister of Australia',
            'Who wrote Pride and Prejudice?': 'Pride and Prejudice',
            'When was Albert Einstein born?': 'Albert Einstein',
            'Explain atoms in detail.': 'atoms',
        }
        for question, expected in items.items():
            with self.subTest(question=question):
                self.assertIn(expected, titles.subjects(question))

    def test_full_catalogue_and_stale_index(self):
        with TemporaryDirectory() as d:
            zim = Path(d) / 'wiki.zim'
            db = Path(d) / 'titles.sqlite3'
            zim.write_bytes(b'example fake archive')
            reader = ModuleType('libzim.reader')
            reader.Archive = MockArchive
            libzim = ModuleType('libzim')
            with patch.dict(sys.modules, {'libzim': libzim, 'libzim.reader': reader}):
                count = titles.build(zim, db)
            self.assertEqual(count, 5)
            tests = {
                'Bob Hawke': 'Bob Hawke',
                'Who was the Australian prime minister in 1986?': 'Prime Minister of Australia',
                'Who wrote Pride and Prejudice?': 'Pride and Prejudice',
                'Tell me about the Eiffel Tower.': 'Eiffel Tower',
            }
            for q, wanted in tests.items():
                with self.subTest(q=q):
                    hit = titles.lookup(q, zim, MockArchive.uuid, db)
                    self.assertIsNotNone(hit, q)
                    self.assertEqual(hit.title, wanted)
            self.assertIsNone(titles.lookup('Who invented the lightbulb?', zim, MockArchive.uuid, db))
            self.assertIsNone(titles.lookup('Who was the Australian prime minister in 1986?', zim, 'wrong-uuid', db))
            zim.write_bytes(b'changed archive')
            self.assertIsNone(titles.lookup('Bob Hawke', zim, MockArchive.uuid, db))

    def test_internal_listing_fallback(self):
        class PrivateListing(MockArchive):
            all_entry_count = 6
            def get_entry_by_path(self, path):
                raise KeyError(path)
            def _get_entry_by_id(self, idx):
                if idx == 5:
                    indexes = struct.pack('<' + 'I' * len(self.data), *range(len(self.data)))
                    return SimpleNamespace(path='listing/titleOrdered/v1',
                                           get_item=lambda: SimpleNamespace(content=memoryview(indexes)))
                return super()._get_entry_by_id(idx)
        with TemporaryDirectory() as d:
            zim = Path(d) / 'wiki.zim'
            zim.write_bytes(b'fake')
            reader = ModuleType('libzim.reader')
            reader.Archive = PrivateListing
            with patch.dict(sys.modules, {'libzim': ModuleType('libzim'), 'libzim.reader': reader}):
                count = titles.build(zim, Path(d) / 'titles.db')
            self.assertEqual(count, 5)

    def test_not_buildable_without_private_accessor(self):
        class Incompatible(MockArchive):
            _get_entry_by_id = None
        with TemporaryDirectory() as d:
            zim = Path(d) / 'wiki.zim'
            zim.write_bytes(b'x')
            reader = ModuleType('libzim.reader')
            reader.Archive = Incompatible
            libzim = ModuleType('libzim')
            with patch.dict(sys.modules, {'libzim': libzim, 'libzim.reader': reader}):
                with self.assertRaises((RuntimeError, TypeError)):
                    titles.build(zim, Path(d) / 'index.db')
            self.assertFalse((Path(d) / 'index.db').exists())


if __name__ == '__main__':
    unittest.main()
