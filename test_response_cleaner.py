"""Regression checks for cleaning only unfinished model response endings."""
from __future__ import annotations

import json
import unittest
from io import BytesIO
from unittest.mock import patch

import crutch
from aria_response_cleaner import trim_unfinished_sentence as trim


class ResponseCleanerTests(unittest.TestCase):
    def test_long_cutoff(self):
        text = 'Atoms contain electrons. Some details about boiling'
        self.assertEqual(trim(text), 'Atoms contain electrons.')

    def test_keeps_final_punctuation(self):
        for answer in ('Done.', 'Really?', 'Yes!', 'He said "Done."', 'That is right!'):
            with self.subTest(answer=answer):
                self.assertEqual(trim(answer), answer)

    def test_no_complete_sentence_keeps_short_reply(self):
        for answer in ('Hello', 'Bob Hawke', 'The radius is 0.5 nanometers and'):
            with self.subTest(answer=answer):
                self.assertEqual(trim(answer), answer)

    def test_decimal_not_sentence_end(self):
        self.assertEqual(trim('The diameter was 0.5 millimetres and'),
                         'The diameter was 0.5 millimetres and')
        self.assertEqual(trim('Atoms are tiny. The size is 0.5 nanometres and'),
                         'Atoms are tiny.')

    def test_abbreviations_and_numbered_sections(self):
        self.assertEqual(trim('Dr. Watson was a researcher and'),
                         'Dr. Watson was a researcher and')
        self.assertEqual(trim('1. First topic is the nucleus and'),
                         '1. First topic is the nucleus and')
        self.assertEqual(trim('Dr. Watson was a researcher. His assistant was'),
                         'Dr. Watson was a researcher.')

    def test_quoted_sentence_end(self):
        self.assertEqual(trim('She said "Hello." Then she began'), 'She said "Hello."')

    def test_model_integration_with_length_finish(self):
        result = {"choices": [{"message": {"content": "An atom is matter. It also has"},
                                "finish_reason": "length"}]}
        class Reply(BytesIO):
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.close()
        def mocked_open(req, timeout):
            return Reply(json.dumps(result).encode('utf-8'))
        with patch.object(crutch.urllib.request, 'urlopen', mocked_open):
            self.assertEqual(crutch.ask_model('Explain atoms', detailed=True),
                             'An atom is matter.')


if __name__ == '__main__':
    unittest.main()
