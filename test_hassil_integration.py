"""Run offline with: python -m unittest -v test_hassil_integration.py"""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import crutch
import aria_hassil_router as matcher

class HassilIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.previous = os.environ.get('CRUTCH_HOME')
        os.environ['CRUTCH_HOME'] = cls.temp.name
        crutch.NOW_OVERRIDE = crutch.datetime(2026, 10, 27, 12, 0)

    @classmethod
    def tearDownClass(cls):
        crutch.NOW_OVERRIDE = None
        if cls.previous is None:
            os.environ.pop('CRUTCH_HOME', None)
        else:
            os.environ['CRUTCH_HOME'] = cls.previous
        cls.temp.cleanup()

    def test_matching_and_answers(self):
        examples = {
            'tell me the current time': ('hassil:TimeNow', '12:00 PM'),
            'what date are we on': ('hassil:DateNow', 'Tuesday'),
            'tell me tomorrow\'s date': ('datetime', 'Wednesday'),
            'what class comes after this': ('hassil:NextClass', 'Next is'),
            'what lesson follows this one': ('hassil:NextClass', 'Next is'),
            'what\'s the next thing on my timetable': ('hassil:NextClass', 'Next is'),
            'which city serves as the capital of California': ('hassil:Capital', 'Sacramento'),
            'which city is the capital of New South Wales': ('hassil:Capital', 'Sydney'),
            'where would i find the capital of france': ('hassil:Capital', 'Paris'),
            'give me the postal abbreviation of California': ('hassil:Abbreviation', 'CA'),
            'how far away is Halloween': ('hassil:Holiday', 'October 31'),
            'which ocean covers the most area': ('hassil:LargestOcean', 'Pacific'),
            'what animal runs the fastest on land': ('hassil:FastestAnimal', 'cheetah'),
            'what does 10 kilograms equal in pounds': ('hassil:UnitConversion', '22.05 pounds'),
            'how much does 15 percent of 200 make': ('percent', '30'),
            'what do you get if you multiply 6 by 7': ('hassil:Times', '42'),
            'can you read everything on my shopping list': ('hassil:ReadShopping', 'empty'),
            'could you tell me every one of my saved notes': ('hassil:ShowNotes', 'no notes'),
        }
        for prompt, (source, expect) in examples.items():
            with self.subTest(prompt=prompt):
                r = crutch.route(prompt)
                self.assertIsNotNone(r)
                self.assertEqual(r[1], source)
                self.assertIn(expect.lower(), r[0].lower())

    def test_all_capitals_with_new_wording(self):
        for place, answer in crutch.US_STATES.items():
            with self.subTest(us=place):
                r = crutch.route(f'which city serves as the capital of {place}')
                self.assertIn(answer[0].lower(), (r or ('',))[0].lower())
        for place, (capital, *_) in crutch.AU_REGIONS.items():
            with self.subTest(australia=place):
                r = crutch.route(f'which city serves as the capital of {place}')
                self.assertIn(capital.lower(), (r or ('',))[0].lower())
        for place, capital in crutch.COUNTRY_CAPITALS.items():
            with self.subTest(country=place):
                r = crutch.route(f'which city serves as the capital of {place}')
                self.assertIn(capital.lower(), (r or ('',))[0].lower())

    def test_unrelated_remains_unmatched(self):
        prompts = [
            'what is the current population of Paris',
            'what time does the next train leave',
            'turn off the living room lights',
            'I was discussing how to convert my thoughts into code',
            'tell me a story about clocks and geography',
            'make up a country and its capital',
            'what is the meaning of life',
            'explain in detail why the sky is blue',
            'can you play minecraft',
            'how much would pizza cost in New York',
            'what does some potatoes equal in bananas',
            'which ocean is the best',
            'remind me next week to pay the bill',
            "what is my next doctor's appointment",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertIsNone(crutch.route(prompt))

    def test_no_double_import_in_bridge(self):
        # The bridge deliberately loads the backend with a different name.
        spec = importlib.util.spec_from_file_location('aria_crutch_backend', ROOT / 'crutch.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.assertIn('Sacramento', module.route('which city serves as the capital of California')[0])
        self.assertEqual(module.route('how much does 20 percent of 100 make')[1], 'percent')

    def test_optional_without_vendored_router(self):
        # Script-level compatibility relies on catching ImportError. Actual
        # fallback is tested in a subprocess by the package smoke test.
        self.assertIn('Paris', crutch.tool_geography('what is the capital of france'))

if __name__ == '__main__':
    unittest.main()
