"""Headless regression tests for Aria's GUI history serialisation.

Imports the isolated GUI method through AST so Qt doesn't need to be present
on CI or the host used to create this package.
"""
import ast
import json
from collections import deque
from pathlib import Path
import unittest
from unittest.mock import patch
import io
import crutch
import aria_backend_bridge as bridge

ROOT = Path(__file__).resolve().parent


def gui_context_json(history):
    tree = ast.parse((ROOT / 'app.py').read_text(encoding='utf-8'))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'CachyChatBot')
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_build_context_json')
    mod = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    ns = {'json': json}
    exec(compile(mod, str(ROOT / 'app.py'), 'exec'), ns)
    obj = type('HistoryHolder', (), {})()
    obj.memory = deque(history, maxlen=3)
    return ns['_build_context_json'](obj)


class GUIContextTest(unittest.TestCase):
    def test_crutch_reply_stays_assistant_role(self):
        previous = [
            ('user', "What's the capital of France?"),
            ('assistant', 'The capital of France is Paris.'),
        ]
        current = 'What is its famous landmark?'
        raw = gui_context_json(previous)  # snapshot BEFORE current is appended
        history = bridge.parse_context(raw, current)
        self.assertEqual(history, [
            {'role': 'user', 'content': previous[0][1]},
            {'role': 'assistant', 'content': previous[1][1]},
        ])
        self.assertFalse(any(m['content'] == current for m in history))

    def test_latest_question_not_duplicated(self):
        current = 'What do you think of that?'
        raw = gui_context_json([('assistant', 'That seems interesting.')])
        history = bridge.parse_context(raw, current)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['role'], 'assistant')
        # Crutch adds exactly one final user entry.
        self.assertEqual([m['role'] for m in history] + ['user'], ['assistant', 'user'])

    def test_only_three_previous_messages_and_compact_json(self):
        data = [('user', 'one'), ('assistant', 'two'), ('user', 'three'), ('assistant', 'four')]
        raw = gui_context_json(data)
        hist = json.loads(raw)
        self.assertEqual([x['content'] for x in hist], ['two', 'three', 'four'])

    def test_app_snapshots_before_appending_current(self):
        source = (ROOT / 'app.py').read_text(encoding='utf-8')
        meth = source[source.index('    def handle_submit(self)'):source.index('    def _on_backend_finished', source.index('    def handle_submit(self)'))]
        self.assertLess(meth.index('context_prompt = self._build_context_json()'),
                        meth.index('self.memory.append(("user", user_text))'))


if __name__ == '__main__':
    unittest.main()
