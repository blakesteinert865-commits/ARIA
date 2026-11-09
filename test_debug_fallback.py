"""Regression tests for request diagnostics and model-offline Wikipedia fallback."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import aria_backend_bridge as bridge
import crutch


class DebugFallbackTests(unittest.TestCase):
    def test_model_unavailable_returns_matching_wiki_passage_and_records_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            debug_file = Path(tmp) / "aria_debug.json"
            calls = []
            reference = types.SimpleNamespace(
                title="Photosynthesis", path="Photosynthesis", passage="Photosynthesis is the process by which plants convert light into chemical energy.", from_cache=True
            )
            fake_rag = types.ModuleType("aria_wiki_rag")
            fake_rag.is_knowledge_question = lambda question: True
            fake_rag.retrieve = lambda question, detailed=False: reference
            fake_rag.model_reference_context = lambda backend, ref, detailed=False: "SYSTEM WITH WIKI REFERENCE"

            class FakeBackend:
                ModelUnavailable = crutch.ModelUnavailable
                MODEL_URL = "http://localhost:1234/v1/chat/completions?secret=do-not-store"
                MODEL_NAME = "test-model"
                DETAILED_SYSTEM_PROMPT = "detail"
                SYSTEM_PROMPT = "system"
                def route(self, prompt):
                    calls.append(("route", prompt))
                    return None
                def respond(self, prompt, use_model=True):
                    calls.append(("respond", prompt, use_model))
                def wants_detailed_reply(self, prompt):
                    return False
                def ask_model(self, prompt, **kwargs):
                    calls.append(("ask_model", prompt, kwargs))
                    raise crutch.ModelUnavailable("connection refused")
                def plain(self, value, **kwargs):
                    return value

            with patch.dict(os.environ, {"ARIA_DEBUG_FILE": str(debug_file)}), \
                 patch.dict(sys.modules, {"aria_wiki_rag": fake_rag}), \
                 patch.object(bridge, "load_backend", return_value=FakeBackend()), \
                 patch.object(sys, "argv", ["aria_backend_bridge.py", "--prompt", "What is photosynthesis?"]), \
                 redirect_stdout(StringIO()) as output:
                result = bridge.main()

            self.assertEqual(result, 0)
            self.assertIn("Offline Wikipedia result: Photosynthesis", output.getvalue())
            self.assertIn("convert light into chemical energy", output.getvalue())
            self.assertEqual(calls[0], ("route", "What is photosynthesis?"))
            self.assertTrue(any(c[0] == "ask_model" for c in calls))
            record = json.loads(debug_file.read_text(encoding="utf-8"))
            self.assertEqual(record["rag_status"], "matched")
            self.assertTrue(record["context_attached"])
            self.assertEqual(record["model_status"], "unreachable")
            self.assertEqual(record["reply_source"], "wiki_offline_fallback")
            self.assertEqual(record["article_title"], "Photosynthesis")
            self.assertNotIn("secret", record["model_endpoint"])

    def test_no_wiki_match_keeps_clear_unavailable_message_and_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            debug_file = Path(tmp) / "aria_debug.json"
            fake_rag = types.ModuleType("aria_wiki_rag")
            fake_rag.is_knowledge_question = lambda question: True
            fake_rag.retrieve = lambda question, detailed=False: None
            class FakeBackend:
                ModelUnavailable = crutch.ModelUnavailable
                MODEL_URL = "http://localhost:1234/v1/chat/completions"
                MODEL_NAME = "test-model"
                def route(self, prompt): return None
                def respond(self, prompt, use_model=True): pass
                def wants_detailed_reply(self, prompt): return False
                def ask_model(self, prompt, **kwargs): raise crutch.ModelUnavailable("offline")
            with patch.dict(os.environ, {"ARIA_DEBUG_FILE": str(debug_file)}), \
                 patch.dict(sys.modules, {"aria_wiki_rag": fake_rag}), \
                 patch.object(bridge, "load_backend", return_value=FakeBackend()), \
                 patch.object(sys, "argv", ["aria_backend_bridge.py", "--prompt", "What is an unknown topic?"]), \
                 redirect_stdout(StringIO()) as output:
                result = bridge.main()
            self.assertEqual(result, 0)
            self.assertIn("model server isn't reachable", output.getvalue())
            record = json.loads(debug_file.read_text(encoding="utf-8"))
            self.assertEqual(record["reply_source"], "unavailable")
            self.assertFalse(record["context_attached"])


if __name__ == "__main__":
    unittest.main()
