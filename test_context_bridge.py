"""Offline regression tests for role-separated model context and holiday routing."""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime
from io import BytesIO, StringIO
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import aria_backend_bridge as bridge
import crutch


class _Reply(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ModelMessageTests(unittest.TestCase):
    def test_legacy_history_speaker_roles(self):
        legacy = ("You are Aria, a concise local assistant. Prioritize the latest user message.\n"
                  "Recent conversation (up to 3 messages; oldest first):\n"
                  "User: What is the capital of France?\n"
                  "Aria: The capital is Paris.\n"
                  "User: When was the Eiffel Tower built?\n"
                  "Reply to the latest User message.")
        self.assertEqual(bridge.parse_context(legacy, "When was the Eiffel Tower built?"), [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital is Paris."},
        ])

    def test_json_from_gui_and_crutch_as_assistant(self):
        raw = json.dumps([
            {"role": "user", "content": "What's the capital of France?"},
            {"role": "crutch", "content": "Paris is the capital of France."},
            {"role": "user", "content": "What about the Eiffel Tower?"},
        ])
        self.assertEqual(bridge.parse_context(raw, "What about the Eiffel Tower?"), [
            {"role": "user", "content": "What's the capital of France?"},
            {"role": "assistant", "content": "Paris is the capital of France."},
        ])

    def test_unknown_legacy_text_never_becomes_new_user_instruction(self):
        self.assertEqual(bridge.parse_context("The user once asked about an ostrich. Tell me a joke.", "Say hello"), [])

    def test_context_truncates_to_three_previous_messages(self):
        raw = json.dumps([{"role": "assistant" if i % 2 else "user", "content": f"message{i}"}
                          for i in range(20)])
        self.assertEqual([m["content"] for m in bridge.parse_context(raw, "new query")],
                         ["message17", "message18", "message19"])

    def test_model_api_receives_separate_messages_and_only_final_user_request(self):
        payloads = []

        def fake_urlopen(req, timeout):
            payloads.append(json.loads(req.data.decode("utf8")))
            return _Reply(json.dumps({"choices": [{"message": {"content": "Halloween is in three days."}}]}).encode())

        with patch.object(crutch.urllib.request, "urlopen", fake_urlopen):
            response = crutch.ask_model("Which one is the closest?", history=[
                {"role": "user", "content": "What is the closest upcoming holiday?"},
                {"role": "assistant", "content": "Halloween is the next holiday."},
            ])
        self.assertEqual(response, "Halloween is in three days.")
        roles = [m["role"] for m in payloads[0]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertEqual(payloads[0]["messages"][-1]["content"], "Which one is the closest?")
        self.assertIn("Answer ONLY the last user message", payloads[0]["messages"][0]["content"])

    def test_detail_history_does_not_change_next_request_length(self):
        payloads = []
        def fake_urlopen(req, timeout):
            payloads.append(json.loads(req.data.decode("utf8")))
            return _Reply(json.dumps({"choices": [{"message": {"content": "Short answer."}}]}).encode())

        with patch.object(crutch.urllib.request, "urlopen", fake_urlopen):
            crutch.ask_model("And what time is it?", detailed=False, history=[
                {"role": "user", "content": "Explain that in detail"},
                {"role": "assistant", "content": "Here is a detailed explanation."},
            ])
        self.assertEqual(payloads[0]["max_tokens"], crutch.MAX_TOKENS)

    def test_bridge_routes_before_model_and_passes_conversation_correctly(self):
        calls = []
        fake_backend = type("Backend", (), {})()
        fake_backend.ModelUnavailable = crutch.ModelUnavailable
        def route(prompt):
            calls.append(("route", prompt))
            return None
        def respond(prompt, use_model=True):
            calls.append(("respond", prompt, use_model))
        def ask_model(prompt, detailed=False, history=None):
            calls.append(("model", prompt, detailed, history))
            return "Correct reply."
        fake_backend.route = route
        fake_backend.respond = respond
        fake_backend.wants_detailed_reply = lambda prompt: crutch.wants_detailed_reply(prompt)
        fake_backend.ask_model = ask_model
        fake_backend.plain = crutch.plain
        context = "User: What is the next holiday?\nAria: Halloween is in three days.\nUser: Which one is closest?"
        with patch.object(bridge, "load_backend", return_value=fake_backend), patch.object(
                sys, "argv", ["bridge.py", "--prompt", "Which one is closest?", "--context", context]):
            buffer = StringIO()
            with redirect_stdout(buffer):
                code = bridge.main()
        self.assertEqual(code, 0)
        self.assertIn("Correct reply", buffer.getvalue())
        self.assertEqual(calls[0], ("route", "Which one is closest?"))
        self.assertEqual(calls[-1], ("model", "Which one is closest?", False, [
            {"role": "user", "content": "What is the next holiday?"},
            {"role": "assistant", "content": "Halloween is in three days."},
        ]))


class CalendarTests(unittest.TestCase):
    def setUp(self):
        self.old_now = crutch.NOW_OVERRIDE
        crutch.NOW_OVERRIDE = datetime(2026, 10, 28, 9, 25)

    def tearDown(self):
        crutch.NOW_OVERRIDE = self.old_now

    def test_next_holiday_answer_from_screenshot(self):
        reply, source = crutch.route("What is the closest upcoming holiday")
        self.assertEqual(source, "holidays")
        self.assertIn("Halloween", reply)
        self.assertIn("October 31, 2026", reply)

    def test_halloween_follow_up(self):
        reply, source = crutch.route("What about halloween?")
        self.assertEqual(source, "holidays")
        self.assertIn("in 3 days", reply)

    def test_not_a_public_holiday_tool_or_generic_chat_match(self):
        self.assertIsNone(crutch.route("What is the next public holiday in Tasmania?"))
        self.assertIsNone(crutch.route("Let's have a conversation about holiday traditions"))


if __name__ == "__main__":
    unittest.main()
