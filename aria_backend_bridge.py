"""Bridge Aria GUI to original Crutch routing and role-separated model history.

Original --prompt goes unmodified to Crutch tools. --context is optional:
  * Recommended: JSON list of {"role":"user"|"assistant", "content":"..."}
  * Backwards compatible: the GUI's 'User: ...' / 'Aria: ...' transcript.
Context is used ONLY when no deterministic tool matches.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path
from aria_response_cleaner import trim_unfinished_sentence

BACKEND_DIR = Path(__file__).resolve().parent
BACKEND_SCRIPT = BACKEND_DIR / "crutch.py"
MODEL_UNAVAILABLE_REPLY = (
    "No tool matched that, and the model server isn't reachable. "
    "Start LM Studio or llama-server, or check CRUTCH_URL."
)
MAX_HISTORY_MESSAGES = 3
MAX_HISTORY_CHARS = 650  # keep prompts compact for a 0.8B model
_ROLES = {"user": "user", "you": "user", "assistant": "assistant",
          "aria": "assistant", "crutch": "assistant", "tool": "assistant"}
_ROLE_LINE = re.compile(r"^\s*(User|You|Assistant|Aria|Crutch|Tool):\s*(.*)$", re.I)
_CONTEXT_FOOTER = re.compile(r"^\s*(?:Reply to the latest User message\.|Recent conversation\b|You are Aria\b)", re.I)


def load_backend():
    if not BACKEND_SCRIPT.is_file():
        raise FileNotFoundError(f"Backend script not found: {BACKEND_SCRIPT}")
    sys.path.insert(0, str(BACKEND_DIR))
    spec = importlib.util.spec_from_file_location("aria_crutch_backend", BACKEND_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load backend script: {BACKEND_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_context(raw: str, latest_prompt: str) -> list[dict[str, str]]:
    """Convert GUI context into *previous* role-separated messages.

    Never feed the GUI's textual framing/instructions to the model as a user
    turn. Unrecognised legacy context is intentionally ignored, not guessed.
    The latest user turn is always added separately by crutch.ask_model().
    """
    raw = (raw or "").strip()
    if not raw:
        return []

    entries = None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            obj = obj.get("messages", obj.get("history"))
        if isinstance(obj, list):
            entries = obj
    except (ValueError, TypeError):
        pass

    history = []
    if entries is not None:
        for item in entries:
            if not isinstance(item, dict):
                continue
            role = _ROLES.get(str(item.get("role", "")).lower())
            content = item.get("content")
            if role and isinstance(content, str) and content.strip():
                history.append({"role": role, "content": content.strip()[:MAX_HISTORY_CHARS]})
    else:
        # Legacy GUI format, including the previous "Recent conversation" wrapper.
        # Only anchored speaker-labelled lines count as conversation turns.
        current = None
        for line in raw.splitlines():
            if _CONTEXT_FOOTER.match(line):
                if current:
                    history.append(current)
                    current = None
                continue
            match = _ROLE_LINE.match(line)
            if match:
                if current:
                    history.append(current)
                current = {"role": _ROLES[match.group(1).lower()],
                           "content": match.group(2).strip()}
            elif current is not None and line.strip():
                current["content"] += "\n" + line.strip()
        if current:
            history.append(current)
        history = [{"role": h["role"], "content": h["content"][:MAX_HISTORY_CHARS]}
                   for h in history if h["content"]]

    # The legacy GUI frequently repeats the current user prompt in --context.
    # The final user turn must occur exactly ONCE in the outgoing API messages.
    if history and history[-1]["role"] == "user" and history[-1]["content"].strip() == latest_prompt.strip():
        history.pop()
    return history[-MAX_HISTORY_MESSAGES:]


def _debug_file() -> Path:
    return Path(os.environ.get("ARIA_DEBUG_FILE", str(BACKEND_DIR / "data" / "aria_debug.json"))).expanduser()


def _write_debug(record: dict) -> None:
    """Persist the latest request trace atomically for the GUI /debug command."""
    path = _debug_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        print(f"[debug] Could not write diagnostics: {type(exc).__name__}: {exc}", file=sys.stderr)


def _safe_endpoint(value: str) -> str:
    """Keep endpoint host/path but never persist credentials or query parameters."""
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if parsed.port:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except (TypeError, ValueError):
        return "<invalid endpoint>"


def main() -> int:
    parser = argparse.ArgumentParser(description="Aria Tools backend bridge")
    parser.add_argument("--prompt", required=True, help="Original user text; passed unchanged to tool routing")
    parser.add_argument("--context", default="", help="Recent GUI history as JSON or legacy transcript")
    parser.add_argument("--stream-events", action="store_true", help="Emit JSON-line events for live GUI streaming")
    args = parser.parse_args()

    def emit(kind: str, value: str) -> None:
        if args.stream_events:
            print(json.dumps({"type": kind, "text": value}, ensure_ascii=False), flush=True)
        elif kind == "final":
            print(value, flush=True)

    trace = {
        "request_id": uuid.uuid4().hex[:12],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": args.prompt[:500],
        "route": "starting", "tool": None,
        "rag_eligible": None, "rag_status": "not_attempted",
        "article_title": None, "article_path": None, "cache_hit": None,
        "passage_chars": 0, "context_attached": False,
        "model_endpoint": None, "model_name": None,
        "model_status": "not_attempted", "model_duration_ms": None,
        "reply_source": None, "fallback_reason": None,
        "errors": [],
    }
    request_started = time.perf_counter()
    reference = None
    try:
        backend = load_backend()
        trace["model_endpoint"] = _safe_endpoint(getattr(backend, "MODEL_URL", ""))
        trace["model_name"] = getattr(backend, "MODEL_NAME", None)
        routed = backend.route(args.prompt)  # Original prompt, always first.
        if routed is not None:
            trace["route"], trace["tool"], trace["reply_source"] = "deterministic_match", routed[1], "crutch"
            emit("final", routed[0])
            return 0

        trace["route"] = "no_deterministic_match"
        # Preserve Crutch's unmatched-question logging without making a second model call.
        backend.respond(args.prompt, use_model=False)
        try:
            detailed = backend.wants_detailed_reply(args.prompt)
            history = parse_context(args.context, args.prompt)
            import aria_wiki_rag
            trace["rag_eligible"] = bool(aria_wiki_rag.is_knowledge_question(args.prompt))
            rag_started = time.perf_counter()
            reference = aria_wiki_rag.retrieve(args.prompt, detailed=detailed)
            trace["rag_duration_ms"] = round((time.perf_counter() - rag_started) * 1000, 2)
            if reference is None:
                trace["rag_status"] = "no_match_or_not_eligible"
            else:
                trace.update({
                    "rag_status": "matched",
                    "article_title": reference.title,
                    "article_path": reference.path,
                    "cache_hit": bool(reference.from_cache),
                    "passage_chars": len(reference.passage),
                })
                system_context = aria_wiki_rag.model_reference_context(
                    backend, reference, detailed=detailed)
                trace["context_attached"] = True
            if os.environ.get("ARIA_RAG_DEBUG"):
                detail = (f"{reference.title!r} cached={reference.from_cache} "
                          f"passage_chars={len(reference.passage)}") if reference else "no accepted article"
                print(f"[rag] {trace.get('rag_duration_ms', 0):.1f} ms | {detail}", file=sys.stderr)
        except Exception as exc:
            system_context = None
            trace["rag_status"] = "error"
            trace["errors"].append(f"RAG {type(exc).__name__}: {exc}")
            if os.environ.get("ARIA_RAG_DEBUG"):
                print(f"[rag] {type(exc).__name__}: {exc}", file=sys.stderr)

        on_delta = (lambda part: emit("delta", part)) if args.stream_events else None
        model_started = time.perf_counter()
        try:
            if trace["context_attached"]:
                raw_reply = backend.ask_model(args.prompt, system=system_context,
                                              detailed=detailed, history=history, on_delta=on_delta)
            else:
                raw_reply = backend.ask_model(args.prompt, detailed=detailed,
                                              history=history, on_delta=on_delta)
            trace["model_status"] = "success"
            reply = backend.plain(raw_reply, max_sentences=0 if detailed else None)
            reply = trim_unfinished_sentence(reply)
            trace["reply_source"] = "model_with_wiki" if trace["context_attached"] else "model"
            emit("final", reply or "(No output returned.)")
            return 0
        except backend.ModelUnavailable as exc:
            trace["model_status"] = "unreachable"
            trace["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            # If the offline keyword/title retrieval found a relevant article,
            # expose that text directly instead of returning an empty failure.
            if reference is not None and reference.passage.strip():
                trace["reply_source"] = "wiki_offline_fallback"
                emit("final", f"Offline Wikipedia result: {reference.title}\n\n{reference.passage}")
            else:
                trace["reply_source"] = "unavailable"
                emit("final", MODEL_UNAVAILABLE_REPLY)
            return 0
        finally:
            trace["model_duration_ms"] = round((time.perf_counter() - model_started) * 1000, 2)
    except Exception as exc:
        trace["route"] = "bridge_error"
        trace["reply_source"] = "error"
        trace["errors"].append(f"Bridge {type(exc).__name__}: {exc}")
        print(f"Bridge Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        trace["total_duration_ms"] = round((time.perf_counter() - request_started) * 1000, 2)
        _write_debug(trace)


if __name__ == "__main__":
    raise SystemExit(main())
