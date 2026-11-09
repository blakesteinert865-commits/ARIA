#!/usr/bin/env python3
"""Benchmark Aria's real offline Wikipedia retrieval without invoking Qwen.

Produces a small JSONL diagnostics file suitable for inspecting retrieval
coverage before building a fine-tuning dataset. Personal notes are not read.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import aria_wiki_rag as rag

EXAMPLES = [
    ("What is an atom?", False),
    ("Explain what an atom is in detail.", True),
    ("What is photosynthesis?", False),
    ("How does photosynthesis work?", False),
    ("Who was Albert Einstein?", False),
    ("When was Albert Einstein born?", False),
    ("Tell me about the Eiffel Tower.", False),
    ("Bob Hawke", False),
    ("Who was the Australian prime minister in 1986?", False),
    ("Who was the Australian prime minister in 1983?", False),
    ("Who wrote Pride and Prejudice?", False),
    ("When did the Titanic sink?", False),
    ("Elaborate on how stars form.", True),
    ("What's the weather right now?", False),
    ("How are you?", False),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('rag_audit.jsonl'))
    parser.add_argument('--questions', type=Path,
                        help='Optional UTF-8 file containing one question per line')
    args = parser.parse_args()
    if args.questions:
        questions = [(q.strip(), False) for q in args.questions.read_text(encoding='utf-8').splitlines() if q.strip()]
    else:
        questions = EXAMPLES
    if not rag.zim_path().is_file():
        print('Missing archive:', rag.zim_path())
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as output:
        for question, detailed in questions:
            start = time.perf_counter()
            ref = rag.retrieve(question, detailed=detailed)
            milliseconds = round((time.perf_counter() - start) * 1000, 2)
            record = {
                'question': question,
                'detailed': detailed,
                'eligible': rag.is_knowledge_question(question),
                'title': ref.title if ref else None,
                'zim_path': ref.path if ref else None,
                'cached': ref.from_cache if ref else False,
                'latency_ms': milliseconds,
                'passage': ref.passage if ref else None,
            }
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f'{milliseconds:7.1f} ms | {question} -> {record["title"] or "NO MATCH"}')
    print('Wrote:', args.output)
    print('These are retrieval diagnostics, not proof of factual correctness.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
