#!/usr/bin/env python3
"""One-time full-title lookup build. No Wikipedia article bodies are copied."""
from pathlib import Path
import argparse
import aria_title_index as ti
import aria_wiki_rag as rag

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=Path, default=ti.INDEX_PATH)
    args = parser.parse_args()
    print('Source:', rag.zim_path(), flush=True)
    count = ti.build(rag.zim_path(), args.index, progress=lambda n: print('Indexed:', n, flush=True))
    print('Finished:', count, 'titles; index:', args.index)
