#!/usr/bin/env python3
"""Opt-in one-time migration to Aria's new single-folder data directory.

Does not delete legacy notes, overwrite existing destination files, or copy
an entire Wikipedia ZIM unnecessarily. The old ZIM is MOVED only with --move-zim.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
LEGACY_CRUTCH = Path.home() / '.crutch'
LEGACY_WIKI = Path.home() / 'Documents' / 'Aria-data' / 'wiki-eng-simple.zim'


def migrate_notes():
    if not LEGACY_CRUTCH.is_dir():
        print(f'No older Crutch data folder found at {LEGACY_CRUTCH}')
        return
    for source in LEGACY_CRUTCH.rglob('*'):
        if not source.is_file() or source.is_symlink():
            continue
        relative = source.relative_to(LEGACY_CRUTCH)
        destination = DATA / relative
        if destination.exists():
            print(f'SKIP existing: {destination}')
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f'Copied: {source} -> {destination}')


def move_zim():
    target = DATA / 'wiki-eng-simple.zim'
    if target.exists():
        print(f'Archive already exists: {target}; no changes made')
        return
    if not LEGACY_WIKI.is_file():
        print(f'Archive not found at old location: {LEGACY_WIKI}')
        print(f'If elsewhere, manually move it into: {target}')
        return
    print(f'Moving archive: {LEGACY_WIKI} -> {target}')
    shutil.move(str(LEGACY_WIKI), str(target))
    print('Archive moved; original path will no longer contain the file.')


def main():
    parser = argparse.ArgumentParser(description='Move Aria/Crutch data into one folder')
    parser.add_argument('--copy-notes', action='store_true', help='Copy ~/.crutch contents, keeping originals')
    parser.add_argument('--move-zim', action='store_true', help='Move the ~1.09 GB archive from ~/Documents/Aria-data')
    args = parser.parse_args()
    if not args.copy_notes and not args.move_zim:
        parser.print_help()
        return
    DATA.mkdir(exist_ok=True, parents=True)
    if args.copy_notes:
        migrate_notes()
    if args.move_zim:
        move_zim()


if __name__ == '__main__':
    main()
