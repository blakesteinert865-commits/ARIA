#!/usr/bin/env python3
"""Install Aria's unfinished-sentence cleaner without replacing crutch.py."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
CRUTCH = HERE / 'crutch.py'
CLEANER = HERE / 'aria_response_cleaner.py'
OLD = '''        return plain(data["choices"][0]["message"]["content"],
                     max_sentences=0 if detailed else None)'''
NEW = '''        from aria_response_cleaner import trim_unfinished_sentence
        response = plain(data["choices"][0]["message"]["content"],
                         max_sentences=0 if detailed else None)
        return trim_unfinished_sentence(response)'''


def main():
    if not CLEANER.is_file():
        raise SystemExit('Missing aria_response_cleaner.py in the same folder.')
    current = CRUTCH.read_text(encoding='utf-8')
    if NEW in current:
        print('Response cleaner is already installed.')
        return
    if current.count(OLD) != 1:
        raise SystemExit('Unexpected crutch.py version. No files changed.')
    revised = current.replace(OLD, NEW, 1)
    compile(revised, str(CRUTCH), 'exec')
    backup = HERE / 'crutch.py.before_cleaner.bak'
    counter = 1
    while backup.exists():
        backup = HERE / f'crutch.py.before_cleaner.{counter}.bak'
        counter += 1
    backup.write_text(current, encoding='utf-8')
    CRUTCH.write_text(revised, encoding='utf-8')
    print('Response cleaner installed successfully.')
    print('Backup:', backup)


if __name__ == '__main__':
    main()
