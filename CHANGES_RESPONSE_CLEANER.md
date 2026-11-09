**This full-release package already has the cleaner integrated in `crutch.py`. Do not run the installer unless working with an older build.**

# Aria: unfinished-response cleaner

This small patch adds `aria_response_cleaner.py` and updates the model-response
path in `crutch.py` so that Qwen responses ending without terminal punctuation
are trimmed back to the last complete sentence. For example:

`Atoms are small. They are made of` becomes `Atoms are small.`

The cleaner preserves ordinary complete replies, common abbreviations,
decimal numbers, and short answers where no safe full-sentence boundary exists.
It acts **only on model-generated answers**, not Crutch or Hassil tool replies.
It does not improve factual accuracy, change token generation speed, or enable
streaming. It cannot reconstruct missing text.

## Install (Fish, Bash, or Zsh)

The archive contains only four new files in `ARIA/`; it does **not** overwrite
your existing `crutch.py`, GUI, title indexes, `data/`, Wikipedia ZIM, or `.venv`.
The installer makes a backup before patching `crutch.py` in place.

```fish
unzip -o ~/Downloads/Aria_Response_Cleaner_Patch.zip -d ~/
cd ~/ARIA
.venv/bin/python install_response_cleaner.py
.venv/bin/python -m unittest -v test_response_cleaner
```

If the installer detects an unexpected existing `crutch.py`, it stops without
changing the code. Running it twice is safe.
