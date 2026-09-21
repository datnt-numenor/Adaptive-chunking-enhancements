# Week 2 environment snapshot

Snapshot date: 2026-09-10 (Asia/Saigon)

## Source and reference

- Repository: `https://github.com/ekimetrics/adaptive-chunking`
- Checked-out branch: `main`
- Pinned commit: `ea87ce8e1a97888f3f179e7f1359ff7f43fb179d`
- Commit date: 2026-07-06
- Paper: arXiv `2603.25333v1`, 2026-03-26
- Downloaded PDF SHA-256: `E07C80EF3359F699BA67BD7F583C4FDE04604CA15314FCB86763D7939878062B`
- The checked-out code contains three post-publication bug fixes affecting coreference offsets, language filtering, and empty-input handling. Therefore, results from this commit must not be described as bit-for-bit execution of the paper-date code without a second run at commit `21fec93`.

## Runtime detected

- Host shell: Windows PowerShell.
- Default `python`: 3.13.3.
- `py` launcher: did not report registered Python installations.
- Explicit interpreter selected for this audit: Python 3.12.10 at `.venv/Scripts/python.exe`.
- Repository declares Python `>=3.11`, tests 3.11--3.13 in CI, and stores `3.11` in `.python-version`.
- `nvidia-smi` was not on `PATH` and was absent from the two usual Windows locations checked.
- WMI GPU inspection was denied by the managed environment. GPU availability is therefore **not verified**, not proven absent.
- The initial shell snapshot contained no API variables. A later `.env` presence check found `JINA_API_KEY`; its value was never printed. The user separately confirmed a minimal Jina request succeeded.

## Environment tiers

### Tier A: audit/core smoke (installed)

The exact direct package list is in `requirements-core-smoke.txt`. The project was installed editable with `--no-deps`; this keeps the first test cheap and avoids pulling several GB of ML dependencies.

Consequences:

- Unit tests: `54 passed, 1 skipped` (`docling`-dependent test skipped), including low-cost runner and Kaggle filename round-trip tests.
- `pip check`: intentionally reports missing declared dependencies `sentence-transformers` and `spacy`.
- Full CLI chunking probe with both expensive methods disabled reaches the sentence splitter import, then stops at missing `spacy`. This is an environment limitation, not evidence of a source-code defect.
- Importing the Jina client initially failed because `adaptive_chunking/jina_embedder.py` imports `httpx`, while `httpx` is absent from `pyproject.toml`. Installing `httpx==0.28.1` in the audit environment resolved this dependency gap.

### Tier B: exact paper reproduction (not installed yet)

`.[paper]` includes parsing, coreference, Torch 2.6, torchvision, LangChain, Stanza, Haystack, OpenAI, DeepEval, Groq, plotting, and notebook packages. Most dependencies are lower-bounded or unpinned; there is no committed lockfile. Installing today without producing a lock can yield a different environment from the paper.

Before Tier B installation:

1. Verify the actual CUDA driver/runtime outside this restricted shell.
2. Choose Python 3.11 or 3.12 and record it.
3. Resolve once, save the full lock/freeze, and record Hugging Face model revisions.
4. Note that Stanza can download its English tokenizer automatically at runtime.
5. Do not run the OpenAI/Jina paths until the smoke configuration and cost guard are fixed.

## Windows-specific observations

- `python -m adaptive_chunking.paper.replicate --help` raises `UnicodeEncodeError` under the default CP1252 console because the CLI epilog contains Unicode tree characters.
- `python -X utf8 -m adaptive_chunking.paper.replicate --help` succeeds.
- `tiktoken` downloads `o200k_base` on first use; the first sandboxed smoke run failed on that network fetch, then succeeded after the tokenizer was cached.

## Bundled dataset inventory

- 33 parsed JSON documents: 9 technical, 16 legal, 8 social-science.
- Locally counted total: 1,180,526 tokens with `tiktoken==0.14.0`, `o200k_base`.
- Paper total: 1,180,529 tokens. Domain deltas are +2 technical, -7 legal, +2 social-science locally, for a net -3. Treat this as a small but real reproducibility delta until tokenizer/data provenance is resolved.
- All 33 parsed documents have the required keys, monotonic unique split points, and pages that concatenate exactly to `full_text`.
- Mention JSON covers all 33 parsed documents.
- Mention Parquet has two extra document stems not present in `adi_parsed`; they are harmless to the current lookup loop but should be excluded from dataset manifests.
