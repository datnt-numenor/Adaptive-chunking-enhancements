# Recovery audit, 2026-09-11

## Paper says

Không tìm thấy bằng chứng trong paper/code.
This audit addresses research tooling and execution recovery, not a paper claim.

## Code actually does

- The supplied shard-01 controller/per-document logs show both documents 00 and
  01 completed upstream chunking. Document 01 was rejected by the old helper's
  unconditional empty-chunk check. Its raw/no_oversizing page chunk is empty;
  small_merged has no empty chunks.
- The saved Kaggle notebook v17 still had an unconditional empty-chunk assertion
  in its export cell even after the helper fix. Its preflight read `cuda` from
  the wrong JSON level and therefore chose CPU.
- Helper now accepts intermediate empty page chunks; validates methods per
  document; refuses automatic paid reruns of ambiguous partial parquet output;
  and reuses an already complete merged shard even without part checkpoints.
- Corrected notebook embeds the helper to avoid silently mounting an older
  dataset helper. Shard 01 requires two valid existing checkpoints before a
  fresh paid run. Export validates ZIP CRC and reports SHA256.
- Local tests: 14 passed. They include recovery with an intermediate blank page,
  reuse of merged output, and refusal to call a paid runner on partial output.
- Downloaded shard-00 ZIP passes CRC, three-stage checks, 11 documents, seven
  methods per document, and duplicate/empty checks. Stage rows: 6321 / 6785 / 5962.

## Analyst inference and execution boundary

The validator false negative explains the second failure. Existing paid output
must be reused. A private, zero-API Kaggle audit notebook was started to verify
the runtime, all 33 documents with six free methods, synthetic checkpoint tests,
and the export cell. Synthetic seven-method fixtures are software tests only
and must never enter research tables.

Kaggle MCP can access saved versions. At inspection time, v17 saved output had
only free preflight files. The automated browser was unauthenticated and could
not open the private interactive notebook. Resuming the user's live shard-01
run is blocked until its checkpoint files/session are accessible. Starting a
new paid run without those files would risk charging for completed documents.

## Verified completion and current blocker

- Private audit notebook `datnguyenksnb/week2-recovery-audit-no-api` v1: COMPLETE.
  BOOTSTRAP_OK, PREFLIGHT_OK, 14 tests passed, all three real shards (33 documents,
  six free methods) passed, and the corrected export cell passed with synthetic
  seven-method fixtures. No paid API was invoked.
- Main notebook `datnguyenksnb/reproduction-adaptive-chunking` v18: COMPLETE;
  BOOTSTRAP_OK, PREFLIGHT_OK, PAID_SHARD_SKIPPED. Corrected cells are published.
- After the user logged in, the editor showed Draft Session off and No
  persistence. A read-only kernel check on the newly started CPU session found
  only `.virtual_documents` in `/kaggle/working` and no `run_manifest.json`.
  There are no accessible shard-01 checkpoints in that runtime.
- Set notebook persistence to Files only and verified the UI reflected it.
  This does not restore files from the old session.
- Continuing the paid shard without replaying the first two documents requires
  an exported checkpoint directory or another saved copy. No paid rerun started.
