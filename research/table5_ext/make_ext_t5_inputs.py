"""Harness-format inputs for Table 5 on the clair_ext test set (new folder; frozen files only read).
QA: runs/qa_ext_v1/qa_frozen.jsonl -> qa_t5.jsonl with the fields phase_i_table5 expects
(reference_answer = answer, domain = category prefix of doc_name, qa_hash = sha256 of the frozen row,
validation_status = "valid": every kept item passed the quote check + both LLM reviews).
Retrieval: the 4B pod run (proxy format = harness format); evaluation json lists the sha256 of every input.
  python -X utf8 research/table5_ext/make_ext_t5_inputs.py --retrieval runs/gpu_retrieval_ext_4b_results/extracted_verified/out/ext_4b_part0 \
      --systems adaptive@plain,sdp_a_t600@plain,sdp_b@plain,raw__langch_recurs_default@plain --out runs/t5_ext_inputs_v1"""
import argparse, hashlib, json, shutil, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--qa", type=Path, default=Path("runs/qa_ext_v1/qa_frozen.jsonl"))
ap.add_argument("--retrieval", type=Path, required=True)
ap.add_argument("--systems", required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
if a.out.exists():
    sys.exit(f"{a.out} exists")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
rows = [json.loads(x) for x in a.qa.read_text(encoding="utf-8").splitlines() if x.strip()]
out = []
for r in rows:
    out.append({"qa_id": r["qa_id"], "qa_hash": hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                "doc_id": r["doc_id"], "doc_name": r["doc_name"], "domain": r["doc_name"].split("_", 1)[0],
                "question": r["question"], "reference_answer": r["answer"], "evidence": r["evidence"],
                "validation_status": "valid"})
qs = [r["question"] for r in out]
if len(set(qs)) != len(qs):
    sys.exit("duplicate question text (DeepEval resumes by query text)")
(a.out / "retrieval-full").mkdir(parents=True)
qa_path = a.out / "qa_t5.jsonl"
qa_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
hashes = {}
for s in a.systems.split(","):
    dst = a.out / "retrieval-full" / f"{s}.jsonl"
    shutil.copyfile(a.retrieval / "retrieval-full" / f"{s}.jsonl", dst)
    n = sum(1 for _ in dst.open(encoding="utf-8"))
    assert n == len(out), (s, n)
    hashes[s] = sha(dst)
(a.out / "retrieval_evaluation.json").write_text(json.dumps({
    "status": "complete", "qa_count": len(out), "input_qa_sha256": sha(qa_path), "input_retrieval_sha256": hashes,
    "derived_from": {"qa_frozen": str(a.qa), "qa_frozen_sha256": sha(a.qa), "retrieval_run": str(a.retrieval)}},
    indent=2), encoding="utf-8")
print(f"{len(out)} QA, systems {list(hashes)} -> {a.out}")
