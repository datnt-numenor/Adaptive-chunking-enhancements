"""So sanh cau tra loi Table 5 giua API chinh thuc va gateway (cung prompt, temperature 0).

Doc generation_cache/ cua 2 (hoac 3) thu muc phase_i_table5.py da tao tu CUNG mot ke hoach (requests_sha256 giong nhau):
  --reference  ban chinh thuc          --candidate  ban gateway
  --baseline   (tuy chon) chinh thuc chay lan 2 -> do muc dao dong tu nhien cua GPT-4.1 o temperature 0
So tung yeu cau: prompt_tokens (phai trung khop neu dung model/tokenizer va prompt khong bi sua), ten model,
cau tra loi trung nguyen van, do giong (difflib), cung tu choi tra loi hay khong.
  python -X utf8 compare_providers.py --reference %R%\\table5_v2_answers --candidate %R%\\table5_v2_answers_gateway
      --baseline %R%\\table5_v2_answers_official_rerun --output %R%\\gateway_check.csv
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ABSTAIN = "I don't know based on the provided context."


def load(run_dir: Path) -> tuple[dict, dict]:
    plan = json.loads((run_dir / "generation_plan.json").read_text(encoding="utf-8"))
    cache = {}
    for path in (run_dir / "generation_cache").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        cache[row["request_hash"]] = row
    return plan, cache


def norm(text: str) -> str:
    return " ".join(text.split())


def pair_stats(ref: dict, other: dict, keys: list[str]) -> dict:
    exact = sum(norm(ref[k]["answer"]) == norm(other[k]["answer"]) for k in keys)
    sims = [difflib.SequenceMatcher(None, norm(ref[k]["answer"]), norm(other[k]["answer"])).ratio() for k in keys]
    tokens = sum(ref[k]["prompt_tokens"] == other[k]["prompt_tokens"] for k in keys)
    abstain = sum((ABSTAIN in ref[k]["answer"]) == (ABSTAIN in other[k]["answer"]) for k in keys)
    return {"n": len(keys), "prompt_tokens_match": tokens, "exact_answer": exact,
            "mean_similarity": statistics.mean(sims) if sims else float("nan"),
            "min_similarity": min(sims) if sims else float("nan"), "abstain_agree": abstain,
            "models_other": dict(Counter(other[k]["model"] for k in keys))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        sys.exit(f"{args.output} da ton tai - khong ghi de")

    ref_plan, ref = load(args.reference)
    cand_plan, cand = load(args.candidate)
    if ref_plan["requests_sha256"] != cand_plan["requests_sha256"]:
        sys.exit("Hai thu muc khong cung mot ke hoach (requests_sha256 khac) - prepare lai tu cung dau vao")
    requests = {json.loads(line)["request_hash"]: json.loads(line) for line in
                (args.reference / "generation_requests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    keys = sorted(set(ref) & set(cand) & set(requests))
    if not keys:
        sys.exit("Chua co yeu cau nao duoc tra loi o ca hai ben")

    results = {"gateway_vs_official": pair_stats(ref, cand, keys)}
    if args.baseline:
        base_plan, base = load(args.baseline)
        if base_plan["requests_sha256"] != ref_plan["requests_sha256"]:
            sys.exit("Thu muc baseline khong cung ke hoach")
        bkeys = sorted(set(keys) & set(base))
        results["official_vs_official_rerun"] = pair_stats(ref, base, bkeys)
        results["gateway_vs_official_same_subset"] = pair_stats(ref, cand, bkeys)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["system_id", "qa_id", "prompt_tokens_official", "prompt_tokens_gateway", "model_official",
                         "model_gateway", "exact", "similarity", "answer_official", "answer_gateway"])
        for k in keys:
            r, c = ref[k], cand[k]
            writer.writerow([requests[k]["system_id"], requests[k]["qa_id"], r["prompt_tokens"], c["prompt_tokens"],
                             r["model"], c["model"], norm(r["answer"]) == norm(c["answer"]),
                             round(difflib.SequenceMatcher(None, norm(r["answer"]), norm(c["answer"])).ratio(), 3),
                             r["answer"], c["answer"]])

    print(f"Model ban chinh thuc tra ve: {dict(Counter(ref[k]['model'] for k in keys))}")
    for name, s in results.items():
        print(f"\n{name} ({s['n']} yeu cau): prompt_tokens trung {s['prompt_tokens_match']}/{s['n']}; "
              f"tra loi trung nguyen van {s['exact_answer']}/{s['n']}; do giong TB {s['mean_similarity']:.3f} "
              f"(thap nhat {s['min_similarity']:.3f}); cung tu choi/khong tu choi {s['abstain_agree']}/{s['n']}; "
              f"model ben kia: {s['models_other']}")
    g = results["gateway_vs_official"]
    print("\n[Goi y doc ket qua - suy luan, khong phai kiem dinh]")
    if g["prompt_tokens_match"] < g["n"]:
        print("- prompt_tokens KHONG trung het: gateway co the sua prompt, chen system prompt, hoac dung model/tokenizer "
              "khac. Khong nen dung gateway cho thi nghiem can tai lap.")
    else:
        print("- prompt_tokens trung 100%: prompt den model nguyen ven va cung tokenizer cua GPT-4.1.")
    if "official_vs_official_rerun" in results:
        b, gs = results["official_vs_official_rerun"], results["gateway_vs_official_same_subset"]
        print(f"- Trung nguyen van: chinh thuc vs chinh thuc {b['exact_answer']}/{b['n']}, gateway vs chinh thuc "
              f"{gs['exact_answer']}/{gs['n']}. Neu hai so gan nhau, khac biet cua gateway nam trong muc dao dong "
              "tu nhien; neu gateway thap han ro ret, co the gateway goi model khac.")
    print(f"\nChi tiet tung cau: {args.output}")


if __name__ == "__main__":
    main()
