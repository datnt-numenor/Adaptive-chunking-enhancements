import pandas as pd, json, numpy as np
def rj(p): return pd.DataFrame([json.loads(x) for x in open(p,encoding="utf-8") if x.strip()])
def block(name, pq, systems):
    pq = pq[pq.system_id.isin(systems)]
    r = pq.pivot(index="qa_id", columns="system_id", values="recall@10")[systems]
    print(f"\n== {name}: {len(r)} questions, systems = {systems}")
    print("mean recall@10 per system:", {s: round(r[s].mean(),3) for s in systems})
    print("recall@10 = 0 per system :", {s: int((r[s]==0).sum()) for s in systems})
    print(f"miss (recall=0) in ALL systems: {int((r.max(axis=1)==0).sum())};  in at least one: {int((r.min(axis=1)==0).sum())}")
    print(f"per-question oracle recall@10 (best chunker per question, not attainable): {r.max(axis=1).mean():.3f}")
    print(f"questions where every system has the same recall@10: {int((r.nunique(axis=1)==1).sum())}/{len(r)}")
    r5 = pq.pivot(index="qa_id", columns="system_id", values="recall@5")[systems]
    print("mean recall@5 per system :", {s: round(r5[s].mean(),3) for s in systems})
e = rj("runs/gpu_retrieval_ext_4b_results/extracted_verified/out/ext_4b_part0/evaluation-full/per_query_metrics.jsonl")
block("clair_ext 4B plain", e, ["adaptive@plain","raw__langch_recurs_default@plain","raw__page@plain","sdp_a_t600@plain","sdp_b@plain"])
c = rj("runs/table5_repr_results/table5_sdp/evaluation-full/per_query_metrics.jsonl")
c2 = rj("runs/table5_sdp_results/table5_sdp/evaluation-full/per_query_metrics.jsonl")
c = pd.concat([c, c2[~c2.system_id.isin(c.system_id.unique())]])
block("CLAIR A5000 plain", c, ["adaptive","raw__langch_recurs_default","raw__page","sdp_a_t600","sdp_b"])
block("CLAIR A5000 +doc title", c, ["adaptive__doc","raw__langch_recurs_default__doc","raw__page__doc","sdp_a_t600__doc","sdp_b__doc"])
