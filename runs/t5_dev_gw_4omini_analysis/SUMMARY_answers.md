# Table 5 - chat luong cau tra loi theo tai lieu (gpt-4o-mini-2024-07-18 qua gateway (phat trien))

Sinh boi `analyze_answers_table5.py` luc 2026-11-08T19:17:19.477538+00:00. So lieu tu file ket qua [Code/Artifact]; dong [Suy luan] la dien giai.
**Chi dung de so sanh noi bo giua cac he thong cung model; khong so voi Table 5 cua paper.**

- Tai lap table5_summary.json: lech lon nhat 1.4e-14.
- Tu choi kem them chu (dem la tu choi): adaptive 0, raw__langch_recurs_default 1, raw__page 0.
- Cau co 10 chunk ngu canh giong het nhau giua hai he thong (bat buoc hoa): {'adaptive~raw__langch_recurs_default': 0, 'adaptive~raw__page': 11, 'raw__langch_recurs_default~raw__page': 0}.

## Trung binh theo tai lieu (33 tai lieu, 95% CI bootstrap)

| He thong | Retrieval Completeness (judge) | Correctness tinh do phu (tu choi = 0) | Ty le tu choi (thap hon = tot hon) |
|---|---:|---:|---:|
| adaptive | 0.914 [0.843, 0.970] | 0.827 [0.762, 0.882] | 0.071 [0.020, 0.141] |
| raw__langch_recurs_default | 0.894 [0.823, 0.949] | 0.770 [0.692, 0.837] | 0.131 [0.061, 0.212] |
| raw__page | 0.884 [0.818, 0.944] | 0.761 [0.679, 0.838] | 0.131 [0.061, 0.212] |

## Adaptive tru doi chung (cap doi theo tai lieu; voi ty le tu choi, am = Adaptive tu choi it hon)

| Metric | Doi chung | Hieu TB | 95% CI | Thang/Hoa/Thua | Rank-biserial | Holm p | MDE (80%) |
|---|---|---:|---|---|---:|---:|---:|
| completeness | raw__langch_recurs_default | +0.0202 | [-0.0202, +0.0606] | 6/24/3 | +0.22 | 0.6954 | 0.060 |
| completeness | raw__page | +0.0303 | [-0.0202, +0.0859] | 6/23/4 | +0.33 | 0.6954 | 0.077 |
| correct_cov | raw__langch_recurs_default | +0.0570 | [+0.0008, +0.1205] | 16/2/15 | +0.24 | 0.5672 | 0.088 |
| correct_cov | raw__page | +0.0662 | [+0.0115, +0.1298] | 14/6/13 | +0.27 | 0.5672 | 0.088 |
| abstain | raw__langch_recurs_default | -0.0606 | [-0.1212, -0.0101] | 1/26/6 | -0.75 | 0.1146 | 0.086 |
| abstain | raw__page | -0.0606 | [-0.1313, +0.0000] | 1/27/5 | -0.76 | 0.1146 | 0.095 |

Holm hieu chinh tren 2 doi chung trong moi metric. Voi 'abstain', 'thang' = Adaptive co ty le tu choi CAO hon.

## Retrieval -> cau tra loi (cau hoi audit; muc cau hoi, gop 3 he thong)

- Spearman recall@10 vs Retrieval Completeness (judge): rho = 0.594 (p = 9.8e-30).
- Spearman recall@10 vs tu choi: rho = -0.587 (p = 6e-29).

| recall@10 | So cau | Ty le tu choi | Completeness | Correctness tinh do phu |
|---|---:|---:|---:|---:|
| =0 | 27 | 0.815 | 0.333 | 0.130 |
| 0-1 | 25 | 0.120 | 0.820 | 0.672 |
| =1 | 245 | 0.033 | 0.967 | 0.870 |

Theo tai lieu: hieu recall@10 (Adaptive - doi chung) co di kem hieu Correctness tinh do phu khong?

| Doi chung | Spearman | p | Tai lieu recall tang | ... va cau tra loi cung tang |
|---|---:|---:|---:|---:|
| raw__langch_recurs_default | 0.651 | 0.000 | 9 | 8 |
| raw__page | 0.377 | 0.030 | 8 | 6 |

## Gioi han

- 33 tai lieu; chenh lech nho hon MDE khong ket luan duoc co hay khong co khac biet.
- Gop 3 he thong o muc cau hoi: cac cau cung tai lieu khong doc lap; Spearman chi mo ta.
- Judge va generator la cung model; chat luong cham bi gioi han boi model nay [Suy luan].
