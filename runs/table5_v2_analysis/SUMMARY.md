# Table 5 retrieval - bo QA v2 (99 cau, 33 tai lieu)

Sinh tu dong boi `analyze_table5_v2.py` luc 2026-11-08T16:11:36.646036+00:00. Moi so lieu duoi day tinh tu file ket qua [Code/Artifact]; cac dong danh dau [Suy luan] la dien giai.

- Kiem tra: tinh lai trung binh, best_fixed_cv, oracle tu per-query khop retrieval_evaluation.json (lech lon nhat 0.0e+00).
- best_fixed_cv chon theo fold: {0: 'raw__langch_recurs_default', 1: 'raw__langch_recurs_default', 2: 'raw__langch_recurs_default', 3: 'raw__langch_recurs_default', 4: 'raw__langch_recurs_default'} (chi chon giua cac he thong co dinh co trong lan chay: ['raw__langch_recurs_default', 'raw__page']).
- Thiet bi: cuda:0, bfloat16; embedding Qwen/Qwen3-Embedding-4B @ 5cf2132abc99cad020ac570b19d031efec650f2b; reranker Snowflake/snowflake-arctic-embed-l-v2.0.

## Trung binh theo tai lieu (95% CI bootstrap)

| He thong | ndcg@10 | recall@10 | recall@5 | recall@1 | mrr@10 | hit@10 |
|---|---:|---:|---:|---:|---:|---:|
| adaptive | 0.704 [0.628, 0.775] | 0.900 [0.830, 0.957] | 0.798 [0.709, 0.878] | 0.423 [0.323, 0.521] | 0.661 [0.569, 0.746] | 0.929 [0.869, 0.980] |
| raw__langch_recurs_default | 0.689 [0.609, 0.764] | 0.854 [0.778, 0.924] | 0.763 [0.666, 0.850] | 0.437 [0.335, 0.538] | 0.668 [0.578, 0.752] | 0.899 [0.838, 0.960] |
| raw__page | 0.662 [0.586, 0.735] | 0.845 [0.769, 0.913] | 0.750 [0.659, 0.833] | 0.356 [0.260, 0.452] | 0.641 [0.552, 0.726] | 0.899 [0.838, 0.949] |
| best_fixed_cv | 0.689 [0.610, 0.762] | 0.854 [0.778, 0.924] | 0.763 [0.665, 0.851] | 0.437 [0.335, 0.538] | 0.668 [0.577, 0.752] | 0.899 [0.838, 0.960] |
| oracle | 0.781 [0.710, 0.846] | 0.945 [0.891, 0.987] | 0.857 [0.768, 0.930] | 0.513 [0.413, 0.611] | 0.756 [0.678, 0.829] | 0.970 [0.929, 1.000] |

## Adaptive tru doi chung (cap doi theo tai lieu)

| Metric | Doi chung | Hieu TB | 95% CI | Thang/Hoa/Thua | Rank-biserial | Holm p | MDE (80%) |
|---|---|---:|---|---|---:|---:|---:|
| ndcg@10 | raw__langch_recurs_default | +0.0145 | [-0.0294, +0.0603] | 17/1/15 | +0.11 | 1.0000 | 0.066 |
| ndcg@10 | raw__page | +0.0422 | [-0.0103, +0.1026] | 16/8/9 | +0.27 | 0.5745 | 0.082 |
| ndcg@10 | best_fixed_cv | +0.0145 | [-0.0300, +0.0610] | 17/1/15 | +0.11 | 1.0000 | 0.066 |
| ndcg@10 | oracle | -0.0776 | [-0.1093, -0.0460] | 6/5/22 | -0.80 | n/a | 0.046 |
| recall@10 | raw__langch_recurs_default | +0.0456 | [-0.0274, +0.1213] | 9/18/6 | +0.34 | 0.7038 | 0.109 |
| recall@10 | raw__page | +0.0546 | [-0.0068, +0.1230] | 8/22/3 | +0.56 | 0.3469 | 0.095 |
| recall@10 | best_fixed_cv | +0.0456 | [-0.0275, +0.1221] | 9/18/6 | +0.34 | 0.7038 | 0.109 |
| recall@10 | oracle | -0.0453 | [-0.0905, -0.0080] | 1/25/7 | -0.78 | n/a | 0.061 |
| recall@5 | raw__langch_recurs_default | +0.0352 | [-0.0489, +0.1182] | 10/15/8 | +0.22 | 1.0000 | 0.119 |
| recall@5 | raw__page | +0.0478 | [-0.0162, +0.1136] | 11/19/3 | +0.50 | 0.1229 | 0.092 |
| recall@5 | best_fixed_cv | +0.0352 | [-0.0453, +0.1160] | 10/15/8 | +0.22 | 1.0000 | 0.119 |
| recall@5 | oracle | -0.0587 | [-0.1375, +0.0092] | 5/20/8 | -0.41 | n/a | 0.108 |
| recall@1 | raw__langch_recurs_default | -0.0139 | [-0.0916, +0.0617] | 10/10/13 | +0.00 | 1.0000 | 0.112 |
| recall@1 | raw__page | +0.0673 | [-0.0008, +0.1462] | 10/18/5 | +0.49 | 0.4329 | 0.107 |
| recall@1 | best_fixed_cv | -0.0139 | [-0.0922, +0.0638] | 10/10/13 | +0.00 | 1.0000 | 0.112 |
| recall@1 | oracle | -0.0904 | [-0.1521, -0.0279] | 6/9/18 | -0.59 | n/a | 0.090 |
| mrr@10 | raw__langch_recurs_default | -0.0078 | [-0.0609, +0.0476] | 11/7/15 | -0.05 | 1.0000 | 0.078 |
| mrr@10 | raw__page | +0.0201 | [-0.0365, +0.0828] | 11/15/7 | +0.15 | 1.0000 | 0.086 |
| mrr@10 | best_fixed_cv | -0.0078 | [-0.0617, +0.0461] | 11/7/15 | -0.05 | 1.0000 | 0.078 |
| mrr@10 | oracle | -0.0957 | [-0.1401, -0.0519] | 3/10/20 | -0.80 | n/a | 0.065 |
| hit@10 | raw__langch_recurs_default | +0.0303 | [-0.0303, +0.1010] | 5/25/3 | +0.44 | 1.0000 | 0.094 |
| hit@10 | raw__page | +0.0303 | [-0.0303, +0.1010] | 5/25/3 | +0.44 | 1.0000 | 0.094 |
| hit@10 | best_fixed_cv | +0.0303 | [-0.0303, +0.1010] | 5/25/3 | +0.44 | 1.0000 | 0.094 |
| hit@10 | oracle | -0.0404 | [-0.0808, -0.0101] | 0/29/4 | -1.00 | n/a | 0.054 |

Metric chinh: ndcg@10. Holm hieu chinh tren 3 doi chung trong moi metric; oracle la can tren nen chi mo ta, khong kiem dinh.
So doi chung co y nghia (Holm p < 0.05) o ndcg@10: 0/3.
MDE (80%) = muc chenh lech toi thieu phat hien duoc voi power 80%, alpha 0.05, uoc tinh bang xap xi chuan tu do lech chuan cua hieu theo tai lieu [Suy luan: xap xi].

## Phan nhom (ndcg@10, chi mo ta - it tai lieu, khong kiem dinh)

| Nhom theo | Nhom | So tai lieu | Doi chung | Adaptive | Doi chung | Hieu | 95% CI |
|---|---|---:|---|---:|---:|---:|---|
| domain_label | Ada | 9 | raw__langch_recurs_default | 0.802 | 0.784 | +0.018 | [-0.102, 0.146] |
| domain_label | Ada | 9 | raw__page | 0.802 | 0.641 | +0.161 | [0.035, 0.318] |
| domain_label | Ada | 9 | best_fixed_cv | 0.802 | 0.784 | +0.018 | [-0.101, 0.148] |
| domain_label | Norma | 16 | raw__langch_recurs_default | 0.674 | 0.667 | +0.007 | [-0.041, 0.058] |
| domain_label | Norma | 16 | raw__page | 0.674 | 0.666 | +0.008 | [-0.029, 0.051] |
| domain_label | Norma | 16 | best_fixed_cv | 0.674 | 0.667 | +0.007 | [-0.040, 0.059] |
| domain_label | Sophia | 8 | raw__langch_recurs_default | 0.653 | 0.627 | +0.026 | [-0.051, 0.090] |
| domain_label | Sophia | 8 | raw__page | 0.653 | 0.676 | -0.023 | [-0.129, 0.082] |
| domain_label | Sophia | 8 | best_fixed_cv | 0.653 | 0.627 | +0.026 | [-0.048, 0.091] |
| language | en | 27 | raw__langch_recurs_default | 0.721 | 0.692 | +0.029 | [-0.022, 0.082] |
| language | en | 27 | raw__page | 0.721 | 0.664 | +0.057 | [-0.004, 0.127] |
| language | en | 27 | best_fixed_cv | 0.721 | 0.692 | +0.029 | [-0.022, 0.082] |
| language | fr | 6 | raw__langch_recurs_default | 0.627 | 0.678 | -0.051 | [-0.118, 0.016] |
| language | fr | 6 | raw__page | 0.627 | 0.651 | -0.024 | [-0.097, 0.023] |
| language | fr | 6 | best_fixed_cv | 0.627 | 0.678 | -0.051 | [-0.117, 0.016] |
| selected_method | llm_regex | 6 | raw__langch_recurs_default | 0.723 | 0.710 | +0.013 | [-0.092, 0.096] |
| selected_method | llm_regex | 6 | raw__page | 0.723 | 0.730 | -0.007 | [-0.143, 0.159] |
| selected_method | llm_regex | 6 | best_fixed_cv | 0.723 | 0.710 | +0.013 | [-0.091, 0.096] |
| selected_method | our_recurs_1100 | 10 | raw__langch_recurs_default | 0.633 | 0.625 | +0.008 | [-0.079, 0.090] |
| selected_method | our_recurs_1100 | 10 | raw__page | 0.633 | 0.524 | +0.110 | [-0.028, 0.268] |
| selected_method | our_recurs_1100 | 10 | best_fixed_cv | 0.633 | 0.625 | +0.008 | [-0.078, 0.090] |
| selected_method | our_recurs_600 | 1 | raw__langch_recurs_default | 0.877 | 0.810 | +0.067 | [n/a, n/a] |
| selected_method | our_recurs_600 | 1 | raw__page | 0.877 | 0.827 | +0.050 | [n/a, n/a] |
| selected_method | our_recurs_600 | 1 | best_fixed_cv | 0.877 | 0.810 | +0.067 | [n/a, n/a] |
| selected_method | page | 16 | raw__langch_recurs_default | 0.730 | 0.714 | +0.016 | [-0.047, 0.089] |
| selected_method | page | 16 | raw__page | 0.730 | 0.712 | +0.018 | [0.002, 0.044] |
| selected_method | page | 16 | best_fixed_cv | 0.730 | 0.714 | +0.016 | [-0.047, 0.088] |

## Theo loai cau hoi (theo vi tri -1/-2/-3, muc cau hoi, chi mo ta)

| Loai | He thong | ndcg@10 | recall@10 | recall@5 | recall@1 | mrr@10 | hit@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| MULTI-PART | adaptive | 0.741 | 0.911 | 0.783 | 0.351 | 0.736 | 1.000 |
| MULTI-PART | best_fixed_cv | 0.740 | 0.868 | 0.768 | 0.383 | 0.760 | 0.970 |
| MULTI-PART | oracle | 0.816 | 0.945 | 0.903 | 0.375 | 0.822 | 1.000 |
| MULTI-PART | raw__langch_recurs_default | 0.740 | 0.868 | 0.768 | 0.383 | 0.760 | 0.970 |
| MULTI-PART | raw__page | 0.703 | 0.834 | 0.760 | 0.272 | 0.733 | 0.909 |
| PARAPHRASED | adaptive | 0.638 | 0.818 | 0.727 | 0.485 | 0.582 | 0.818 |
| PARAPHRASED | best_fixed_cv | 0.589 | 0.758 | 0.697 | 0.409 | 0.540 | 0.758 |
| PARAPHRASED | oracle | 0.720 | 0.909 | 0.818 | 0.545 | 0.661 | 0.909 |
| PARAPHRASED | raw__langch_recurs_default | 0.589 | 0.758 | 0.697 | 0.409 | 0.540 | 0.758 |
| PARAPHRASED | raw__page | 0.600 | 0.848 | 0.758 | 0.364 | 0.521 | 0.848 |
| STRUCTURE | adaptive | 0.733 | 0.970 | 0.884 | 0.433 | 0.664 | 0.970 |
| STRUCTURE | best_fixed_cv | 0.738 | 0.937 | 0.823 | 0.519 | 0.705 | 0.970 |
| STRUCTURE | oracle | 0.808 | 0.981 | 0.849 | 0.620 | 0.787 | 1.000 |
| STRUCTURE | raw__langch_recurs_default | 0.738 | 0.937 | 0.823 | 0.519 | 0.705 | 0.970 |
| STRUCTURE | raw__page | 0.682 | 0.853 | 0.733 | 0.432 | 0.668 | 0.939 |

## Gioi han

- 33 tai lieu: CI rong; chenh lech nho hon MDE khong the ket luan la co hay khong co khac biet.
- best_fixed_cv chi chon giua cac he thong co dinh co trong lan chay, chua phai moc 'best fixed' day du.
- Phan nhom va loai cau hoi la mo ta, khong hieu chinh boi so sanh boi.
- So voi Table 5 cua nhom: khac bo QA (v2) va lua chon Adaptive (llm_regex khac) [Code/Artifact].
