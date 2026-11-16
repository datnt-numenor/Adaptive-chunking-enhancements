# Báo cáo số liệu: chunker mới SDP so với Adaptive Chunking

Ngày 10/11/2026. Mọi số liệu đều có nhãn **[Paper]**, **[Code/Artifact]** (kèm đường dẫn) hoặc **[Suy luận]**.
"Có ý nghĩa" nghĩa là có ý nghĩa thống kê: Wilcoxon (Pratt) theo tài liệu, hiệu chỉnh Holm, CI bootstrap 10.000 lần
theo tài liệu, seed 2026.

## 0. Tóm tắt

1. **Table 3 (điểm nội tại, trung bình 5 metric):** chunker **SDP-B** hơn Adaptive của paper trên **cả hai bộ dữ liệu**:
   - **+0,90** trên CLAIR (33 tài liệu), CI [0,64; 1,15];
   - **+0,96** trên **bộ test mới gồm 129 tài liệu** chưa từng dùng để chỉnh tham số, CI [0,81; 1,10]. Giả thuyết này đã được đăng ký trước.
2. **Retrieval (recall@10 là metric chính):** SDP **không kém** Adaptive (mọi CI đều chứa 0), nhưng cũng **chưa hơn** một cách có ý nghĩa.
3. **Thêm tên tài liệu vào đầu chunk:**
   - Trên CLAIR, nDCG@10 của Adaptive tăng từ 0,704 lên 0,764 (+0,060, Holm p = 0,002), có ý nghĩa ở cả 6 hệ thống.
   - Trên bộ test, **không lặp lại được**. [Suy luận] Lợi ích phụ thuộc vào việc câu hỏi có nêu tên tài liệu hay không.
4. **Rủi ro "luyện đề":** mức tăng Table 3 của SDP-B chủ yếu đến từ ICC, mà ICC đo bằng chính embedding SDP-B dùng
   để cắt chunk (mục 6).
5. **Table 5 (chất lượng câu trả lời, gpt-4o-mini):** **không có chênh lệch nào có ý nghĩa sau Holm**, trên CLAIR (qua gateway và qua
   API chính thức) cũng như trên bộ test. Trên bộ test, mọi Δ Correctness so với Adaptive nằm trong khoảng ±0,3 điểm. Trên CLAIR,
   SDP-B có ước lượng điểm **thấp hơn** Adaptive: −5,8 điểm Correctness (Holm p = 0,44). Xem mục 4.

## 1. Thiết lập

| Thành phần | Mô tả |
|---|---|
| Baseline | Adaptive (paper): với mỗi tài liệu, chọn phương pháp có trung bình 5 metric cao nhất trong 4 ứng viên (page, our_recurs_1100, our_recurs_600, llm_regex), sau hậu xử lý của paper. Cùng code `find_best_method` |
| SDP-A | Quy hoạch động theo cấu trúc, chỉ cắt tại ranh giới block, độ dài mục tiêu 600 token (t600) |
| SDP-B | Embedding jina-v3 + Dinkelbach, tham số mặc định đặt trước khi xem điểm |
| Hậu xử lý | Giống paper: tách chunk > 1.100 token, gộp chunk < 100 token (trần 1.150) |
| 5 metric | SC, BI, ICC, DCC, RC. Dùng code gốc `compute_metrics_per_origin`, jina-v3 ghim revision `ab036b0`, bf16 |
| Retrieval | Pipeline của harness nhóm: BM25 top-50 + Qwen3-Embedding-4B top-50 → hợp → Snowflake arctic-embed-l-v2.0 xếp hạng lại → top-10; truy hồi trên toàn bộ kho tài liệu; recall = độ phủ ký tự của evidence |
| Thống kê | Đơn vị là tài liệu; Wilcoxon Pratt + Holm; bootstrap 10.000 lần |

### Bộ test mới `clair_ext`, đăng ký trước tại `research\data_expansion\PREREGISTRATION.md`

- [Code/Artifact] **129 tài liệu**, 2,57 triệu token, cùng lĩnh vực với CLAIR (quản trị và chính sách AI): arXiv 108, Norma 11,
  Ada 7, Sophia 3. Parse bằng Docling 2.149 (tắt OCR). Danh sách nằm ở `data\clair_ext_set\set_manifest.json`.
- [Code/Artifact] **Bộ QA:** gpt-4.1-2025-04-14 sinh 774 câu. Còn 621 câu có quote khớp nguyên văn văn bản. Giữ **438 câu
  trên 126 tài liệu**, tức những câu mà **cả** Claude Sonnet 4.5 **và** Gemini 2.5 Pro đều đồng ý giữ. Freeze sha256 `1a7c1d78…`
  (`runs\qa_ext_v1\qa_frozen.jsonl`) **trước** mọi lần chạy retrieval.
- [Code/Artifact] Tham số của SDP (t600, cấu hình B mặc định) được chốt **trước** khi có bộ test. Không chỉnh gì trên bộ test.

## 2. Table 3: điểm nội tại

### 2a. CLAIR (33 tài liệu), [Code/Artifact] `research\structure_chunker\compare.py runs\sdp\B_default runs\sdp\A_target600`

| Phương pháp | RC | ICC | DCC | BI | SC | **Trung bình 5 metric** |
|---|---:|---:|---:|---:|---:|---:|
| page | 97,15 | 69,24 | 86,43 | 99,89 | 99,88 | 90,52 |
| our_recurs_1100 | 98,96 | 66,57 | 89,76 | 98,13 | 99,97 | 90,68 |
| **Adaptive (paper)** | 98,96 | 68,35 | 88,63 | 99,42 | 99,93 | **91,06** |
| SDP-A t600 | 98,81 | 67,30 | 89,28 | 99,94 | 99,94 | 91,05 |
| **SDP-B** | 97,81 | **75,98** | 85,99 | 99,94 | 99,96 | **91,94** |

- SDP-B − Adaptive = **+0,90**, CI [+0,64; +1,15], thắng 31/33 tài liệu, p = 6,4e-7.
- SDP-A t600 − Adaptive = +0,00, CI [−0,20; +0,21], không khác biệt.

### 2b. Bộ test `clair_ext` (129 tài liệu), [Code/Artifact] `runs\ext_prepared_v1\table3_summary.csv`, `table3_comparisons.csv`

| Phương pháp | SC | BI | ICC | DCC | RC | **Trung bình 5 metric** | − Adaptive [CI 95%] | Thắng/thua |
|---|---:|---:|---:|---:|---:|---:|---|---|
| our_recurs_600 | 100,00 | 92,84 | 69,77 | 85,36 | 93,44 | 88,27 | −2,35 [−2,56; −2,15] | 0/129 |
| llm_regex | 99,62 | 97,80 | 70,60 | 86,29 | 94,43 | 89,74 | −0,88 [−1,05; −0,73] | 0/102 |
| our_recurs_1100 | 99,96 | 96,95 | 67,62 | 90,78 | 95,91 | 90,23 | −0,39 [−0,50; −0,30] | 0/76 |
| page | 99,73 | 99,16 | 69,87 | 87,86 | 94,91 | 90,30 | −0,33 [−0,42; −0,24] | 0/65 |
| **Adaptive (paper)** | 99,96 | 98,42 | 69,27 | 89,30 | 96,23 | **90,62** | — | — |
| SDP-A t600 | 99,84 | 99,63 | 70,06 | 89,14 | 95,37 | 90,80 | **+0,17** [0,04; 0,30], p = 3,8e-4 | 88/41 |
| **SDP-B** | 99,92 | 99,63 | **76,96** | 86,80 | 94,63 | **91,58** | **+0,96** [0,81; 1,10], p = 2e-17 | **114/15** |

- **Kết luận H3 (đăng ký trước): xác nhận.** SDP-B ≥ Adaptive, mức tăng gần như bằng trên CLAIR (+0,96 so với +0,90).
- [Code/Artifact] Adaptive chọn page cho 64 tài liệu, our_recurs_1100 cho 53, llm_regex cho 12 (`adaptive_selections.json`).

## 3. Retrieval

### 3a. CLAIR, harness chính thức trên RTX A5000 (99 câu), [Code/Artifact] `runs\table5_repr_results\table5_sdp`

Thống kê ở `runs\table5_repr_analysis_doc_vs_plain\` và `runs\table5_repr_analysis_vs_adaptive\`.

| Hệ thống | recall@10 (thường → thêm tên) | nDCG@10 (thường → thêm tên) | Δ nDCG [CI], Holm p |
|---|---|---|---|
| **Adaptive** | 0,900 → 0,915 | 0,704 → **0,764** | +0,060 [0,031; 0,093], 0,002 |
| LC default | 0,854 → 0,905 | 0,689 → 0,749 | +0,059 [0,022; 0,106], 0,012 |
| page | 0,845 → 0,906 | 0,662 → 0,728 | +0,066 [0,031; 0,100], 0,008 |
| SDP-A t600 | 0,886 → **0,917** | 0,720 → **0,777** | +0,057 [0,022; 0,093], 0,012 |
| SDP-B | 0,840 → 0,892 | 0,677 → 0,722 | +0,045 [0,016; 0,078], 0,012 |
| SDP-Bq | 0,862 → 0,894 | 0,683 → 0,725 | +0,042 [0,017; 0,068], 0,012 |

- recall@10 tăng ở cả 6 hệ thống khi thêm tên tài liệu (+0,015 đến +0,060), nhưng **không mức tăng nào có ý nghĩa sau Holm**.
- So với Adaptive gốc, SDP-A t600 + tên tài liệu có nDCG@10 **+0,073** [0,028; 0,120], Holm p = 0,020. recall@10 +0,017 [−0,038; 0,075].
- So với Adaptive + tên tài liệu (cùng cách biểu diễn), SDP-A t600 + tên tài liệu hơn +0,002 recall@10 và +0,013 nDCG, không có ý nghĩa.
- [Code/Artifact] Lần chạy A5000 trước, dùng SDP-A t800 do CV chọn theo Table 3 (`runs\table5_sdp_analysis`): mọi SDP − Adaptive đều âm,
  CI chứa 0, Holm p = 1.
- [Suy luận] Cả t600 lẫn ý tưởng thêm tên tài liệu đều được tìm ra bằng bản chạy thử trên **chính 99 câu này**. Vì vậy 3a là
  xác nhận bằng model chính thức, **không phải** bằng chứng độc lập. Khi chọn theo 5 fold, t600 được chọn ở 4/5 fold, và kết quả
  (bản chạy thử trên laptop) là recall@10 −0,012 [−0,055; 0,028] so với Adaptive + tên tài liệu.

### 3b. Bộ test `clair_ext`: Qwen3-4B trên H100 (438 câu, 126 tài liệu)

[Code/Artifact] Kết quả ở `runs\gpu_retrieval_ext_4b_results\extracted_verified\out\ext_4b_part0`. Thống kê ở
`runs\ext_4b_analysis_vs_adaptive\` và `runs\ext_4b_analysis_doc_vs_plain\`.

| Hệ thống | recall@10 thường | recall@10 thêm tên | nDCG@10 thường | nDCG@10 thêm tên |
|---|---:|---:|---:|---:|
| Adaptive | 0,853 | 0,853 | 0,689 | 0,690 |
| LC default | 0,844 | 0,849 | 0,695 | 0,698 |
| page | 0,851 | 0,859 | 0,673 | 0,686 |
| SDP-A t600 | 0,852 | 0,858 | 0,693 | **0,706** |
| SDP-B | 0,845 | **0,863** | 0,672 | 0,687 |

(Bảng tính trung bình theo câu. Thống kê tính theo tài liệu.)

- **H1 (SDP-A t600 ≥ Adaptive về recall@10, cùng bản thêm tên tài liệu):** +0,007 [−0,020; +0,036], thắng/hòa/thua 17/95/14,
  Holm p = 1. Ước lượng điểm theo đúng hướng nhưng **không phân biệt được**. SDP-B + tên tài liệu so với Adaptive + tên tài liệu: +0,002 [−0,023; +0,025].
- **H2 (thêm tên tài liệu giúp mọi hệ thống): không lặp lại được.** recall@10 thay đổi từ −0,003 đến +0,014, nDCG@10 từ −0,001 đến +0,012,
  tất cả có Holm p = 1.
- [Code/Artifact, heuristic] Tỉ lệ câu hỏi nhắc ít nhất 50% số từ trong tên tài liệu: CLAIR **52%**, bộ test **26%**. [Suy luận] Thêm
  tên tài liệu chỉ giúp khi câu hỏi nêu tên tài liệu. Đây là một **giới hạn** cần báo cáo, không phải cải tiến chung.

## 4. Table 5: chất lượng câu trả lời

Thiết lập: harness nhóm (`phase_i_table5.py`, `phase_i_judge.py`, DeepEval 3.5.9), sinh câu trả lời và chấm bằng
`gpt-4o-mini-2024-07-18`. Danh sách hệ thống và số câu hỏi được thay trong bộ nhớ bởi `research\table5_ext\t5_multi.py`
(harness trên đĩa giữ nguyên; log ở `<run>\t5_multi_log.jsonl`). [Code/Artifact] Prompt của 3 hệ thống baseline trùng 297/297 với
lần chạy `runs\t5_dev_gw_4omini`.

Thống kê: `research\table5_ext\analyze_t5_multi.py` → `runs\t5_analysis_v1\{clair_gw, clair_official, ext_gw}\`. Từ chối là câu trả lời
**chứa** chuỗi từ chối, giống `phase_i_summarize.py` đã vá. **Correctness (độ phủ)** tính câu từ chối = 0. Mọi Δ tính bằng điểm phần trăm.
[Code/Artifact] Trung bình tính lại khớp với `table5_summary.json` của harness (sai lệch ≤ 3e-14) ở 3 hệ thống mà harness tóm tắt được.
Bước `summarize` của harness lỗi ở các run còn lại vì nó cố định 3 tên hệ thống. Bước chấm thì đã xong đủ ở mọi run.

### 4a. CLAIR (99 câu, 33 tài liệu), qua gateway, `runs\t5_clair_sdp_gw_4omini`

| Hệ thống | Retrieval Completeness | Correctness (câu trả lời) | **Correctness (độ phủ)** | Từ chối |
|---|---:|---:|---:|---:|
| **Adaptive** | 92,4 | 89,1 | **82,8** | **7** |
| LC default | 90,9 | 89,0 | 78,2 | 12 |
| page | 87,9 | 87,7 | 76,2 | 13 |
| Adaptive + tên tài liệu | 93,9 | 89,0 | 82,7 | 7 |
| SDP-A t600 | 91,9 | 90,2 | 80,1 | 11 |
| SDP-A t600 + tên tài liệu | **94,4** | 88,8 | 82,6 | 7 |
| SDP-B | 88,9 | 86,6 | 77,0 | 11 |

| So sánh | Δ Correctness (độ phủ) [CI 95%] | Holm p | Δ số câu từ chối |
|---|---|---:|---|
| SDP-A t600 − Adaptive | −2,7 [−7,2; 1,3] | 1,0 | +4 (Holm p = 0,27) |
| SDP-B − Adaptive | −5,8 [−10,3; −1,9] | 0,44 | +4 (Holm p = 0,27) |
| LC default − Adaptive | −4,6 [−10,9; 1,2] | 1,0 | +5 |
| page − Adaptive | −6,7 [−13,1; −0,9] | 1,0 | +6 |
| Adaptive + tên − Adaptive | −0,1 [−3,7; 3,3] | 1,0 | 0 |
| **SDP-A + tên − Adaptive + tên** (tương tự H1) | −0,2 [−2,6; 2,4] | 0,78 | 0 |

- **Không chênh lệch nào có ý nghĩa sau Holm.** Hai CI bootstrap (SDP-B, page) không chứa 0, nhưng Wilcoxon thì không có ý nghĩa
  (p chưa hiệu chỉnh 0,07 và 0,22). Đây là CI chưa hiệu chỉnh cho nhiều phép so sánh, nên **không** coi là bằng chứng.
- [Suy luận] Ước lượng điểm cho thấy **SDP-B không tốt hơn mà có thể kém hơn** Adaptive về câu trả lời trên CLAIR, dù Table 3 cao hơn
  +0,90. Điều này phù hợp với rủi ro "luyện đề" ở mục 6.
- Thêm tên tài liệu giúp nDCG@10 (mục 3a) nhưng **không** đổi chất lượng câu trả lời (Δ −0,1). [Suy luận] Phù hợp với cơ chế đã biết:
  câu trả lời phụ thuộc vào việc evidence **có** trong top-10, không phụ thuộc thứ hạng bên trong top-10.

### 4b. CLAIR, qua **API chính thức** OpenAI, `runs\t5_clair_sdp_official_4omini`

| Hệ thống | Retrieval Completeness | Correctness (câu trả lời) | **Correctness (độ phủ)** | Từ chối |
|---|---:|---:|---:|---:|
| Adaptive | 92,4 | 88,4 | 81,3 | 8 |
| Adaptive + tên tài liệu | 93,9 | 88,9 | **83,5** | **6** |
| SDP-A t600 + tên tài liệu | 93,9 | 89,2 | 82,0 | 8 |

- Adaptive + tên − Adaptive: +2,3 [−0,9; 5,6], Holm p = 0,31. SDP-A + tên − Adaptive: +0,7 [−3,0; 4,3], Holm p = 0,94.
- SDP-A + tên − Adaptive + tên: −1,5 [−4,9; 1,8], p = 0,34.
- **Kết luận như qua gateway:** không có chênh lệch có ý nghĩa. Không so sánh trực tiếp số của gateway với số của API chính thức.

### 4c. Bộ test (438 câu, 126 tài liệu), qua gateway, 4 run riêng `runs\t5_ext_gw_4omini_*`

| Hệ thống | Retrieval Completeness | Correctness (câu trả lời) | **Correctness (độ phủ)** | Từ chối | Δ độ phủ − Adaptive [CI] | Holm p |
|---|---:|---:|---:|---:|---|---:|
| Adaptive | 91,8 | 87,8 | 76,4 | 57 | — | — |
| SDP-A t600 | 92,1 | 87,9 | 77,3 | 53 | +0,3 [−2,1; 2,5] | 1,0 |
| SDP-B | 92,0 | 87,4 | 76,8 | 53 | +0,2 [−2,2; 2,6] | 1,0 |
| LC default | 90,9 | 87,5 | 76,9 | 53 | +0,2 [−2,2; 2,6] | 1,0 |

- Các hệ thống **không phân biệt được** về câu trả lời (thắng/thua theo tài liệu khoảng 50/50, ví dụ SDP-A 68/56). Đúng như dự đoán
  từ retrieval (mục 3b).
- Các hệ thống dùng bản thường (không thêm tên tài liệu), vì trên bộ test thêm tên không giúp retrieval (H2 không lặp lại).

### 4d. Độ nhiễu của model đánh giá, [Code/Artifact] so sánh `generated_output` từng câu

| So sánh | Câu trả lời giống hệt | Cùng từ chối / không từ chối |
|---|---:|---:|
| Gateway, cùng prompt, chạy lại (`t5_dev_gw_4omini` so với run mới; 3 hệ thống) | 63–71/99 | 97–99/99 |
| Gateway so với API chính thức (3 hệ thống) | 48–58/99 | 98/99 |

- Chạy lại cùng prompt qua gateway làm Correctness (độ phủ) của LC default đổi từ 77,0 lên 78,2 (số câu từ chối từ 13 xuống 12).
  [Suy luận] Nhiễu giữa hai lần chạy khoảng 1 điểm, cùng cỡ với nhiều Δ ở trên. Đây thêm một lý do để không đọc các ước lượng điểm
  nhỏ như một kết quả.

## 5. Kiểm tra độ tin cậy

| Kiểm tra | Kết quả | Bằng chứng |
|---|---|---|
| 5 metric trên laptop (RTX 4060) so với Kaggle | lệch tối đa 7,5e-4 | [Code/Artifact] `score_full.py validate` |
| 5 metric trên pod H100 so với Kaggle | lệch tối đa 8,4e-4 | [Code/Artifact] `runs\gpu_metrics_ext_v1_results\extracted_verified\metrics_out\validate.log` |
| Laptop so với H100 (32 tài liệu SDP-A cùng tính trên hai máy) | lệch tối đa 1,1e-3 | [Code/Artifact] `runs\ext_metrics_merged_v1\merge_manifest.json` |
| Ghép số hai máy | Mọi phương pháp của cùng một tài liệu được tính trên một máy (120 tài liệu trên H100, 9 trên laptop); không thiếu, không trùng | như trên |
| Retrieval 4B trên pod so với A5000 (CLAIR, 4 hệ thống) | recall@10 trùng 98–100% số câu; nDCG r = 0,965–0,990, trung bình thấp hơn 0,006–0,014 | [Code/Artifact] `...\out\clair_check_vs_a5000.txt` |
| Bản chạy thử 0.6B trên laptop so với A5000 (CLAIR, 12 hệ thống) | recall@10 trùng 96–100%, nDCG r = 0,95–0,99 | [Code/Artifact] `research\data_expansion\compare_with_a5000.py` trên `runs\proxy\r1_repr` |
| Bộ test: bản chạy thử 0.6B trên laptop so với 4B trên H100 (10 lượt) | recall@10 trùng 97–100% số câu; kết luận như nhau | [Code/Artifact] `runs\proxy_ext\r1` so với `...\out\ext_4b_part0` |
| Coreference (cho RC) trên bộ test so với mentions gốc của CLAIR | Jaccard 1,00 và 0,56 trên 2 tài liệu kiểm tra | [Code/Artifact] `mentions_ext.py --validate 2` |

## 6. Hạn chế, phải nói rõ khi trình bày

1. **[Suy luận] Luyện đề Table 3.** Ở cả hai bộ dữ liệu, SDP-B tăng ICC khoảng +7,6 đến +7,7 điểm nhưng DCC giảm 2,5–2,6 điểm. ICC được
   đo bằng jina-v3, cũng là embedding SDP-B dùng để chọn điểm cắt. Retrieval không tăng tương ứng, nên chưa thể nói SDP-B làm RAG tốt hơn.
2. **Retrieval không đủ mạnh để phân biệt.** Trên bộ test, SDP-A và Adaptive có recall@10 bằng nhau ở 95/126 tài liệu (khoảng 75%). Mọi chênh lệch
   giữa các chunker đều < 0,02.
3. **RC của bộ test** dùng mentions tự chạy lại. `replicate.py` của paper không chạy được nguyên bản (mục 7), và kiểm tra trên 2 tài liệu
   cho Jaccard 0,56 ở một tài liệu. RC chỉ ảnh hưởng đến lựa chọn của Adaptive và điểm RC; các metric khác không phụ thuộc.
4. **Bộ QA của bộ test do LLM sinh và LLM review** (đã lọc chéo bởi 2 nhà cung cấp, không có người đọc lại). Câu hỏi ít nêu tên tài liệu
   hơn bộ CLAIR.
5. **Table 5 chỉ có một model đánh giá (gpt-4o-mini)**, và với 33 tài liệu thì độ nhạy thấp. Model sinh câu trả lời không tất định:
   chạy lại cùng prompt chỉ cho 63–71% câu trả lời giống hệt (mục 4d). Trên bộ test (126 tài liệu), CI của Δ khoảng ±2,4 điểm,
   nên chỉ phát hiện được chênh lệch lớn hơn mức đó.

## 7. Phát hiện audit mới (bổ sung `AUDIT_REPORT_final.md`)

- **[Code]** `replicate.py:338` gọi `CoreferenceSolver(max_context_tokens=2000)` mà **không** truyền tham số bắt buộc
  `pre_splitter` (`metrics.py:693`), nên bước `mentions` của paper không chạy được nguyên bản. Bản chạy lại dùng pre-splitter
  cắt theo dòng, giữ nguyên mọi ký tự (`research\data_expansion\mentions_ext.py`).
- **[Code]** `DoclingParser.convert_raw_results_to_markdown` chèn PageBreak **sau khi** đã tính offset, nên offset bị lệch.
  Parser của bộ test ghi ngắt trang đúng offset (`research\data_expansion\parse_docling.py`).
- **[Code]** Checkpoint maverick-coref không nạp được với torch ≥ 2.6 (`weights_only`). Đã xử lý bằng cách cho phép đúng các lớp
  omegaconf cần thiết, không tắt kiểm tra an toàn. Cần pin transformers 4.46.3 (bản 5.x lỗi dtype Half/Float).
- **[Artifact]** Card 8 GB hết VRAM (OOM) khi tính DCC cho tài liệu lớn, vì DCC nhúng một cửa sổ 3.000 token cho **mỗi** chunk và
  chiếm 94% thời gian tính metric. Cần GPU ≥ 24 GB, hoặc dùng `research\data_expansion\score_shard.py` để chia việc cho nhiều GPU.

## 8. Tái lập và chi phí

- Code mới nằm trong `research\data_expansion\` và `research\structure_chunker\`. Không sửa `src\` của repo gốc, không sửa harness nhóm.
- Các gói pod: `runs\gpu_metrics_ext_v1\metrics_bundle.zip` (sha `11ee7e88…`), `runs\gpu_retrieval_ext_4b\retrieval_bundle.zip`
  (sha `8177dad6…`). Kết quả trả về được kiểm tra sha256 trước khi dùng.
- **Chi phí API** (giá niêm yết, qua gateway): khoảng **$17,6**, gồm llm_regex $3,7, sinh QA $3,2, review $10,6.
- **GPU thuê:** A5000 (retrieval CLAIR), 2× H100 khoảng 1 giờ (metric), 1× H100 khoảng 1 giờ (retrieval 4B).
- **Chi phí Table 5** ([Code/Artifact] tổng `actual_usd` trong `generation_journal.jsonl` và `judge_journal.jsonl`, tính theo giá khai báo
  0,15/0,60 USD mỗi triệu token):
  - qua gateway: CLAIR **$1,75**, bộ test **$4,49** (4 run: $1,20 / $1,15 / $0,99 / $1,14);
  - qua API chính thức: **$0,79**.
  - Số tiền thực bị trừ trên gateway có thể khác. Hãy xem lại trên trang tài khoản.

## 9. Bước tiếp theo đề xuất

1. Kiểm tra luyện đề: chạy SDP-B với embedding độc lập (`--embed-model qwen0.6b`) rồi chấm lại Table 3.
2. Table 5 đã xong (mục 4) và không phân biệt được các chunker. Nếu muốn có kết luận về câu trả lời, cần nhiều câu hỏi hơn hoặc câu hỏi khó hơn
   (xem bước 3), không phải chạy lại cùng thiết lập.
3. Để retrieval có đủ độ nhạy: thêm câu hỏi khó hơn (evidence nằm ở nhiều nơi), hoặc dùng metric ít bị hòa hơn (recall@5, nDCG).
4. Cho người đọc lại khoảng 50 câu QA của bộ test để ước lượng tỉ lệ lỗi.
