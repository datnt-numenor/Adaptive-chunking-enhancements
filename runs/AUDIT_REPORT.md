# Báo cáo tái lập và audit: Adaptive Chunking (LREC 2026)

> **Trạng thái:** bản nháp. Các số liệu lấy từ log và kết quả do người chạy gửi trong phiên làm việc.
> Trước khi nộp, chạy `runs\verify_report.py` để đối chiếu với artifact. Mục nào ghi **[cần xác nhận]** phải điền hoặc kiểm tra lại.

- **Paper:** *Adaptive Chunking: Optimizing Chunking-Method Selection for RAG*, arXiv 2603.25333v1
- **Code:** `https://github.com/ekimetrics/adaptive-chunking`, commit `ea87ce8e1a97888f3f179e7f1359ff7f43fb179d`
- **Dữ liệu:** 33 tài liệu CLAIR đã parse sẵn (`data/clair/adi_parsed`) cùng mention coreference tính sẵn (`data/clair/mentions`)
- **Người chạy / thời gian:** [cần xác nhận] / 30/09–02/11/2026

Quy ước trình bày theo `AGENTS.md`:
- **Paper nói**: điều paper khẳng định.
- **Code/artifact cho thấy**: kết quả từ code, lệnh đã chạy hoặc file kết quả.
- **Suy luận**: diễn giải của người phân tích.

---

## 1. Tóm tắt

| Kết quả của paper | Kết quả tái lập | Đánh giá |
|---|---|---|
| Table 1: thống kê corpus | Số tài liệu, mean, max, min theo từng lĩnh vực khớp tuyệt đối. Tổng token 1.180.526 so với 1.180.529 | Tái lập được |
| Table 2: kích thước chunk | 6/7 phương pháp khớp tuyệt đối. Sentence và LLM-regex lệch | Tái lập phần lớn |
| Table 3: 5 metric nội tại | Adaptive đạt **91,06** so với 91,07. 6 hàng lệch ≤ 0,01 điểm. Sentence lệch −1,30, LLM-regex lệch +0,40 | Tái lập được |
| Table 4: phương pháp Adaptive chọn | page 16/33 và recursive-600 1/33 khớp. LLM-regex 6/33 so với 2/33, recursive-1100 10/33 so với 14/33 | Tái lập một phần, không ổn định |
| Figure 1: tương quan Spearman | 10/10 hệ số cùng dấu, lệch ≤ 0,07 | Tái lập được |
| Wilcoxon p < 0,001 | 8/8 so sánh có p < 0,001 | Đúng về số, nhưng p nhỏ là tất yếu do cách định nghĩa Adaptive (F5) |
| Table 5: RAG | Chưa chạy | Ngoài phạm vi lần này |

**Phát hiện quan trọng nhất (F6).** Phần lớn lợi thế của cách cắt theo trang trong Adaptive đến từ metric Block Integrity, mà metric này lại gắn với cách parser biểu diễn chỗ ngắt trang:
- Khi bỏ BI khỏi bộ chọn, page chỉ còn được chọn ở 9/33 tài liệu, thay vì 16/33.
- Khi không còn coi chỗ ngắt trang là ranh giới đặc biệt, page được chọn ở 0/33 tài liệu.

---

## 2. Môi trường và chi phí

| Thành phần | Laptop (chunking, phân tích) | Kaggle (metrics) |
|---|---|---|
| Hệ điều hành / phần cứng | Windows, NVIDIA RTX 4060 Laptop 8 GB, driver 537.32 | 2× Tesla T4 |
| Python | 3.11.9 | 3.13.15 |
| torch | 2.6.0+cu124 | 2.11.0+cu128 (bản cài sẵn của Kaggle) |
| Thư viện được khoá phiên bản | transformers 4.57.6, sentence-transformers 5.1.2, tiktoken 0.14.0 | transformers 4.57.6, sentence-transformers 5.1.2 |
| pandas / numpy | 3.0.6 / 2.4.6 | bản cài sẵn của Kaggle [cần xác nhận] |
| stanza | 1.14.0 | — |
| Model metric | — | `jinaai/jina-embeddings-v3` @ `ab036b023d30b4d1138c4c3bfa9f0c445ab455d6`; code `xlm-roberta-flash-implementation` @ `bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3`; không có flash-attn, dùng attention gốc của PyTorch |
| Model LLM-regex | `gpt-4o-2024-08-06` (code gọi `gpt-4o`, temperature 0) | — |

Toàn bộ phiên bản thư viện trên laptop được lưu trong `runs/pip_freeze.txt`.

**Chi phí:**
- **OpenAI:** 33 request, 324.453 token đầu vào và 830 token đầu ra. Ước tính khoảng $0,82 theo giá gpt-4o là $2,50/1M token đầu vào và $10/1M token đầu ra [cần xác nhận số tiền thực trong trang Usage của OpenAI].
- **GPU:** dùng Kaggle miễn phí. Tổng thời gian phiên khoảng 12 giờ, gồm cả hai lần chạy lỗi [cần xác nhận lượng quota đã dùng].

---

## 3. Quy trình đã chạy

Tất cả lệnh chạy từ thư mục gốc repo, với `PYTHONUTF8=1` và `python -X utf8`.

| Bước | Lệnh hoặc script | Output |
|---|---|---|
| Kiểm thử | `pytest -q` → 42 passed, 1 skipped | — |
| Dữ liệu và Table 1 | `runs/explore_data.py` | Màn hình |
| Chunking không tốn phí (6 phương pháp) | `replicate --steps chunking --skip-llm-regex --skip-semantic` | `runs/nosem/` |
| Chunking đầy đủ (7 phương pháp) | `runs/run_chunking_cached.py ... --steps chunking --skip-semantic`. Script này bọc code gốc, gửi từng request OpenAI tuần tự cách nhau 65 giây và cache phản hồi | `runs/full/chunks/`, `runs/openai_cache/` |
| Kiểm tra chunking | `runs/check_chunks.py`, `runs/check_llm_regex.py`, `runs/diagnose_llm_regex.py` | `runs/llm_regex_diagnosis.txt` |
| Đo thử metrics | Tập con 2 tài liệu (17.146 token, 1,45% corpus) trên laptop: 27,7 phút | `runs/subset_out/` |
| Metrics | Kaggle, 4 shard; `metrics` chạy trên GPU 0 và `raw_metrics` trên GPU 1 cùng lúc; ô 4 bản v3 | `runs/kaggle/shard*/` |
| Gộp shard | `runs/merge_kaggle_shards.py` → 2.310 dòng processed và 1.320 dòng raw, 33 tài liệu, không trùng lặp | `runs/full_kaggle/` |
| Table 3 và Adaptive | `runs/table3_adaptive.py`; đối chiếu bằng `replicate --steps analysis table3` | `runs/full_kaggle/audit/` |
| Audit BI và Figure 1 | `runs/bi_audit.py` | `runs/full_kaggle/audit/bi_audit_selections.csv` |

**Thời gian chạy metrics trên Kaggle:**

| Shard | Số tài liệu | Token | `metrics` | `raw_metrics` |
|---|---:|---:|---:|---:|
| 0 | 1 | 391.660 | 167,2 phút | ≤ 167,2 phút (bản v1 không đo riêng từng bước) |
| 1 | 10 | 262.762 | 116,3 | 84,2 |
| 2 | 10 | 262.908 | 125,0 | 102,5 (chạy lại) |
| 3 | 12 | 263.196 | 109,2 | 83,7 |

---

## 4. Kết quả tái lập

### 4.1 Table 1: thống kê corpus (tokenizer `o200k_base`, tiktoken 0.14.0)

| Lĩnh vực | Số tài liệu | Mean | Max | Min | Tổng (bạn) | Tổng (paper) |
|---|---:|---:|---:|---:|---:|---:|
| Tech | 9 | 5.257 | 18.605 | 1.475 | 47.315 | 47.313 |
| Legal | 16 | 30.895 | 118.424 | 523 | 494.313 | 494.320 |
| Social | 8 | 79.862 | 391.660 | 8.013 | 638.898 | 638.896 |

Số tài liệu, mean, max và min khớp tuyệt đối. Tổng token lệch +2 / −7 / +2 token, tức dưới 0,01%. Không tìm thấy bằng chứng trong paper/code để giải thích các chênh lệch này.

### 4.2 Table 2: kích thước chunk

| Phương pháp | Stage | Bạn: mean / max / min / số chunk | Paper |
|---|---|---|---|
| our recursive 1100 | small_merged | 878 / 1141 / 104 / 1345 | giống hệt |
| our recursive 600 | small_merged | 496 / 691 / 102 / 2381 | giống hệt |
| page (hậu xử lý) | small_merged | 663 / 1146 / 72 / 1780 | giống hệt |
| LC recursive 1100 | raw | 706 / 1101 / 2 / 1675 | giống hệt |
| LC recursive default | raw | 773 / 1364 / 57 / 1557 | giống hệt |
| page (raw) | raw | 669 / 1765 / 0 / 1765 | giống hệt |
| **sentence** | raw | **133 / 606 / 4 / 8912** | 146 / 1025 / 4 / 8125 |
| **LLM regex** | small_merged | **622 / 1144 / 78 / 1899** | 518 / 1146 / 69 / 2279 |
| Adaptive (`best` của code gốc) | — | 723 / 1146 / 86 / 1633 | 724 / 1146 / 86 / 1631 |
| semantic | — | không chạy | 693 / 17146 / 1 / 1706 |

### 4.3 Table 3: metric nội tại (%, mean ± std theo tài liệu)

| Phương pháp | RC | ICC | DCC | BI | SC | Mean | Paper | Chênh |
|---|---|---|---|---|---|---:|---:|---:|
| \* LLM regex | 97,9±3,0 | 70,4±5,7 | 84,1±5,6 | 98,7±1,6 | 99,8±0,8 | 90,20 | 89,80 | +0,40 |
| \* our recursive 1100 | 99,0±1,8 | 66,6±3,8 | 89,8±2,5 | 98,1±1,9 | 100,0±0,1 | 90,68 | 90,68 | 0,00 |
| \* our recursive 600 | 97,2±2,9 | 69,5±4,1 | 84,7±3,1 | 94,8±3,6 | 100,0±0,0 | 89,24 | 89,24 | 0,00 |
| \* page (hậu xử lý) | 97,2±3,5 | 69,2±3,6 | 86,4±4,6 | 99,9±0,3 | 99,9±0,4 | 90,52 | 90,52 | 0,00 |
| † LC recursive 1100 | 98,4±3,1 | 64,7±3,4 | 86,8±2,8 | 98,6±1,4 | 93,3±7,2 | 88,36 | 88,37 | −0,01 |
| † LC recursive default | 96,1±4,3 | 65,6±3,6 | 88,8±2,4 | 95,0±2,8 | 97,7±6,3 | 88,62 | 88,62 | 0,00 |
| † page (raw) | 97,1±3,5 | 69,3±3,4 | 86,1±5,0 | 100,0±0,0 | 92,7±9,7 | 89,03 | 89,03 | 0,00 |
| † semantic | — | — | — | — | — | — | 76,49 | không chạy |
| † sentence | 86,6±10,4 | 78,6±2,1 | 72,0±5,9 | 60,9±9,2 | 61,7±21,5 | 71,96 | 73,26 | −1,30 |
| **Adaptive (4 ứng viên)** | 99,0±1,5 | 68,3±4,7 | 88,6±3,6 | 99,4±1,2 | 99,9±0,3 | **91,06** | 91,07 | −0,01 |

`runs/table3_adaptive.py` và lệnh `table3` của code gốc cho **cùng một con số** ở cả 8 hàng phương pháp. Hàng `best` trong Table 1 của code gốc cũng trùng với hàng Adaptive ở trên.

### 4.4 Table 4: tỷ lệ phương pháp được Adaptive chọn

| Phương pháp | Bạn | Paper (%) | Paper quy ra số tài liệu |
|---|---:|---:|---:|
| page | 16/33 (48,5%) | 48 | 16 |
| our recursive 1100 | 10/33 (30,3%) | 42 | 14 |
| LLM regex | 6/33 (18,2%) | 6 | 2 |
| our recursive 600 | 1/33 (3,0%) | 3 | 1 |

### 4.5 Figure 1: tương quan Spearman (212 cặp tài liệu × phương pháp, kết quả đã hậu xử lý)

| Cặp | RC–ICC | RC–DCC | ICC–DCC | RC–BI | ICC–BI | DCC–BI | RC–SC | ICC–SC | DCC–SC | BI–SC |
|---|---|---|---|---|---|---|---|---|---|---|
| Bạn | −0,20 | +0,30 | −0,46 | +0,35 | −0,28 | +0,25 | +0,08 | +0,17 | −0,20 | +0,04 |
| Paper | −0,21 | +0,24 | −0,44 | +0,31 | −0,34 | +0,31 | +0,04 | +0,12 | −0,16 | +0,02 |

Paper không ghi rõ Figure 1 được tính trên tập dữ liệu nào. Ở đây giả định là kết quả của chunk đã hậu xử lý.

---

## 5. Các phát hiện audit

### F1. Pipeline metric nội tại tái lập được
- **Paper nói:** Adaptive đạt 91,07, cao nhất trong Table 3.
- **Code/artifact cho thấy:** kết quả là 91,06. 6 phương pháp tất định lệch ≤ 0,01 điểm. Table 1, Table 2 (phần tất định) và Figure 1 đều khớp. Embedding được tính bằng torch 2.11 trên GPU T4, khác torch 2.6.0 mà repo pin, nhưng kết quả không bị ảnh hưởng đáng kể.
- **Suy luận:** code, dữ liệu và metric đủ để tái lập con số metric nội tại của paper.

### F2. Sentence chunking không tái lập được
- **Paper nói:** sentence chunking gom 5 câu thành một chunk, dùng Stanza (§2.2). Table 2 ghi 8.125 chunk, max 1.025 token. Table 3 ghi mean 73,26.
- **Code/artifact cho thấy:**
  - `replicate.py:172` gọi `SentenceSplitter(method="stanza", sentences_per_chunk=5)`, đúng như paper mô tả.
  - Từ commit phát hành `21fec93` đến `ea87ce8`, không có commit nào sửa `paper/splitters.py` hay `replicate.py`.
  - Kết quả chạy lại: 8.912 chunk, max 606 token, mean 71,96 (lệch −1,30).
- **Suy luận (chưa kiểm chứng):** nhiều khả năng do phiên bản Stanza hoặc model tách câu của Stanza khác với lúc paper chạy, vì `pyproject.toml` chỉ khai báo `stanza>=1.10.1`. Có thể kiểm chứng bằng cách chạy lại chunking với `stanza==1.10.1`.

### F3. LLM-regex: khác model và âm thầm dùng phương án dự phòng
- **Paper nói:** LLM-regex dùng GPT-5 (Table 2, 3). LLM trả về regex trong thẻ `<regex>…</regex>` (Appendix B). Paper không nói gì về trường hợp sinh regex thất bại.
- **Code/artifact cho thấy:**
  - `run_chunking()` mặc định dùng `gpt-4o`, và CLI không có tham số để đổi model.
  - Với `gpt-4o-2024-08-06`, **7/33 tài liệu (21%)** không tách được. Lý do: LLM trả regex trong khối ```` ```regex ```` thay vì thẻ `<regex>`, nên `extract_llm_regex` (`paper/splitters.py:557`) trả về `None`. Khi đó `split_text` trả nguyên tài liệu thành 1 chunk mà không cảnh báo gì (`paper/splitters.py:551-552`). Bước hậu xử lý sau đó cắt tài liệu bằng recursive splitter.
  - Bản thân các regex trong 7 trường hợp này đều hợp lệ: chúng tách theo tiêu đề `#` và `##`.
  - Tài liệu bị ảnh hưởng: Ada *Standards - Interpretability*; Norma *EDPS Generative AI guidelines*; Sophia *Consultation paper on AI regulation*, *Livre blanc - Impact AI*, *RAPPORT CI INTELLIGENCE ARTIFICIELLE*, *Regulating under Uncertainty* (tài liệu dài nhất corpus) và *Responsible AI*.
  - Trong 6 tài liệu Adaptive chọn LLM-regex, **có 3 tài liệu thuộc nhóm thất bại này**: *Standards - Interpretability*, *Consultation paper…* và *Responsible AI…*.
- **Suy luận:**
  - Hàng LLM-regex trong Table 2, 3, 4 thực chất trộn regex thật với recursive splitter.
  - Tỷ lệ thất bại phụ thuộc vào model. Đây là điểm yếu về độ bền của phương pháp, và paper không báo cáo.
  - Một thí nghiệm không tốn phí, vì mọi phản hồi đã được cache: cho bộ đọc chấp nhận cả khối ```` ``` ````, rồi đo lại.

### F4. Phân bố lựa chọn của Adaptive không ổn định
- **Paper nói:** LLM-regex được chọn cho 6% tài liệu, recursive-1100 cho 42% (Table 4).
- **Code/artifact cho thấy:** LLM-regex được chọn 18,2%, recursive-1100 được chọn 30,3%. Có 4 tài liệu đổi lựa chọn, nhưng điểm Adaptive vẫn là 91,06.
- **Suy luận:** điểm tổng ổn định, nhưng việc Adaptive chọn phương pháp nào phụ thuộc vào output của LLM. Vì vậy khi giải thích Table 4 về mặt nội dung tài liệu, ví dụ "tài liệu luật hợp với regex", cần thận trọng.

### F5. Kiểm định Wilcoxon cho p nhỏ là điều tất yếu
- **Paper nói:** Adaptive khác mọi phương pháp với Wilcoxon p < 0,001 (chú thích Table 3).
- **Code/artifact cho thấy:**
  - Chạy lại kiểm định Wilcoxon theo cặp tài liệu: cả 8 so sánh đều có p < 0,001. Mức hơn trung bình so với ứng viên tốt nhất (recursive-1100) chỉ là +0,39 điểm, với 20/33 tài liệu có chênh lệch khác 0.
  - Code công khai không có phần kiểm định này.
- **Suy luận:**
  - Adaptive được định nghĩa là ứng viên có điểm **cao nhất** trên chính các metric đem ra so sánh. Vì vậy so với 4 ứng viên, chênh lệch theo từng tài liệu luôn ≥ 0, và p nhỏ là hệ quả của định nghĩa, không phải bằng chứng độc lập.
  - Mức cải thiện thực tế rất nhỏ, và chỉ là cải thiện trên metric nội tại, chưa phải trên RAG.

### F6. Block Integrity thiên vị cách cắt theo trang
- **Paper nói:** "structural blocks rarely cross page boundaries" (§3.1). Page được chọn cho 48% tài liệu, và paper giải thích là do "pagination semantics" (§3.4).
- **Code/artifact cho thấy:**
  - BI dùng `split_points` do parser sinh ra (`metrics.py:280-323`).
  - 1.738/1.761 dấu `<!-- PageBreak -->` nằm cách một ranh giới block không quá 5 ký tự. Corpus có 1.732 chỗ chuyển trang.
  - BI tự tính lại khớp với BI của code gốc (chênh lệch tối đa 0).
  - So sánh 3 biến thể của bộ chọn:

| Biến thể | llm_regex | rec-1100 | rec-600 | page | Số tài liệu đổi |
|---|---:|---:|---:|---:|---:|
| A. Gốc: 5 metric, trọng số bằng nhau | 6 | 10 | 1 | 16 | — |
| B. Bỏ BI: 4 metric | 6 | 16 | 2 | 9 | 9 |
| C. BI trung lập với trang: bỏ 3.470 ranh giới tại 1.732 chỗ ngắt trang | 14 | 18 | 1 | 0 | 16 |

  - Điểm BI của page giảm từ 99,9 xuống 75,0 khi chuyển sang cách tính trung lập. Các ứng viên khác chỉ giảm 2,9–5,1 điểm.
- **Suy luận:**
  - Phần lớn lợi thế của page đến từ cách parser biểu diễn chỗ ngắt trang, chứ không có bằng chứng độc lập về "pagination semantics".
  - Biến thể C là kịch bản xấu nhất cho page, vì nó coi cả những chỗ đoạn văn thật sự kết thúc ở cuối trang là nằm giữa block. Mức thiên vị thật nằm giữa B và C.
  - Đây là kết luận trên metric nội tại. Muốn biết tác động lên RAG thì cần đánh giá retrieval.

### F7. Tập ứng viên trong code khác với paper
- **Paper nói:** Adaptive chỉ chọn trong 4 phương pháp đánh dấu `*` (Table 2, 3).
- **Code/artifact cho thấy:** `replicate.py:662` gọi `output_best_chunks` mà không truyền `candidate_methods`. Bước `analysis` vì vậy chọn `best` trong mọi phương pháp có metric, ở lần chạy này là 7 phương pháp. Kết quả `best` vẫn trùng hoàn toàn với Adaptive 4 ứng viên.
- **Suy luận:** trong lần chạy này sai lệch không làm đổi số liệu, nhưng nó có thể làm đổi kết quả với dữ liệu hoặc phương pháp khác.

### F8. Prompt cho bước RAG trong code khác Appendix của paper
- **Paper nói:** prompt sinh QA được trình bày ở Appendix C, prompt sinh câu trả lời ở Appendix E.
- **Code/artifact cho thấy:**
  - Nội dung `qa_generation_prompt` (`replicate.py:683-692`) và `qa_prompt` (`replicate.py:745-752`) khác hẳn hai Appendix trên. Không tìm thấy văn bản của Appendix C hay E ở bất kỳ đâu trong repo.
  - Prompt chấm Retrieval Completeness (`rag_eval.py:41`) thì khớp với Appendix D.
  - Bước `rag` còn lấy baseline từ `small_merged`, tức chunk đã hậu xử lý (`replicate.py:672-676`), trong khi paper ghi là chunk thô.
- **Suy luận:** chạy `--steps rag` sẽ không tái lập đúng protocol của Table 5. Phần này chưa chạy trong lần tái lập này.

### F9. Các lỗi kỹ thuật gặp khi chạy (không ảnh hưởng đến số liệu)

| Vấn đề | Bằng chứng | Cách xử lý |
|---|---|---|
| Repo không khai báo `pyarrow` (dùng để ghi Parquet) và `httpx` (dùng trong `jina_embedder.py`) | Lỗi import khi chạy | Cài thêm hai gói này |
| `maverick-coref` không cài được trên Windows | `UnicodeDecodeError` khi `setup.py` đọc README bằng bảng mã cp1252 | Không cài, vì mention đã có sẵn |
| pandas 3: `np.fill_diagonal(m.values, ...)` báo lỗi read-only | `analysis.py:822` | Sửa thành gán `m.iat[i, i]`; công thức không đổi |
| `_make_embedder` nạp model vào GPU 0 trước rồi mới chuyển sang `--device` | `replicate.py:364-365`; lỗi OOM ở shard 2 | Mỗi tiến trình chỉ được thấy 1 GPU qua `CUDA_VISIBLE_DEVICES` |
| Code gửi 33 request OpenAI cùng lúc và không cache | `split_documents.py:133` | Bọc code: gửi tuần tự, cache phản hồi |
| Kaspersky can thiệp HTTPS làm Python báo lỗi SSL | `CERTIFICATE_VERIFY_FAILED` | Đặt python.exe làm ứng dụng tin cậy, không tắt xác minh SSL |

---

## 6. Sai lệch so với protocol, và các lần chạy lỗi

- **Không chạy semantic chunking**, vì Windows không có flash-attn và GPU T4 không hỗ trợ flash-attn 2. Semantic không phải ứng viên của Adaptive, nên chỉ thiếu hàng semantic ở Table 2 và Table 3.
- **LLM-regex dùng `gpt-4o-2024-08-06`**, không phải GPT-5 như paper.
- **Metrics tính trên Kaggle**, chia 4 shard rồi gộp lại. Môi trường là torch 2.11 với Python 3.13, khác chunking chạy trên laptop với torch 2.6 và Python 3.11.
- **Đã sửa 1 chỗ trong code gốc** (`analysis.py`, vòng gán đường chéo ma trận tương quan) để tương thích pandas 3.
- **Các lần chạy lỗi, không đưa vào kết quả:**
  1. Shard 1, ô 4 bản v1: `raw_metrics` lỗi `FileNotFoundError mlp.py` do hai tiến trình cùng tải code model một lúc. Đã chạy lại thành công.
  2. Shard 2, ô 4 bản v2: `raw_metrics` hết bộ nhớ GPU (OOM) do F9. Đã chạy lại riêng `raw_metrics` bằng bản v3. Phần `metrics` của lần chạy lỗi vẫn được dùng, vì bước đó trả `returncode` 0.

## 7. Giới hạn

- Kết luận ở đây chỉ dựa trên **metric nội tại**. Chưa đánh giá retrieval hay chất lượng câu trả lời (Table 5).
- Corpus chỉ có 33 tài liệu, và không có tập dữ liệu độc lập để kiểm chứng.
- Chưa kiểm chứng giả thuyết Stanza ở F2.
- Biến thể BI trung lập (F6) chỉ là một phép thử độ nhạy, không phải định nghĩa BI mới.

## 8. Đề xuất tiếp theo

1. Chạy lại LLM-regex với bộ đọc chấp nhận cả khối ```` ``` ````, dùng lại cache nên không tốn phí. Mục tiêu: đo F3 ảnh hưởng thế nào đến Table 3 và Table 4.
2. Chạy lại chunking với `stanza==1.10.1` để kiểm chứng F2.
3. Đánh giá retrieval cho 3 biến thể A, B, C của F6, để biết thiên vị của BI có làm retrieval kém đi không. Việc này cần bộ QA đã freeze và GPU, nên cần thống nhất với nhóm trước.

## Phụ lục: artifact

| File | Nội dung |
|---|---|
| `runs/pip_freeze.txt` | Phiên bản thư viện trên laptop |
| `runs/openai_cache/usage.jsonl` | Token của từng request OpenAI |
| `runs/llm_regex_diagnosis.txt` | Nguyên nhân 7 tài liệu LLM-regex thất bại |
| `runs/full/chunks/` | Chunk của 7 phương pháp × 3 stage |
| `runs/kaggle/shard*/out/` | Output, log, `run_status.json` và `shard_manifest.json` của từng shard |
| `runs/full_kaggle/results*/` | Metric đã gộp |
| `runs/full_kaggle/audit/` | `table3_local_vs_paper.csv`, `adaptive_selection.csv`, `bi_audit_selections.csv`, `upstream_table3.log` |







------------Output từ claude----------------
Tôi đã đối chiếu xong. Gần như mọi con số trong bản nháp đều khớp với artifact. Báo cáo có 3 chỗ sửa, không đụng vào src/ và phần "Suy luận".

Về cách chạy: Lệnh python -X utf8 runs/verify_report.py báo lỗi vì Python hệ thống không có pandas, nên tôi chạy bằng .venv\Scripts\python.exe. Để kiểm thêm các số mà verify_report.py không in ra, tôi chạy lại explore_data.py và check_chunks.py runs/full. Hai script table3_adaptive.py và bi_audit.py có ghi CSV, nên tôi chạy chúng trên một bản sao trong scratchpad để không ghi đè artifactừng byte với runs/full_kaggle/audit/ (lệnh
 lập về "pagination semantics".                                                                                - Biến thể C là kịch bản xấu nhất chỗ đoạn văn thật sự kết thúc ở cuối
trang là nằm giữa block. Mức thiên vị thật nằm giữa B và C.

  Ran 1 shell command

Tôi đã đối chiếu xong. Gần như mọi con số trong bản nháp đều khớp với artifact. Báo cáo có 3 chỗ sửa, không đụng vào  src/ và phần "Suy luận".

Về cách chạy: Lệnh python -X utf8 runs/verify_report.py báo lỗi vì Python hệ thống không có pandas, nên tôi chạy bằng .venv\Scripts\python.exe. Để kiểm thêm các sn ra, tôi chạy lại explore_data.py vàcheck_chunks.py runs/full. Hai script table3_adaptive.py và bi_audit.py có ghi CSV, nên tôi chạy chúng trên một bản sao trong scratchpad để không ghi đè artifact gốc. Các CSV sinh ra trùng từng byte với runs/full_kaggle/audit/ (lệnh fc báo "no differences").

Các chỗ đã sửa trong runs/AUDIT_REPORT.md

┌───────────────────┬──────────────────────────────┬──────────────────┬──────────────────────────────────────────┐
│      Vị trí       │            Trước             │       Sau        │                  Nguồn                   │
├───────────────────┼──────────────────────────────┼──────────────────┼──────────────────────────────────────────┤
│ Mục 2, dòng torch │ 2.6.0+cu124 [cần xác nhận    │ 2.6.0+cu124      │ pip_freeze.txt: torch==2.6.0+cu124       │
│                   │ trong pip_freeze.txt]        │                  │                                          │
├───────────────────┼──────────────────────────────┼──────────────────┼──────────────────────────────────────────┤
│ Mục 2, dòng       │ [cần xác nhận: pip show      │ 1.14.0           │ pip_freeze.txt: stanza==1.14.0           │
│ stanza            │ stanza]                                                       │
├───────────────────┼──────────────────────────────┼──────────────────┼──────────────────────────────────────────┤
│ F6, BI của các    │                       i_audit.py: llm_regex 98,7→93,7;        │
│ ứng viên khác     │ "chỉ giảm 3–5 điểm"          │ 2,9–5,1 điểm"    │ rec-1100 98,1→95,2; rec-600 94,8→89,7    │
│ page              │                              │                  │                                          │
└───────────────────┴──────────────────────────────┴──────────────────┴──────────────────────────────────────────┘

Các mục [cần xác nhận] vẫn để nguyên (không

- Tên người chạy (dòng 9).
- pandas/numpy trên Kaggle: log và run_statu
- Số tiền OpenAI thực trả. Phần ước tính đã kiểm lại: 324.453 × $2,5/1M + 830 × $10/1M ≈ $0,82.
- Quota GPU Kaggle. Theo giờ start/end trong 5 file run_status.json, tổng thời gian là khoảng 10,3 giờ. Lần chạy lỗi của shard 1 (ô 4 bản v1) không để lại run_ng 12 giờ" vừa không kiểm được vừa không bịartifact bác bỏ. Tôi chưa sửa.

Đã khớp với artifact

- Môi trường:
  - Laptop: Python 3.11.9; transformers 4.57.6, sentence-transformers 5.1.2, tiktoken 0.14.0; pandas 3.0.6 / numpy 2.4.6; có pyarrow và httpx.
  - Kaggle: 2× T4, torch 2.11.0+cu128, Python 3.13.15. Revision của jina-v3 và xlm-roberta-flash giống nhau ở cả 5 shard. Log ghi "flash_attn is not installed".
- OpenAI: 33 request, model gpt-4o-2024-08-06, 324.453 token vào / 830 token ra.
- Shard:
  - Số tài liệu 1/10/10/12; token 391.660 / 262.762 / 262.908 / 263.196, cộng lại đúng 1.180.526.
  - Các thời gian chạy đều khớp. Shard 2 có raw_metrics rc=1, và log ghi torch.OutOfMemoryError.
  - Dữ liệu gộp: 2.310 dòng processed và 1.320 dòng raw, 33 tài liệu, không trùng lặp.
  - Tập con đo thử có 17.146 token, tức 1,45% corpus.
- Bảng kết quả:
  - Table 1: khớp từng ô; tổng token của paper là 1.180.529.
  - Table 2: khớp từng ô, kể cả hàng best 72
  - Table 3: khớp mean ± std của cả 9 hàng, cũng khớp upstream_table3.log.
  - Table 4: 16/10/6/1.
  - Figure 1: khớp 10/10 hệ số, tính trên 21

  - F3: đúng 7 tài liệu thất bại, trong đó 3 tài liệu được Adaptive chọn LLM-regex.
  - F5: 8/8 so sánh có p < 0,001; Adaptive hơn recursive-1100 trung bình +0,39 điểm, 20/33 tài liệu có chênh lệch khác 0.
  - F6: 1.738/1.761 dấu ngắt trang, 1.732 chỗ chuyển trang, bỏ 3.470 ranh giới, BI tự tính lệch tối đa 0, bảng A/B/C khớp, page giảm 99,9→75,0.

Không kiểm được từ artifact

Các số sau tôi giữ nguyên vì không có file nào ghi lại để đối chiếu:
- kết quả pytest 42 passed / 1 skipped;
- driver 537.32;
- 27,7 phút đo thử trên laptop;
- "4 tài liệu đổi lựa chọn" ở F4;
- các số liệu của paper;
- số dòng trích dẫn trong src/.

git không có trong PATH nên tôi chưa chạy được git status. Ngoài báo cáo, tôi không ghi file nào trong runs/.