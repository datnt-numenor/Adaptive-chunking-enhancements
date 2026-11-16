# Báo cáo tiến độ — Tái lập và cải tiến Adaptive Chunking (Nhóm 5)

*Cập nhật: 04/11/2026 (bản trước: 22/10/2026)*

## 1. Tóm tắt

Nhóm đã tái lập thành công phần đánh giá intrinsic của bài báo, và đã tái lập lần thứ hai một cách độc lập trên hạ tầng khác (laptop + Kaggle, 30/09–02/11). Trong lần chạy này, 6 phương pháp tất định lệch không quá 0,01 điểm so với bài báo, còn Adaptive đạt 91,06 (bài báo: 91,07; lần chạy đầu của nhóm: 90,95). Table 1, Table 2 (phần tất định) và Figure 1 cũng khớp.

Tuyên bố về cải thiện downstream RAG chưa được xác nhận. Trên bộ 99 QA đầu tiên của nhóm, Adaptive đạt nDCG@10 = 75,21%, thấp hơn phương pháp cố định tốt nhất (80,25%). Không so sánh nào có ý nghĩa thống kê sau hiệu chỉnh Holm. Vì bộ QA này gần bão hòa ở Retrieval Completeness, nhóm đã xây và đóng băng bộ QA thứ hai, khó hơn (QA-v2, 04/11).

Lần audit thứ hai bổ sung các phát hiện mới. LLM-regex âm thầm rơi về recursive splitter ở 7/33 tài liệu. Phép thử Wilcoxon của bài báo cho p nhỏ là hệ quả tất yếu của cách định nghĩa Adaptive. Ngoài ra, khi không còn coi chỗ ngắt trang là ranh giới block, page không được chọn ở tài liệu nào (0/33), xác nhận Block Integrity thiên vị page.

Phân tích code metric cho thấy 3/5 metric (SC, BI, RC) đã gần trần. Mọi khoảng tăng còn lại nằm ở ICC và DCC, và hai metric này kéo ngược nhau theo kích thước chunk. Từ đó nhóm đề xuất một hướng cải tiến mới: **chunker cắt theo cấu trúc, chọn điểm cắt bằng quy hoạch động** (mục 5.3), cho kích thước chunk thay đổi theo nội dung thay vì cố định cho cả tài liệu.

## 2. Giới thiệu bài báo gốc

Bài báo đề xuất chọn phương pháp chunking riêng cho từng tài liệu thay vì dùng một phương pháp cho cả corpus. Mỗi phương pháp ứng viên được chấm bằng 5 intrinsic metrics với trọng số đều 20%. Phương pháp có điểm trung bình cao nhất được chọn.

| Metric | Ký hiệu | Code đo gì (`metrics.py`, `compute_metrics.py`) |
| --- | --- | --- |
| References Completeness | RC | 1 − tỉ lệ cặp thực thể–đại từ bị một ranh giới chunk chen vào giữa |
| Intrachunk Cohesion | ICC | Trung bình cosine giữa từng block và cả chunk (jina-v3). Chunk chỉ có 1 block bị loại khỏi phép tính |
| Document Contextual Coherence | DCC | Cosine giữa chunk và cửa sổ khoảng 3.000 token gồm các chunk liền kề |
| Block Integrity | BI | Tỉ lệ block của parser (đoạn giữa hai `split_points`) không bị điểm cắt nào rơi vào giữa, sai số 5 ký tự |
| Size Compliance | SC | Tỉ lệ chunk dài từ 100 đến 1.100 token |

Selector Adaptive chọn trong 4 ứng viên đã qua hậu xử lý: page post-processed, our recursive s=1100, our recursive s=600 và LLM-regex. Theo bài báo, Adaptive đạt điểm intrinsic cao nhất (Table 3) và cải thiện rõ rệt chất lượng trả lời RAG so với LangChain recursive default và page raw (Table 5).

## 3. Các công việc đã làm

Nhóm đã hoàn thành toàn bộ baseline tái lập, thử nghiệm cải tiến đầu tiên (Phase J), một lần tái lập và audit độc lập, và bộ QA thứ hai. Mọi kết quả đều có manifest và checksum SHA256.

| Giai đoạn | Nội dung | Kết quả đầu ra |
| --- | --- | --- |
| 03–04/11 — QA-v2 | Xây bộ QA khó: mỗi tài liệu 3 câu thuộc 3 loại (nhiều phần, phụ thuộc cấu trúc, diễn đạt lại). Quote evidence được kiểm tra tự động, phải khớp nguyên văn và xuất hiện đúng 1 lần trên trang | 99 QA / 33 tài liệu, đóng băng 04/11 (mục 4.5) |
| 30/09–02/11 — Tái lập độc lập | Chạy lại chunking (7 phương pháp, trừ semantic) trên laptop RTX 4060, metrics trên Kaggle 2× T4 (4 shard); đối chiếu từng con số bằng `verify_report.py` | `runs/AUDIT_REPORT.md`, 9 phát hiện audit F1–F9 |
| Phase J (28/09) | Thử nghiệm selector bỏ Block Integrity trên mixed index thật, dùng lại 99 QA và cấu hình retrieval đã khóa | 99 QA × top-10; chạy trên RTX A5000 trong 669 giây |
| Tuần 3 — Phase I | Chạy lại Table 5 (answer quality) bằng GPT-4.1 + DeepEval cho Adaptive, LangChain recursive default raw và page raw | 297 dòng hệ thống × câu hỏi |
| Tuần 3 — Phase H | Thống kê cấp tài liệu: bootstrap 10.000 mẫu, Wilcoxon cặp, hiệu chỉnh Holm; phân tích độ nhạy khi bỏ BI | CI 95% và p-value cho mọi so sánh Adaptive–fixed |
| Tuần 3 — Phase G | Benchmark retrieval 10 hệ thống trên 99 QA (QA-v1: GPT-4.1 sinh 3 QA/tài liệu, có evidence span, duyệt thủ công) | 990 dòng hệ thống × câu hỏi, 9.900 kết quả top-10 |
| Tuần 2 | Tái lập pipeline chunking và 5 intrinsic metrics trên 33 tài liệu CLAIR (kỹ thuật, pháp lý, khoa học xã hội), 8 phương pháp | 2.640 dòng processed + 1.650 dòng raw metrics; 63 test passed |
| Tuần 1 | Đọc bài báo, audit mã nguồn upstream (pin commit ea87ce8), chọn 3 hướng cải tiến | Báo cáo audit paper–code |

Cấu hình thực nghiệm chính:

- **Chunking và metrics (lần 1):** semantic chunking chạy trên RunPod RTX A5000 (Qwen3-Embedding-0.6B); metrics chạy trên Kaggle T4 (Jina embeddings).
- **Chunking và metrics (lần 2):** laptop RTX 4060 8 GB, torch 2.6.0; metrics trên Kaggle 2× T4, torch 2.11, `jinaai/jina-embeddings-v3` pin revision `ab036b0`. LLM-regex dùng `gpt-4o-2024-08-06`, có cache phản hồi (33 request, khoảng 0,82 USD).
- **Retrieval:** BM25 top-50 + dense top-50 (Qwen3-Embedding-4B), hợp nhất, rerank lấy top-10 (Snowflake Arctic Embed L v2.0). Mọi model revision đều được pin.
- **Sinh và chấm câu trả lời:** `gpt-4.1-2025-04-14`, nhiệt độ 0, DeepEval 3.5.9.

## 4. Kết quả và so sánh với bài báo

Phần intrinsic (Table 3) tái lập thành công ở cả hai lần chạy. Phần answer quality (Table 5) không tái lập được thứ hạng. Đánh giá retrieval có evidence mà nhóm bổ sung không cho thấy Adaptive có lợi thế ổn định.

### 4.1 Table 3 — chất lượng chunking intrinsic

Điểm là trung bình 5 metrics (%). Dấu `*` là phương pháp post-processed, được phép tham gia Adaptive. Dấu `†` là phương pháp raw dùng để đối chiếu.

| Phương pháp | Bài báo (%) | Nhóm, lần 1 (%) | Nhóm, lần 2 (%) | Chênh lần 2 (điểm) |
| --- | ---: | ---: | ---: | ---: |
| Adaptive | 91,07 | 90,95 | 91,06 | −0,01 |
| Our recursive s=1100 \* | 90,68 | 90,68 | 90,68 | 0,00 |
| Page post-processed \* | 90,52 | 90,52 | 90,52 | 0,00 |
| LLM-regex \* | 89,80 | 89,99 | 90,20 | +0,40 |
| Our recursive s=600 \* | 89,24 | 89,24 | 89,24 | 0,00 |
| Page raw † | 89,03 | 89,03 | 89,03 | 0,00 |
| LangChain recursive default † | 88,62 | 88,62 | 88,62 | 0,00 |
| LangChain recursive s=1100 † | 88,37 | 88,36 | 88,36 | −0,01 |
| Semantic raw † | 76,49 | 76,56 | không chạy | — |
| Sentence raw † | 73,26 | 71,96 | 71,96 | −1,30 |

Các phương pháp tất định cho cùng một con số ở cả hai lần chạy. Chỉ LLM-regex và Adaptive thay đổi, vì chúng phụ thuộc output của LLM. LLM-regex lệch so với bài báo có thể do bài báo ghi dùng GPT-5 trong khi mã nguồn công khai dùng `gpt-4o`, và do lỗi đọc regex (F3, mục 4.6). Sentence raw lệch nhiều nhất, chủ yếu do metric SC; nhiều khả năng do phiên bản Stanza khác (`pyproject.toml` chỉ khai báo `stanza>=1.10.1`), chưa kiểm chứng.

Điểm từng metric ở lần 2 cho thấy ba metric đầu đã gần trần, còn ICC và DCC kéo ngược nhau theo kích thước chunk:

| Phương pháp | RC | ICC | DCC | BI | SC | ICC + DCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Our recursive s=1100 \* | 99,0 | 66,6 | 89,8 | 98,1 | 100,0 | 156,4 |
| Page post-processed \* | 97,2 | 69,2 | 86,4 | 99,9 | 99,9 | 155,7 |
| LLM-regex \* | 97,9 | 70,4 | 84,1 | 98,7 | 99,8 | 154,5 |
| Our recursive s=600 \* | 97,2 | 69,5 | 84,7 | 94,8 | 100,0 | 154,2 |
| Sentence raw † | 86,6 | 78,6 | 72,0 | 60,9 | 61,7 | 150,6 |
| Adaptive (4 ứng viên) | 99,0 | 68,3 | 88,6 | 99,4 | 99,9 | 157,0 |

Chunk nhỏ có ICC cao nhưng DCC thấp (sentence). Chunk to có DCC cao nhưng ICC thấp (recursive s=1100). Tổng ICC + DCC của mọi phương pháp đơn lẻ chỉ nằm trong khoảng 150,6–156,4.

Phân bố lựa chọn của Adaptive gần với bài báo, nhưng thay đổi giữa hai lần chạy ở các tài liệu mà LLM-regex cạnh tranh:

| Ứng viên được chọn | Bài báo | Nhóm, lần 1 | Nhóm, lần 2 |
| --- | ---: | ---: | ---: |
| Page post-processed | 48% | 45,5% (15/33) | 48,5% (16/33) |
| Our recursive s=1100 | 42% | 39,4% (13/33) | 30,3% (10/33) |
| LLM-regex | 6% | 12,1% (4/33) | 18,2% (6/33) |
| Our recursive s=600 | 3% | 3,0% (1/33) | 3,0% (1/33) |

Điểm tổng của Adaptive ổn định, nhưng việc nó chọn phương pháp nào phụ thuộc vào output của LLM. Vì vậy cần thận trọng khi giải thích Table 4 theo nội dung tài liệu.

### 4.2 Table 5 — chất lượng câu trả lời

| Chỉ số | Nguồn | Adaptive | LC recursive default | Page raw |
| --- | --- | ---: | ---: | ---: |
| Retrieval Completeness (%) | Bài báo | 67,68 | 58,08 | 59,09 |
| Retrieval Completeness (%) | Nhóm | 98,99 | 99,49 | 100,00 |
| Answer Correctness (%) | Bài báo | 78,01 | 70,11 | 73,33 |
| Answer Correctness (%) | Nhóm | 94,32 | 94,65 | 94,25 |
| Correctness điều chỉnh theo coverage (%) | Nhóm | 93,37 | 94,65 | 93,30 |
| Số câu được trả lời (/99) | Bài báo | 65 | 49 | 49 |
| Số câu được trả lời (/99) | Nhóm | 98 | 99 | 98 |

Trên QA-v1, Adaptive không đứng đầu ở chỉ số nào. Ba hệ thống chênh nhau dưới 0,5 điểm. Retrieval Completeness gần bão hòa nên hầu như không phân biệt được các hệ thống.

Không được so trực tiếp số của nhóm với số của bài báo. Bộ 99 QA gốc không được công bố, nên nhóm phải tự tạo bộ QA mới. Vì vậy Phase I là một lần chạy lại có kiểm soát, không phải tái lập đúng con số. Audit lần 2 còn cho thấy prompt sinh QA và prompt sinh câu trả lời trong code (`replicate.py`) khác hẳn Appendix C và E của bài báo, và bước `rag` lấy baseline từ chunk đã hậu xử lý thay vì chunk thô (F8). Chạy `--steps rag` của code gốc vì thế cũng không tái lập đúng protocol của Table 5.

### 4.3 Retrieval có evidence (QA-v1)

Bài báo không báo cáo Hit@k, MRR hay nDCG. Nhóm tính các chỉ số này dựa trên độ trùng giữa source span của chunk và evidence span của câu hỏi.
Hit@10 cho biết top 10 có ít nhất một chunk chạm vào evidence hay không. Mỗi câu hỏi được 1 hoặc 0, rồi lấy trung bình.
MRR@10 (Mean Reciprocal Rank) chỉ xét chunk liên quan đầu tiên và tính điểm bằng 1/hạng của nó. Hạng 1 được 1 điểm, hạng 2 được 0,5, hạng 5 được 0,2, không có trong top 10 thì được 0.
nDCG@10 (normalized Discounted Cumulative Gain) là chỉ số đầy đủ nhất, nên nhóm dùng làm chỉ số chính. Nó tính cả mười vị trí: mỗi chunk đóng góp độ liên quan của nó (0–1), chia cho log₂(hạng + 1), nên chunk càng xuống thấp càng bị giảm trọng số. Tổng này được chia cho giá trị lý tưởng, tức điểm đạt được nếu xếp các chunk tốt nhất có thể lên đầu.

| Hệ thống | Hit@10 (%) | MRR@10 (%) | nDCG@10 (%) |
| --- | ---: | ---: | ---: |
| *Oracle (chọn đúng phương pháp tốt nhất cho từng tài liệu)* | | | *94,29* |
| Raw LC recursive s=1100 (best-fixed) | 94,95 | **75,47** | **80,25** |
| Processed our recursive s=1100 | 93,94 | 74,16 | 78,85 |
| Raw LC recursive default | **97,98** | 72,02 | 77,98 |
| Processed LLM-regex | 93,94 | 72,06 | 77,36 |
| Processed our recursive s=600 | 91,92 | 71,88 | 76,60 |
| **Adaptive** | 90,91 | 70,15 | 75,21 |
| Processed page | 90,91 | 67,51 | 73,16 |
| Raw page | 91,92 | 66,54 | 72,70 |
| Raw semantic | 88,89 | 67,44 | 72,32 |
| Raw sentence | 82,83 | 61,21 | 62,46 |

Về nDCG@10, Adaptive đứng thứ 6/10, thấp hơn best-fixed 5,04 điểm và còn cách oracle 19,08 điểm.

So sánh ở cấp tài liệu cho nDCG@10:

- Adaptive so với LC default: −0,0277, CI 95% [−0,0786; +0,0238].
- Adaptive so với page raw: +0,0251, CI 95% [−0,0197; +0,0725].
- Adaptive so với LC s=1100: −0,0504, CI 95% [−0,1006; −0,0035], Holm p = 0,33.

Sau hiệu chỉnh Holm, không so sánh nào có ý nghĩa ở ngưỡng 0,05.

### 4.4 Phát hiện về Block Integrity

Block Integrity thiên vị page chunking do cách biểu diễn dữ liệu, nên nó không chỉ đo chất lượng chunk. BI dùng `split_points` do parser sinh ra, và gần như mọi chỗ ngắt trang đều trùng một điểm tách: 1.730/1.732 ranh giới trang (99,88%) trong sai số 5 ký tự.

Lần audit thứ hai so sánh 3 biến thể của bộ chọn:

| Biến thể | LLM-regex | rec-1100 | rec-600 | page | Số tài liệu đổi |
| --- | ---: | ---: | ---: | ---: | ---: |
| A. Gốc: 5 metric, trọng số bằng nhau | 6 | 10 | 1 | 16 | — |
| B. Bỏ BI: 4 metric | 6 | 16 | 2 | 9 | 9 |
| C. BI trung lập với trang: bỏ 3.470 ranh giới tại 1.732 chỗ ngắt trang | 14 | 18 | 1 | 0 | 16 |

Với cách tính trung lập, BI của page giảm từ 99,9 xuống 75,0, trong khi các ứng viên khác chỉ giảm 2,9–5,1 điểm. Biến thể C là kịch bản xấu nhất cho page, vì nó coi cả những chỗ đoạn văn thật sự kết thúc ở cuối trang là nằm giữa block. Mức thiên vị thật nằm giữa B và C. Dù vậy, không có bằng chứng độc lập cho lập luận "pagination semantics" của bài báo.

### 4.5 Bộ QA khó (QA-v2)

QA-v1 gần bão hòa (Retrieval Completeness 99–100%), nên khó phân biệt các cách chunking. QA-v2 được thiết kế để chỉ trả lời tốt được khi tài liệu được cắt tốt.

- **Cách sinh:** mỗi tài liệu được xuất thành một cửa sổ các trang liên tiếp, tối đa 45.000 ký tự (khoảng 11k token, gần với 10k token bài báo dùng). Vị trí bắt đầu chọn ngẫu nhiên với seed 2026 và được ghi lại (`export_for_chat.py`). Model sinh là Claude Opus 5.5 (claude.ai, 03/11/2026), khác họ với model sinh và chấm câu trả lời (GPT-4.1), nhằm giảm thiên vị tự đánh giá.
- **Ba loại câu hỏi, mỗi loại một câu cho mỗi tài liệu:** nhiều phần (cần bằng chứng từ ít nhất hai chỗ), phụ thuộc cấu trúc (bảng, danh sách, định nghĩa, đại từ/viết tắt định nghĩa ở chỗ khác) và diễn đạt lại (không dùng lại từ ngữ đặc trưng của văn bản).
- **Kiểm tra tự động (`validate_qa.py`):** đúng schema, quote phải khớp nguyên văn và xuất hiện đúng 1 lần trên trang, câu hỏi phải tự đủ nghĩa. Mỗi câu được gắn nhãn độ khó.
- **Đóng băng:** 04/11/2026, SHA256 `dca47521…c6c8bb31`, có lưu hash của prompt.

| Thuộc tính | Số câu (/99) |
| --- | ---: |
| Có từ 2 bằng chứng trở lên | 70 |
| Bằng chứng nằm trên nhiều block của parser | 53 |
| Bằng chứng nằm trên nhiều trang | 43 |
| Bằng chứng nằm trong bảng | 5 |
| Độ trùng từ giữa câu hỏi và bằng chứng (trung bình) | 0,21 |

Chưa chạy retrieval hay answer quality trên QA-v2.

### 4.6 Các phát hiện audit khác

| Mã | Phát hiện | Hệ quả |
| --- | --- | --- |
| F3 | Với `gpt-4o`, LLM-regex thất bại ở 7/33 tài liệu (21%): LLM trả regex trong khối ```` ```regex ```` thay vì thẻ `<regex>`, nên `extract_llm_regex` trả `None` và tài liệu bị trả nguyên thành 1 chunk mà không có cảnh báo. Bước hậu xử lý sau đó cắt bằng recursive splitter | Hàng LLM-regex thực chất trộn regex thật với recursive. 3/6 tài liệu Adaptive chọn LLM-regex thuộc nhóm thất bại này |
| F5 | Wilcoxon cặp: 8/8 so sánh có p < 0,001, nhưng Adaptive chỉ hơn ứng viên tốt nhất +0,39 điểm, với 20/33 tài liệu có chênh lệch khác 0 | Adaptive được định nghĩa là ứng viên có điểm cao nhất trên chính các metric đem so, nên p nhỏ là hệ quả của định nghĩa, không phải bằng chứng độc lập |
| F7 | `replicate.py` gọi `output_best_chunks` mà không truyền `candidate_methods`, nên chọn trong mọi phương pháp có metric thay vì 4 ứng viên | Lần chạy này không đổi số liệu, nhưng có thể đổi kết quả với dữ liệu khác |
| F8 | Prompt RAG trong code khác Appendix C và E; baseline của bước `rag` lấy từ chunk đã hậu xử lý | Code gốc không tái lập đúng protocol của Table 5 |

## 5. Hướng cải tiến

Hướng đầu tiên (bỏ BI) đã chạy xong nhưng chưa đủ để thành đóng góp. Sau lần audit thứ hai, nhóm bổ sung một hướng mới ở tầng chunking: thay vì chỉ chọn giữa các phương pháp có kích thước cố định, chunker mới cho kích thước thay đổi theo nội dung ngay trong một tài liệu.

| Hướng | Trạng thái | Kết quả / mục tiêu |
| --- | --- | --- |
| Selector bỏ Block Integrity (Phase J) | Đã làm | nDCG@10 tăng +1,69 điểm, chưa có ý nghĩa thống kê |
| Reliability-Aware Downstream-Calibrated Selector | Đang làm | Thu hẹp khoảng cách tới oracle (75,21 → 94,29 nDCG@10) |
| Structure-aware DP Chunker | Mới đề xuất, chưa chạy | Tăng ICC + DCC mà không mất SC, BI, RC; kiểm tra retrieval trên QA-v1 và QA-v2 |
| Context-Enriched Chunk Representation | Sẽ làm | Đo tác động của tiêu đề mục tới retrieval và câu trả lời |
| Query-Aware Multi-Index Selection | Tạm hoãn | Nhường nguồn lực cho chunker DP; chỉ làm nếu dư thời gian và ngân sách |

### 5.1 Đã làm: selector bỏ Block Integrity

Phase J xây một mixed index thật. Trong đó BI có trọng số 0, còn 4 metric còn lại mỗi metric 25%. Kết quả nDCG@10 = 76,90%, so với 75,21% của Adaptive gốc, tức tăng 1,69 điểm (p = 0,56).

### 5.2 Đang làm: Reliability-Aware Downstream-Calibrated Selector

Một selector nhẹ học trọng số từ kết quả retrieval/QA, thay vì chia đều trọng số cho 5 metrics.

- Chia train/test theo tài liệu, 5 fold, seed 2026, phân tầng theo domain, để tránh rò rỉ dữ liệu giữa các tài liệu.
- So sánh 4 cách chọn: equal-weight, learned-weight, best-fixed và oracle.
- Báo cáo regret so với oracle, Recall@k, MRR, nDCG và answer quality, kèm ablation bỏ BI.
- Đánh giá trên cả QA-v1 và QA-v2, vì QA-v1 gần bão hòa.

### 5.3 Mới đề xuất: Structure-aware DP Chunker

**Động lực.** Mọi phương pháp hiện có đều dùng một kích thước chunk cho cả tài liệu, nên chỉ chọn được một điểm trên đường đánh đổi ICC–DCC (mục 4.1). Chunker mới gom các block cùng chủ đề thành chunk to (DCC cao mà ICC vẫn cao), và cắt ngay ở chỗ đổi chủ đề.

**Đầu vào:** `full_text` và `split_points` có sẵn trong `adi_parsed/*.json`, cùng bộ đếm token `o200k_base`. Corpus có 10.975 block, trung bình khoảng 108 token/block (từ 10 đến 3.237 block mỗi tài liệu, trung vị 143).

**Thuật toán:**

1. **Tách block** tại `split_points`, đánh dấu block tiêu đề (bắt đầu bằng `#`, chiếm 20,3% số block) và block bảng.
2. **Chỉ cắt tại ranh giới block**, nên BI = 100% theo định nghĩa. Ngoại lệ là block dài hơn 1.100 token, phải cắt bên trong tại ranh giới dòng hoặc câu. Toàn corpus chỉ có 9 block như vậy (dài nhất 1.686 token), nên mức mất BI không đáng kể.
3. **Ràng buộc cứng:** mỗi chunk 100–1.100 token (SC ≈ 100%); không cắt ngay sau block tiêu đề. **Phạt mềm:** cắt trước block mở đầu bằng đại từ (proxy cho RC, không cần chạy coref).
4. **Chọn điểm cắt bằng quy hoạch động.** Nhúng mỗi block một lần; embedding của chunk xấp xỉ bằng trung bình các block bên trong có trọng số theo số token, rồi chuẩn hoá. Điểm của một chunk là `ICC_proxy + λ·DCC_proxy − μ·[mở đầu bằng đại từ]`. Độ phức tạp O(n·K), với K là số block tối đa trong một chunk; chỉ bước nhúng block cần GPU (Kaggle là đủ).
5. **Chấm điểm bằng code gốc:** cho chunk qua `merge_small_chunks_to_neighbours` (stage `small_merged`), rồi chạy `compute_metrics_per_origin` như các phương pháp khác, để điểm so được trực tiếp trong Table 3.

Tham số λ, μ và cỡ tối đa được chọn bằng cross-validation 5 fold theo tài liệu (cùng fold với selector). Không chỉnh tham số sau khi đã xem điểm trên cả 33 tài liệu.

**Các điểm cần xử lý trước khi chạy** (rút ra khi đối chiếu đề xuất với `metrics.py`):

- **Hàm mục tiêu thiên vị nhiều chunk.** DP cộng dồn điểm theo từng chunk, trong khi ICC và DCC lấy trung bình theo chunk. Nếu điểm mỗi chunk dương, DP sẽ ưu tiên cắt thành nhiều chunk nhỏ, ngược với mục tiêu. Cần chuẩn hoá: trừ một hằng số τ cho mỗi chunk (chọn bằng cross-validation), hoặc nhân điểm với tỉ lệ token của chunk.
- **Proxy chưa khớp hoàn toàn với metric.** DCC thật dùng cửa sổ gồm các chunk *phía sau* (trượt từng chunk một), nên không tách được thành điểm cục bộ của từng chunk. ICC thật bỏ qua chunk chỉ có 1 block. Embedding trung bình có trọng số chỉ là xấp xỉ, vì jina-v3 không tuyến tính. Cần đo tương quan giữa proxy và metric thật trên 2 tài liệu smoke trước khi chạy đủ 33 tài liệu.
- **Giả định RC ≈ 99 là lạc quan.** RC = 99,0 hiện chỉ đạt được ở recursive s=1100, tức chunk to nhất; recursive s=600 và page chỉ đạt 97,2. Proxy đại từ chỉ chạm tới 109 block (1,0%), nên RC chủ yếu vẫn do kích thước chunk quyết định.

**Kỳ vọng (ước lượng, chưa chạy).** Điểm cuối ≈ (100 + ~99,8 + RC + ICC + DCC) / 5.

| ICC + DCC | RC = 99,0 | RC = 97,5 |
| ---: | ---: | ---: |
| 156 (như phương pháp đơn lẻ tốt nhất) | ≈ 91,0 | ≈ 90,7 |
| 158 | ≈ 91,4 | ≈ 91,1 |
| 160 | ≈ 91,8 | ≈ 91,5 |

So với Adaptive (90,95–91,06), mức tăng thực tế nhiều khả năng chỉ khoảng +0,3 đến +1 điểm, vì 3/5 metric đã chạm trần. Độ lệch chuẩn giữa các tài liệu của ICC/DCC là ±2–6 điểm, nên phải kiểm định theo từng tài liệu (bootstrap CI, Wilcoxon), không chỉ so điểm trung bình.

**Các điểm phải ghi rõ khi báo cáo:**

- Chunker dùng chính jina-v3 mà metric dùng, tức là tối ưu thẳng vào thước đo. Cần thêm một biến thể dùng embedder khác (ví dụ Qwen3-Embedding-0.6B đã dùng cho semantic chunking) để chọn điểm cắt. Nếu điểm Table 3 vẫn tăng, kết quả đáng tin hơn.
- So sánh chunker mới đứng riêng với Adaptive 4 ứng viên. Nếu cho nó làm ứng viên thứ 5 thì Adaptive tự động ≥ nó (cùng lý do với F5); biến thể này chỉ báo cáo riêng.
- Biến thể RC dùng thẳng file `mentions/` để tránh cắt giữa các cặp phải ghi rõ là đang dùng dữ liệu mà metric dùng để chấm.
- Kiểm tra retrieval trên cả QA-v1 và QA-v2, để trả lời câu hỏi "điểm Table 3 tăng thì retrieval có tăng không". Đây là điểm quan trọng nhất, vì cả hai lần đánh giá trước đều cho thấy điểm intrinsic cao không kéo theo retrieval tốt hơn.

**Triển khai hai bước:**

- **Bản A, không cần embedding:** chỉ gồm bước 1–3. Chạy trên CPU trong vài phút, cho ngay điểm SC, BI, RC.
- **Bản B, có DP:** nhúng block trên Kaggle, đo tương quan proxy–metric và chấm đủ 5 metric bằng code gốc trên 2 tài liệu smoke, rồi mới chạy đủ 33 tài liệu.

### 5.4 Sẽ làm: Context-Enriched Chunk Representation

Giữ nguyên ranh giới chunk và chỉ thay đổi cách biểu diễn, để không lẫn tác động của biểu diễn với tác động của chunking. Cơ sở của hướng này: mã nguồn gốc có lưu `titles_context` nhưng bỏ đi khi index.

| Biến thể | Văn bản để embed | Văn bản đưa cho LLM |
| --- | --- | --- |
| V0 | Chunk gốc | Chunk gốc |
| V1 | Tiêu đề mục + chunk | Chunk gốc |
| V2 | Tiêu đề mục + chunk | Tiêu đề mục + chunk |

Các chỉ số theo dõi gồm chất lượng retrieval, chất lượng câu trả lời, độ dài context, latency và chi phí token.

### 5.5 Tạm hoãn: Query-Aware Multi-Index Selection

Hướng này chỉ làm khi đáp ứng đủ các điều kiện: hai bộ QA đã đóng băng, các index ổn định, baseline hoàn tất, chunker DP đã có kết quả, và còn ít nhất 35% ngân sách API (tổng ngân sách 60–80 USD). Nếu không đủ điều kiện, thời gian này dành cho phân tích lỗi và hoàn thiện các hướng trên.

## 6. Giới hạn và rủi ro

Giới hạn lớn nhất là không có bộ QA gốc. Vì vậy kết luận downstream chỉ áp dụng cho hai bộ QA và cấu hình retrieval của nhóm.

1. Bộ QA gốc của Table 5 không được công bố, nên Phase I là lần chạy lại có kiểm soát, không phải tái lập đúng con số. Prompt RAG trong code cũng khác Appendix của bài báo (F8).
2. LLM-regex dùng `gpt-4o` thay vì GPT-5 như bài báo ghi, và thất bại âm thầm ở 7/33 tài liệu (F3). Kết quả của LLM-regex và Adaptive thay đổi giữa các lần chạy.
3. Corpus bắt đầu từ JSON đã parse sẵn; nhóm không chạy lại bước PDF → Azure Document Intelligence.
4. Bài báo báo Wilcoxon p<0,001 cho Table 3. Nhóm chạy lại được con số này, nhưng p nhỏ là hệ quả của định nghĩa Adaptive (F5).
5. QA-v1 gần bão hòa (Retrieval Completeness 99–100%). QA-v2 được xây để khắc phục, nhưng chưa có kết quả, và được sinh bằng một LLM duy nhất.
6. Chỉ có 33 tài liệu, nên selector học được và tham số của chunker DP dễ bị overfit, còn các kiểm định có độ mạnh thấp.
7. Chunker DP tối ưu trực tiếp trên embedder của metric, nên điểm intrinsic của nó dễ bị thổi phồng (mục 5.3).
8. Ngân sách API giới hạn ở 60–80 USD, ảnh hưởng tới số lần chạy LLM judge. Nếu chạy qua gateway, cần kiểm tra gateway gọi đúng model (`compare_gateway.py` so số token đầu vào và câu trả lời với API chính thức).

## 7. Kế hoạch các tuần còn lại

Thứ tự ưu tiên sau lần cập nhật này: chạy baseline trên QA-v2, làm chunker DP song song với selector, sau đó đến biểu diễn chunk. QA-v1 và QA-v2 đã đóng băng; các fold theo tài liệu, model revision và artifact của Phase G/I/J được giữ nguyên làm baseline.

| Thời gian | Công việc | Đầu ra dự kiến |
| --- | --- | --- |
| 05–11/11 | Retrieval 10 hệ thống trên QA-v2; chunker DP bản A (SC, BI, RC trên CPU) | Bảng Hit/MRR/nDCG trên QA-v2; điểm 3 metric của bản A |
| 12–18/11 | Chunker DP bản B: smoke 2 tài liệu, tương quan proxy–metric, rồi đủ 33 tài liệu; selector học trọng số | Hàng mới trong Table 3; bảng equal-weight / learned / best-fixed / oracle, regret |
| 19–25/11 | Retrieval cho chunker DP (QA-v1, QA-v2), biến thể embedder khác; Context-Enriched V0/V1/V2 | Bảng retrieval, answer quality, latency, chi phí |
| 26/11 trở đi | Phân tích lỗi, kiểm tra robustness; viết báo cáo cuối kỳ và chuẩn bị bảo vệ | Báo cáo, slide, poster |

## Tài liệu tham khảo

1. *Adaptive Chunking: Optimizing Chunking-Method Selection for RAG* — bài báo gốc, arXiv 2603.25333v1.
2. Nhóm 5. [Adaptive-chunking-enhancements](https://github.com/datnt-numenor/Adaptive-chunking-enhancements) — mã nguồn và kết quả thực nghiệm, commit 2110947 (29/09/2026).
3. Nhóm 5. [Báo cáo so sánh kết quả thực nghiệm với bài báo gốc](https://github.com/datnt-numenor/Adaptive-chunking-enhancements/blob/main/research/week3/PAPER_EXPERIMENT_COMPARISON_REPORT.md) (24/09/2026).
4. Nhóm 5. [Week 3 handoff — kết quả Phase J](https://github.com/datnt-numenor/Adaptive-chunking-enhancements/blob/main/research/week3/HANDOFF.md) (28/09/2026).
5. Nhóm 5. Báo cáo tái lập và audit lần 2 — `runs/AUDIT_REPORT.md` (02/11/2026).
6. Nhóm 5. Bộ QA-v2 — `runs/qa_new/qa_new_frozen.jsonl` và `qa_new_frozen_manifest.json` (04/11/2026).
