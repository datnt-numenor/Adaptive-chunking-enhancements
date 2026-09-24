# Báo cáo so sánh kết quả thực nghiệm với bài báo gốc

**Bài báo đối chiếu:** *Adaptive Chunking: Optimizing Chunking-Method Selection for RAG*  
**Phạm vi thực nghiệm:** 33 tài liệu CLAIR, 8 phương pháp chunking, đánh giá intrinsic, retrieval và answer quality  
**Thời điểm chốt kết quả:** 24/09/2026  
**Trạng thái:** Baseline thực nghiệm bài báo gốc đã hoàn thành trong phạm vi có thể tái lập từ mã nguồn và dữ liệu công khai.

## 1. Tóm tắt kết luận

Kết quả thực nghiệm chia thành hai nhóm có mức độ tái lập khác nhau:

1. **Table 3 về chất lượng chunking được tái lập rất sát.** Trong chín phương pháp cố định, bảy hàng lệch không quá 0,07 điểm phần trăm ở điểm trung bình; LLM-regex lệch +0,19 và sentence raw lệch lớn nhất, -1,30 điểm. Điều này xác nhận pipeline chunking và năm intrinsic metrics nhìn chung có thể tái lập.
2. **Table 5 về RAG không tái lập được các trị số công bố.** Bài báo cho Adaptive đứng đầu rõ rệt, còn lần chạy có kiểm soát của nhóm cho ba hệ thống gần như bão hòa và raw page/raw LangChain nhỉnh hơn Adaptive rất ít. Đây không phải bằng chứng bài báo sai, vì bộ 99 QA gốc không được công bố; nhóm phải tạo và duyệt một bộ QA mới có evidence.
3. **Đánh giá retrieval có evidence không cho thấy lợi thế Adaptive ổn định.** Adaptive đạt nDCG@10 = 75,21%, thấp hơn raw LangChain recursive default (77,98%) và raw LangChain s=1100 (80,25%), nhưng cao hơn raw page (72,70%). Các so sánh nDCG@10 Adaptive–fixed không còn có ý nghĩa thống kê sau hiệu chỉnh Holm.
4. **Block Integrity có dấu hiệu phụ thuộc mạnh vào cấu trúc trang.** 99,88% page boundaries trùng parser split points trong sai số năm ký tự. Bỏ BI khỏi selector làm thay đổi 9/33 quyết định và giảm số lần chọn page từ 15 xuống 9. Đây là điểm yếu thực nghiệm rõ nhất cần kiểm tra ở bước cải tiến.

Vì vậy, có thể kết luận rằng **phần intrinsic của bài báo được tái lập thành công**, còn **tuyên bố Adaptive cải thiện downstream RAG chưa được xác nhận trên bộ QA có evidence của nhóm**.

## 2. Cấu hình và mức độ tương thích

| Thành phần | Bài báo | Thực nghiệm của nhóm | Đánh giá |
|---|---|---|---|
| Corpus | 33 tài liệu CLAIR, ba domain | Đúng 33 tài liệu CLAIR, ba domain | Tương thích |
| Chunking methods | 8 phương pháp | Đúng 8 phương pháp | Tương thích |
| Intrinsic metrics | RC, ICC, DCC, BI, SC; trọng số đều | Giữ đúng năm metrics và trọng số 20% | Tương thích |
| Adaptive candidates | 4 phương pháp có dấu `*` | Selector downstream bị khóa đúng 4 ứng viên | Tương thích sau khi sửa harness |
| Semantic chunker | Qwen3-Embedding-0.6B | Cùng model, revision `97b0c614...`, bf16/FlashAttention 2 trên A5000 | Tương thích và được pin |
| LLM-regex | Bài báo ghi GPT-5 | Mã nguồn công khai mặc định và lần chạy dùng `gpt-4o` | Không hoàn toàn tương thích |
| QA cho Table 5 | GPT-4.1 tạo 3 QA/tài liệu | GPT-4.1 tạo 99 QA, có evidence và duyệt thủ công | Cùng quy trình tổng quát, khác bộ QA |
| QA gốc | Không được công bố | Không thể sử dụng lại | Giới hạn chính |
| Retrieval | BM25 top-50 + dense top-50, hợp nhất, rerank top-10 | Giữ đúng cấu hình; pin embedding/reranker revisions | Tương thích ở mức protocol |
| Answer/judge | GPT-4.1, nhiệt độ 0; DeepEval | `gpt-4.1-2025-04-14`, DeepEval 3.5.9 | Tương thích ở mức protocol |

Mã nguồn upstream cho phần tái lập được pin tại commit `ea87ce8e1a97888f3f179e7f1359ff7f43fb179d`. Kết quả Table 3 dùng semantic chunking chạy trên RTX A5000 và metric shards chạy trên Kaggle Tesla T4. Phase G/I dùng manifest, model revision, input hash và cache để bảo đảm khả năng resume và truy vết.

## 3. So sánh Table 3 — intrinsic chunking quality

`*` là phương pháp post-processed và được phép tham gia Adaptive; `†` là phương pháp raw dùng để đối chiếu. Các giá trị dưới đây là điểm trung bình phần trăm của năm intrinsic metrics.

| Phương pháp | Stage | Bài báo | Thực nghiệm | Chênh lệch |
|---|:---:|---:|---:|---:|
| Our recursive, s=1100 | `*` | 90,68 | 90,68 | 0,00 |
| Our recursive, s=600 | `*` | 89,24 | 89,24 | 0,00 |
| Page post-processed | `*` | 90,52 | 90,52 | 0,00 |
| LLM-regex | `*` | 89,80 | 89,99 | +0,19 |
| LangChain recursive, s=1100 | `†` | 88,37 | 88,36 | -0,01 |
| LangChain recursive default | `†` | 88,62 | 88,62 | 0,00 |
| Page raw | `†` | 89,03 | 89,03 | 0,00 |
| Semantic raw | `†` | 76,49 | 76,56 | +0,07 |
| Sentence raw | `†` | 73,26 | 71,96 | **-1,30** |

### 3.1 Chi tiết các hàng sai khác đáng chú ý

- **LLM-regex:** RC 97,5 so với 98,0; ICC 71,3 so với 70,9; DCC 82,7 so với 82,4; BI 98,8 so với 98,1. Sai khác có thể liên quan đến việc lần chạy dùng `gpt-4o` trong khi Table 2/3 của paper ghi GPT-5.
- **Semantic raw:** điểm trung bình gần như giữ nguyên dù từng metric lệch nhẹ; chênh lệch lớn nhất là SC 48,9 so với 48,1.
- **Sentence raw:** SC đạt 61,7 thay vì 67,2 và BI đạt 60,9 thay vì 61,9, tạo ra phần lớn chênh lệch -1,30 ở mean.
- Các chênh lệch trên là mô tả định lượng. Không nên gọi chúng là khác biệt có ý nghĩa thống kê nếu chưa có kiểm định theo tài liệu cho chính Table 3.

### 3.2 Adaptive và hành vi lựa chọn

Khi chỉ xét đúng bốn ứng viên của paper, kết quả nhóm thu được:

| Chỉ số Adaptive | Bài báo | Thực nghiệm |
|---|---:|---:|
| RC | 99,0 | 98,96 |
| ICC | 68,2 | 68,63 |
| DCC | 88,8 | 88,34 |
| BI | 99,4 | 99,42 |
| SC | 99,9 | 99,93 |
| Mean | **91,07** | **90,95** |

Điểm Adaptive của nhóm thấp hơn paper khoảng 0,12 điểm phần trăm, nhưng vẫn rất gần. Giá trị 90,95 được tính lại từ bốn ứng viên đã khóa; không sử dụng hàng `best=91,06` của CLI cũ vì đường chạy đó từng cho cả tám phương pháp tham gia lựa chọn.

| Phương pháp được chọn | Bài báo | Thực nghiệm |
|---|---:|---:|
| Page post-processed | 48% | 15/33 = 45,5% |
| Our recursive, s=1100 | 42% | 13/33 = 39,4% |
| LLM-regex | 6% | 4/33 = 12,1% |
| Our recursive, s=600 | 3% | 1/33 = 3,0% |

Phân bố vẫn cho thấy selector dùng nhiều hơn một chunker, nhưng lần chạy của nhóm chọn LLM-regex nhiều hơn paper.

## 4. So sánh Table 5 — answer quality

### 4.1 Kết quả công bố trong paper

| Metric | Adaptive | LC recursive default | Page raw |
|---|---:|---:|---:|
| Retrieval Completeness | **67,68** | 58,08 | 59,09 |
| Answer Correctness, chỉ câu có trả lời | **78,01** | 70,11 | 73,33 |
| Mean | **71,77** | 62,07 | 63,80 |
| Số câu được trả lời | **65/99** | 49/99 | 49/99 |

Paper kết luận Adaptive tăng Retrieval Completeness khoảng 8,6–9,6 điểm phần trăm, tăng Correctness khoảng 4,7–7,9 điểm và trả lời thêm 16 câu.

### 4.2 Kết quả chạy có kiểm soát của nhóm

| Metric | Adaptive | LC recursive default raw | Page raw |
|---|---:|---:|---:|
| Retrieval Completeness | 98,99 | 99,49 | **100,00** |
| Answer Correctness, chỉ câu có trả lời | 94,32 | **94,65** | 94,25 |
| Mean theo code upstream | 96,67 | 97,07 | **97,14** |
| Correctness đã điều chỉnh coverage | 93,37 | **94,65** | 93,30 |
| Số câu được trả lời | 98/99 | **99/99** | 98/99 |

Kết quả nhóm **không giữ nguyên thứ hạng của paper**:

- Adaptive không đứng đầu Retrieval Completeness, Correctness hoặc flat mean.
- Chênh lệch giữa ba hệ thống rất nhỏ: chỉ 0,40 điểm Correctness và 0,48 điểm flat mean giữa cao nhất và thấp nhất.
- Retrieval Completeness gần bão hòa 99–100%, nên metric này hầu như không còn khả năng phân biệt hệ thống trên bộ QA mới.

Không nên lấy hiệu `98,99 - 67,68` để tuyên bố hệ thống được cải thiện 31,31 điểm. Hai lần chạy dùng hai bộ QA khác nhau; bộ QA gốc của paper không được công bố và không thể tái tạo chính xác. Vì vậy Phase I là **controlled code/protocol rerun**, không phải numerical reproduction của Table 5.

## 5. Đánh giá retrieval có evidence của nhóm

Paper không công bố Hit@k, Recall@k, MRR hoặc nDCG trên evidence spans. Nhóm bổ sung đánh giá này cho cùng 99 QA đã freeze, sử dụng overlap giữa source spans của chunk và evidence làm relevance.

| Hệ thống | Hit@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| Raw LC recursive, s=1100 | 94,95 | **75,47** | **80,25** |
| Processed our recursive, s=1100 | 93,94 | 74,16 | 78,85 |
| Raw LC recursive default | **97,98** | 72,02 | 77,98 |
| Processed LLM-regex | 93,94 | 72,06 | 77,36 |
| Processed our recursive, s=600 | 91,92 | 71,88 | 76,60 |
| **Adaptive** | 90,91 | 70,15 | 75,21 |
| Processed page | 90,91 | 67,51 | 73,16 |
| Raw page | 91,92 | 66,54 | 72,70 |
| Raw semantic | 88,89 | 67,44 | 72,32 |
| Raw sentence | 82,83 | 61,21 | 62,46 |

Best-fixed theo cross-validation là raw LC recursive s=1100 với nDCG@10 = 80,25%. Oracle upper bound đạt nDCG@10 = 94,29%, cho thấy vẫn còn khoảng trống lớn cho selector tốt hơn.

Ở mức tài liệu:

- Adaptive − raw LC default: nDCG@10 = -0,0277; bootstrap 95% CI `[-0,0786; +0,0238]`; Holm p = 1.
- Adaptive − raw page: nDCG@10 = +0,0251; bootstrap 95% CI `[-0,0197; +0,0725]`; Holm p = 1.
- Adaptive − raw LC s=1100: nDCG@10 = -0,0504; bootstrap 95% CI `[-0,1006; -0,0035]`; Holm p = 0,3266.

Không so sánh nào còn có ý nghĩa ở ngưỡng 0,05 sau hiệu chỉnh Holm. Do đó, kết quả hiện tại không xác nhận tuyên bố rằng Adaptive ổn định hơn các fixed chunker ở downstream retrieval.

## 6. Block Integrity và độ nhạy của selector

Audit cấu trúc cho thấy 1.730/1.732 page boundaries, tương đương 99,88%, trùng với parser split points trong sai số năm ký tự. Điều này khiến BI có khả năng ưu tiên page chunking do cách dữ liệu được biểu diễn, thay vì chỉ đo một thuộc tính độc lập của chất lượng chunk.

Khi đặt trọng số BI bằng 0 và chia đều trọng số cho bốn metric còn lại:

- 9/33 tài liệu đổi phương pháp được chọn;
- page giảm từ 15 xuống 9 tài liệu;
- our recursive s=1100 tăng từ 13 lên 20 tài liệu;
- replay nDCG@10 tăng từ 74,75% lên 76,26%;
- mức tăng +1,51 điểm có CI 95% `[-1,09; +4,60]` và Holm p = 0,4444.

Replay này chỉ là phân tích độ nhạy, chưa phải một lần retrieval mixed-index chính xác. Tuy vậy, nó cung cấp giả thuyết thực nghiệm rõ ràng nhất cho bước cải tiến tiếp theo: kiểm tra selector giảm phụ thuộc vào BI/page structure.

## 7. Các giới hạn cần nêu khi báo cáo

1. Bộ 99 QA gốc, câu trả lời tham chiếu và pairing dùng cho kiểm định Table 5 không được công bố. **Không tìm thấy bằng chứng trong paper/code** đủ để tái tạo đúng từng query của Table 5.
2. LLM-regex của paper được ghi là GPT-5 nhưng mã nguồn và lần chạy tái lập dùng `gpt-4o`; vì vậy hàng này không phải tái lập model-identical.
3. Corpus bắt đầu từ parsed JSON có sẵn; nhóm không tái chạy bước PDF → Azure Document Intelligence.
4. Table 3 của paper báo Wilcoxon `p<0,001`, nhưng mã nguồn công khai không chứa quy trình đủ để khôi phục chính xác phép ghép cặp và kiểm định đó.
5. Correctness trong Table 5 chỉ tính trên câu được trả lời. Vì vậy báo cáo phải kèm answer coverage hoặc coverage-adjusted correctness để tránh làm điểm số có vẻ cao hơn khi hệ thống abstain nhiều.
6. Các kết luận Phase G/H áp dụng cho bộ QA mới có evidence, model revisions và retrieval configuration đã pin; không tự động khái quát sang mọi corpus hoặc RAG stack.

## 8. Kết luận cuối cùng

### Paper nói gì

Paper cho rằng Adaptive Chunking đạt điểm intrinsic cao nhất và tạo ra cải thiện downstream rõ rệt so với LangChain recursive default và page raw.

### Code và artifact thực tế cho thấy

- Fixed-method Table 3 được tái lập rất sát, xác nhận phần lớn pipeline intrinsic.
- Corrected Adaptive đạt 90,95 so với 91,07 của paper và vẫn chọn nhiều chunker theo tài liệu.
- Trên bộ QA mới có evidence, Adaptive không vượt best-fixed về nDCG@10 và không đứng đầu Table 5 controlled rerun.
- Không có lợi thế nDCG@10 Adaptive–fixed nào đạt ý nghĩa thống kê sau Holm correction.
- BI có quan hệ rất mạnh với page/parser boundaries và tác động đáng kể đến lựa chọn của selector.

### Suy luận của nhóm

Kết quả ủng hộ giá trị của framework đánh giá intrinsic và ý tưởng chọn chunker theo tài liệu, nhưng chưa đủ để xác nhận rằng selector equal-weight hiện tại cải thiện downstream RAG một cách ổn định. Hướng nghiên cứu hợp lý là cải tiến selector theo tín hiệu downstream, giảm độ nhạy với BI/page structure và đánh giá trên split theo tài liệu với uncertainty đầy đủ.

## 9. Nguồn bằng chứng

- [Bài báo gốc](../../paper_arxiv_2603.25333.pdf), đặc biệt Table 3 trang PDF 7 và Table 5 trang PDF 9.
- [Log tái lập Table 3](../week2/artifacts/final_table3/analysis_table3.log).
- [Báo cáo kết quả Week 2](../week2/RESULTS.md).
- [Audit paper–README–code](../week2/paper_code_audit.md).
- [Kết quả retrieval Phase G](artifacts/evaluation-full/retrieval_evaluation.json).
- [Phân tích thống kê và BI Phase H](artifacts/phase-h-analysis/SUMMARY.md).
- [Manifest kết quả Phase I](PHASE_I_RESULT_MANIFEST.json).
- [Kết quả tổng hợp Table 5 local](artifacts/phase-i-table5/judge-full/summary/table5_summary.json).

