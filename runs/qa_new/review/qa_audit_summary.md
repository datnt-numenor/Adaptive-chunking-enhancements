# Tổng hợp kiểm tra ngữ nghĩa bộ 99 QA (Phần C)

> File này do `runs/qa_new/review/summarize_audit.py` sinh tự động từ `qa_audit.jsonl`. Mọi con số đều do script tính. Với mỗi `qa_id`, script dùng bản ghi có `revision` cao nhất.

## 0. Dữ liệu đầu vào

- `qa_new_frozen.jsonl` sha256 lúc chạy: `dca47521d0d5922b034f376496c55f960b85c604ef74d1923a31a1ccb6c8bb31`. Manifest: `dca47521d0d5922b034f376496c55f960b85c604ef74d1923a31a1ccb6c8bb31`, **khớp**.
- `qa_audit.jsonl`: 109 dòng, 99 `qa_id` khác nhau. Có 10 `qa_id` có bản sửa (revision > 1): 01-1, 01-2, 01-3, 02-1, 02-2, 02-3, 06-1, 23-1, 29-1, 30-1.
- `qa_id` của file frozen chưa được chấm: không có.

## 1. Số lượng nhãn B1–B8 và verdict

Cột `-1/-2/-3` là vị trí câu trong tài liệu, tức loại mà prompt kỳ vọng.

### B1 `grounded`

| Nhãn | Tổng | Ada (n=27) | Norma (n=48) | Sophia (n=24) | -1 (n=33) | -2 (n=33) | -3 (n=33) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SUPPORTED | 76 | 22 | 37 | 17 | 25 | 22 | 29 |
| PARTIAL | 23 | 5 | 11 | 7 | 8 | 11 | 4 |
| UNSUPPORTED | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| CONTRADICTED | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### B2 `evidence_complete`

| Nhãn | Tổng | Ada (n=27) | Norma (n=48) | Sophia (n=24) | -1 (n=33) | -2 (n=33) | -3 (n=33) |
|---|---:|---:|---:|---:|---:|---:|---:|
| true | 43 | 13 | 21 | 9 | 12 | 15 | 16 |
| false | 56 | 14 | 27 | 15 | 21 | 18 | 17 |

### `missing_info_location`

| Nhãn | Tổng | Ada (n=27) | Norma (n=48) | Sophia (n=24) | -1 (n=33) | -2 (n=33) | -3 (n=33) |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 43 | 13 | 21 | 9 | 12 | 15 | 16 |
| adjacent | 47 | 13 | 23 | 11 | 15 | 17 | 15 |
| same_page | 3 | 0 | 1 | 2 | 2 | 0 | 1 |
| other_page | 5 | 1 | 2 | 2 | 4 | 0 | 1 |
| not_in_document | 1 | 0 | 1 | 0 | 0 | 1 | 0 |

### B3 `self_contained`

| Nhãn | Tổng | Ada (n=27) | Norma (n=48) | Sophia (n=24) | -1 (n=33) | -2 (n=33) | -3 (n=33) |
|---|---:|---:|---:|---:|---:|---:|---:|
| true | 98 | 26 | 48 | 24 | 33 | 32 | 33 |
| false | 1 | 1 | 0 | 0 | 0 | 1 | 0 |

### B4 `actual_type`

| Nhãn | Tổng | Ada (n=27) | Norma (n=48) | Sophia (n=24) | -1 (n=33) | -2 (n=33) | -3 (n=33) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MULTI-PART | 34 | 9 | 16 | 9 | 33 | 1 | 0 |
| STRUCTURE | 30 | 7 | 16 | 7 | 0 | 30 | 0 |
| PARAPHRASED | 33 | 9 | 16 | 8 | 0 | 0 | 33 |
| SIMPLE | 2 | 2 | 0 | 0 | 0 | 2 | 0 |

### B5 `unique_answer`

| Nhãn | Tổng | Ada (n=27) | Norma (n=48) | Sophia (n=24) | -1 (n=33) | -2 (n=33) | -3 (n=33) |
|---|---:|---:|---:|---:|---:|---:|---:|
| UNIQUE | 98 | 27 | 47 | 24 | 33 | 33 | 32 |
| AMBIGUOUS | 1 | 0 | 1 | 0 | 0 | 0 | 1 |

### B6 `translation`

| Nhãn | Tổng | Ada (n=27) | Norma (n=48) | Sophia (n=24) | -1 (n=33) | -2 (n=33) | -3 (n=33) |
|---|---:|---:|---:|---:|---:|---:|---:|
| OK | 18 | 3 | 6 | 9 | 6 | 6 | 6 |
| ISSUE | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| N/A | 81 | 24 | 42 | 15 | 27 | 27 | 27 |

### B7 `chunking_sensitivity` [Suy luận]

| Nhãn | Tổng | Ada (n=27) | Norma (n=48) | Sophia (n=24) | -1 (n=33) | -2 (n=33) | -3 (n=33) |
|---|---:|---:|---:|---:|---:|---:|---:|
| EASY | 6 | 4 | 1 | 1 | 0 | 6 | 0 |
| MEDIUM | 45 | 13 | 22 | 10 | 1 | 11 | 33 |
| HARD | 48 | 10 | 25 | 13 | 32 | 16 | 0 |

### B8 `forbidden_source`

| Nhãn | Tổng | Ada (n=27) | Norma (n=48) | Sophia (n=24) | -1 (n=33) | -2 (n=33) | -3 (n=33) |
|---|---:|---:|---:|---:|---:|---:|---:|
| true | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| false | 99 | 27 | 48 | 24 | 33 | 33 | 33 |

### `verdict`

| Nhãn | Tổng | Ada (n=27) | Norma (n=48) | Sophia (n=24) | -1 (n=33) | -2 (n=33) | -3 (n=33) |
|---|---:|---:|---:|---:|---:|---:|---:|
| KEEP | 42 | 13 | 20 | 9 | 12 | 15 | 15 |
| FIX | 57 | 14 | 28 | 15 | 21 | 18 | 18 |
| DROP | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### `doc_language` (heuristic)

| Nhãn | Tổng | Ada (n=27) | Norma (n=48) | Sophia (n=24) | -1 (n=33) | -2 (n=33) | -3 (n=33) |
|---|---:|---:|---:|---:|---:|---:|---:|
| en | 81 | 24 | 42 | 15 | 27 | 27 | 27 |
| fr | 18 | 3 | 6 | 9 | 6 | 6 | 6 |

### `fix_type` (trong các câu FIX; một câu có thể có nhiều loại)

| fix_type | Số câu FIX |
|---|---:|
| evidence_extend | 46 |
| evidence_add | 18 |
| answer_change | 1 |
| question_change | 3 |
| *(chỉ sửa evidence, giữ nguyên câu hỏi và câu trả lời)* | 54 |

### `minor_issues` (không ảnh hưởng verdict)

| minor_issue | Số câu | qa_id |
|---|---:|---|
| duplicate_passage_exec_summary_p4 | 1 | 21-1 |
| duplicate_passage_exec_summary_p5 | 1 | 21-2 |
| evidence_fix_infeasible_3_quotes | 3 | 23-1, 29-1, 30-1 |
| length_31_words | 3 | 03-1, 15-1, 30-2 |
| length_32_words | 2 | 08-2, 14-1 |
| residual_missing_after_fix | 3 | 23-1, 29-1, 30-1 |

### Loại kỳ vọng (vị trí) × loại thực tế (B4)

| expected \ actual | MULTI-PART | STRUCTURE | PARAPHRASED | SIMPLE |
|---|---:|---:|---:|---:|
| -1 (MULTI-PART) | 33 | 0 | 0 | 0 |
| -2 (STRUCTURE) | 1 | 30 | 0 | 2 |
| -3 (PARAPHRASED) | 0 | 0 | 33 | 0 |

Loại thực tế trùng loại kỳ vọng: 96/99 (97.0%).

## 2. Độ nhạy với chunking (B7) và nguy cơ bão hoà

- Phân bố trên bộ frozen hiện tại: EASY=6, MEDIUM=45, HARD=48. **Tỉ lệ EASY = 6/99 (6.1%)**.
  - Ada: EASY 4/27 (14.8%), HARD 10/27 (37.0%)
  - Norma: EASY 1/48 (2.1%), HARD 25/48 (52.1%)
  - Sophia: EASY 1/24 (4.2%), HARD 13/24 (54.2%)
  - -1 (MULTI-PART): EASY 0/33 (0.0%), HARD 32/33 (97.0%)
  - -2 (STRUCTURE): EASY 6/33 (18.2%), HARD 16/33 (48.5%)
  - -3 (PARAPHRASED): EASY 0/33 (0.0%), HARD 0/33 (0.0%)
- Các câu EASY (đánh dấu để thống kê, không làm đổi verdict): 03-2 (KEEP), 04-2 (FIX), 07-2 (KEEP), 09-2 (KEEP), 17-2 (KEEP), 32-2 (FIX).
- Nếu áp dụng mọi FIX và bỏ mọi DROP (bộ v2 giả định, 99 câu): EASY=6, MEDIUM=45, HARD=48; tỉ lệ EASY = 6.1%.
- Các câu FIX có B7 thay đổi sau khi sửa: 13-3 (MEDIUM→HARD), 16-2 (HARD→MEDIUM).

## 3. Danh sách các câu FIX và DROP

Cột 'Vấn đề' liệt kê các nhãn không đạt. Cột 'Lý do' là câu đầu của `reasoning` [Suy luận]. Toàn văn, cùng `evidence_for_verdict` [Bằng chứng] và `proposed_fix` đầy đủ, nằm trong `qa_audit.jsonl`.

| qa_id | verdict | Vấn đề | fix_type | Đề xuất sửa | Lý do |
|---|---|---|---|---|---|
| 01-2 | FIX | B2=false (adjacent) | evidence_extend | evidence: 2 quote (trang 9, 9; 167, 233 ký tự) | Phần trả lời chính (Art. 15/16/17) được quote hỗ trợ đầy đủ, nhưng ý giới hạn 'input and output' nằm ở câu dẫn ngay trước danh sách, ngoài quote, nên B2=false.… |
| 01-3 | FIX | B1=PARTIAL; B2=false (adjacent) | evidence_extend | evidence: 2 quote (trang 8, 8; 153, 177 ký tự) | Một trong bốn khoản 'bị bỏ ra' (nine terabytes) là một phần của câu trả lời chính nhưng không có trong quote, nên chấm PARTIAL.… |
| 02-1 | FIX | B1=PARTIAL; B2=false (other_page) | evidence_extend, evidence_add | evidence: 3 quote (trang 20, 21, 22; 121, 154, 257 ký tự) | Phần trả lời về tài khoản/secrets được quote p21 hỗ trợ đủ. Phần 'khi nào được chia sẻ GPU' thiếu một trong hai điều kiện, và hai ý '(training, deployment, production)' cùng 'not shared with other business applications' không có trong quote, nên chấm PARTIAL.… |
| 02-2 | FIX | B2=false (adjacent) | evidence_extend, evidence_add | evidence: 3 quote (trang 13, 13, 13; 184, 264, 66 ký tự) | Phần trả lời chính đúng và có quote hỗ trợ, nhưng danh sách 3 loại (manipulation, infection, exfiltration) chỉ có ở ngữ cảnh sát quote (cùng danh sách), nên B2=false, missing_info_location=adjacent.… |
| 03-1 | FIX | B2=false (adjacent); minor: length_31_words | evidence_extend | evidence: 2 quote (trang 3, 4; 157, 198 ký tự) | Cả hai ý chính đều được quote hỗ trợ, nhưng hai ý phụ của câu trả lời nằm ở câu liền kề, ngoài quote, nên B2=false.… |
| 04-2 | FIX | B1=PARTIAL; B2=false (adjacent) | evidence_extend | evidence: 1 quote (trang 1; 247 ký tự) | Thiếu một trong ba thuộc tính trong quote, nên chấm PARTIAL.… |
| 05-2 | FIX | B2=false (adjacent); B3=false | evidence_extend, question_change | evidence: 2 quote (trang 1, 2; 110, 231 ký tự); câu hỏi mới: “The responsible-AI standards chapter on Robustness lists causes of data skew. Besides training design, unavailable production features, data dependencies and upstream changes, what final cause is given?” | B3=false vì lỗi ngữ pháp, có thể do sửa tay; đề xuất 'list' thành 'lists'.… |
| 06-2 | FIX | B1=PARTIAL; B2=false (adjacent) | evidence_extend | evidence: 2 quote (trang 3, 4; 227, 299 ký tự) | Thiếu 3/5 mục trong quote, nên chấm PARTIAL. Danh sách vắt qua ngắt trang (SP@5583/5603).… |
| 06-3 | FIX | B2=false (adjacent) | evidence_add | evidence: 2 quote (trang 5, 5; 170, 135 ký tự) | Phần trả lời chính được quote hỗ trợ; câu thứ hai không được câu hỏi hỏi tới và nằm ngoài quote, nên B2=false.… |
| 07-3 | FIX | B2=false (adjacent) | evidence_extend | evidence: 1 quote (trang 7; 204 ký tự) | Các con số được quote hỗ trợ, nhưng tên mô hình trong câu trả lời nằm ngoài quote, nên B2=false.… |
| 08-1 | FIX | B2=false (adjacent) | evidence_extend | evidence: 2 quote (trang 1, 3; 92, 172 ký tự) | Phần trả lời chính (bài Mitchell et al. 2019; Google ra một toolkit năm 2020) được quote hỗ trợ, nhưng tên riêng 'Model Card Toolkit' nằm ngoài quote, nên B2=false.… |
| 08-2 | FIX | B1=PARTIAL; B2=false (adjacent); minor: length_32_words | evidence_extend | evidence: 2 quote (trang 2, 2; 57, 289 ký tự) | Nửa đầu câu hỏi ('what should the Training Data section contain') có đáp án nằm ngoài quote, nên chấm PARTIAL.… |
| 08-3 | FIX | B2=false (adjacent) | evidence_extend | evidence: 1 quote (trang 4; 209 ký tự) | Câu hỏi hỏi 'which company', và quote có 'MAIF', nên phần trả lời chính đạt.… |
| 09-3 | FIX | B2=false (adjacent) | evidence_add | evidence: 2 quote (trang 6, 6; 183, 139 ký tự) | Phần trả lời chính (repair_level) đạt, nhưng tên kỹ thuật trong câu trả lời nằm ngoài quote, nên B2=false.… |
| 10-1 | FIX | B2=false (adjacent) | evidence_extend | evidence: 2 quote (trang 6, 7; 172, 258 ký tự) | Hai con số chính có trong quote, còn phần so sánh '+70%, 500 triệu/năm' nằm ngoài, nên B2=false.… |
| 10-3 | FIX | B2=false (adjacent) | evidence_add | evidence: 2 quote (trang 13, 13; 117, 107 ký tự) | Con số chính đạt, nhưng danh tính nhóm nghề ('ICT specialists') nằm ngoài quote, nên B2=false.… |
| 11-2 | FIX | B1=PARTIAL; B2=false (adjacent) | evidence_extend | evidence: 3 quote (trang 47, 47, 47; 159, 297, 297 ký tự) | Câu hỏi hỏi trực tiếp 'which recipients', mà 2/3 nhóm thiếu trong quote, nên chấm PARTIAL.… |
| 11-3 | FIX | B5=AMBIGUOUS | question_change | evidence: 2 quote (trang 35, 35; 149, 143 ký tự); câu hỏi mới: “Under the California CCPA, by up to what total period may a business extend its response when requests are complex or numerous, and when must it notify the consumer?” | Câu hỏi hỏi 'how much additional time', và p12 cho đáp án hợp lệ 'thêm 45 ngày', khác với đáp án frozen '90 ngày tổng'.… |
| 12-1 | FIX | B2=false (adjacent) | evidence_extend, evidence_add | evidence: 3 quote (trang 4, 4, 5; 104, 120, 217 ký tự) | Hai nghĩa vụ chính đều có trong quote, nhưng hai ý phụ (không né tránh biện pháp kỹ thuật; RFC 9309) nằm ngoài quote, nên B2=false.… |
| 12-2 | FIX | B2=false (adjacent) | evidence_extend | evidence: 2 quote (trang 4, 5; 268, 202 ký tự) | Phần trả lời chính đạt, nhưng ý thứ hai nằm ngoài quote, nên B2=false.… |
| 13-1 | FIX | B2=false (adjacent) | evidence_extend | evidence: 2 quote (trang 25, 26; 260, 218 ký tự) | Hai con số chính có trong quote, nhưng mốc tính thời hạn của cả hai nằm ngoài quote, nên B2=false.… |
| 13-2 | FIX | B2=false (adjacent) | evidence_extend, evidence_add | evidence: 3 quote (trang 17, 29, 29; 200, 238, 109 ký tự) | Quy mô nhóm, thời gian và ngân sách (đúng các điều câu hỏi hỏi) có trong quote.… |
| 13-3 | FIX | B2=false (other_page) | evidence_add | evidence: 3 quote (trang 18, 19, 19; 130, 142, 173 ký tự) | Con số chính 'at least five' có trong quote, nhưng hai ý của câu trả lời nằm ngoài quote, trong đó một ý ở trang khác (p18), nên B2=false và missing_info_location=other_page.… |
| 14-2 | FIX | B1=PARTIAL; B2=false (adjacent) | evidence_extend | evidence: 2 quote (trang 8, 7; 287, 276 ký tự) | Kết luận 'chỉ dành cho DPs' cần biết (a) đây là dòng nào và (b) cột thứ ba là DPs.… |
| 14-3 | FIX | B2=false (adjacent) | evidence_add | evidence: 2 quote (trang 9, 9; 156, 151 ký tự) | Đơn vị và độ chính xác (đúng điều câu hỏi hỏi) có trong quote; câu thứ hai của câu trả lời nằm ngoài quote, nên B2=false.… |
| 15-1 | FIX | B1=PARTIAL; B2=false (adjacent); minor: length_31_words | evidence_extend | evidence: 3 quote (trang 8, 3, 3; 258, 190, 142 ký tự) | Nếu không có cụm 'give effect to Article 3, paragraph 1, sub-paragraphs a and b', không có quote nào nói báo cáo phải bao gồm hoạt động của các chủ thể ở Art. 3(1)(a)/(b).… |
| 15-2 | FIX | B2=false (adjacent) | evidence_extend | evidence: 3 quote (trang 11, 11, 11; 156, 236, 89 ký tự) | Phần trả lời chính (bảo lưu theo Art. 33(1) và nội dung của nó) có trong quote.… |
| 16-1 | FIX | B1=PARTIAL; B2=false (adjacent) | evidence_extend, evidence_add | evidence: 3 quote (trang 3, 3, 4; 97, 266, 139 ký tự) | Câu hỏi hỏi 'which Annex III systems', mà một nửa đáp án (point 1) nằm ngoài quote, nên chấm PARTIAL.… |
| 16-2 | FIX | B1=PARTIAL; B2=false (not_in_document) | question_change, answer_change, evidence_extend | evidence: 2 quote (trang 2, 2; 123, 292 ký tự); câu hỏi mới: “According to EDPB Statement 3/2024, for which AI use is the AI Act lex specialis vis-à-vis the LED, and what processing does Directive (EU) 2016/680 govern?”; câu trả lời mới: “For remote biometric identification, the AI Act is lex specialis vis-à-vis the LED (Art. 5(1)(h) AI Act). Directive (EU) 2016/680 governs the processing of personal data by competent authorities for preventing, investigating, detecting or prosecuting criminal offences.” | Câu trả lời 'LED, which is Directive (EU) 2016/680' và việc dùng fn 5 để mô tả 'that directive' dựa vào kiến thức ngoài (LED = Law Enforcement Directive = 2016/680), nên chấm PARTIAL với missing_info_location=not_in_document.… |
| 16-3 | FIX | B2=false (adjacent) | evidence_extend | evidence: 1 quote (trang 4; 203 ký tự) | Ý chính (nguồn lực nhân sự và tài chính do Member States cung cấp) có trong quote, nhưng chi tiết 'including as MSAs' của câu trả lời nằm ngoài quote, nên B2=false.… |
| 19-1 | FIX | B2=false (other_page) | evidence_add | evidence: 3 quote (trang 8, 9, 5; 221, 170, 48 ký tự) | Phần trả lời chính ('No' và nội dung Art. 4(1)) có trong quote.… |
| 20-2 | FIX | B1=PARTIAL; B2=false (adjacent) | evidence_extend | evidence: 2 quote (trang 33, 34; 225, 281 ký tự) | Câu hỏi hỏi 'which bodies', mà chủ thể 'public authorities' không nằm trong quote nào, nên chấm PARTIAL.… |
| 21-1 | FIX | B2=false (adjacent); minor: duplicate_passage_exec_summary_p4 | evidence_extend | evidence: 3 quote (trang 20, 21, 22; 214, 129, 147 ký tự) | Ba ý chính có trong quote, nhưng căn cứ 'Regulation 1025/2012 requires ESOs' của câu trả lời nằm ngoài quote, nên B2=false.… |
| 21-3 | FIX | B2=false (adjacent) | evidence_extend | evidence: 2 quote (trang 21, 21; 141, 202 ký tự) | Hai phần chính (Microsoft và Google; vai trò 'deployer') có trong quote, nhưng phần cuối định nghĩa 'deployer' trong câu trả lời nằm ngoài quote, nên B2=false.… |
| 22-1 | FIX | B2=false (adjacent) | evidence_add | evidence: 3 quote (trang 45, 52, 52; 182, 200, 145 ký tự) | Hai ý chính có trong quote và được dịch đúng. Câu 'sharply reduces confidentiality risks without harming generalisation' nằm ngoài quote, nên B2=false.… |
| 22-2 | FIX | B1=PARTIAL; B2=false (adjacent) | evidence_extend, evidence_add | evidence: 3 quote (trang 44, 44, 45; 233, 82, 214 ký tự) | Câu hỏi hỏi trực tiếp 'which four dimensions', mà 3/4 tên nằm ngoài quote, nên chấm PARTIAL.… |
| 22-3 | FIX | B2=false (adjacent) | evidence_extend | evidence: 1 quote (trang 43; 270 ký tự) | Phần trả lời chính (cảm biến hồng ngoại, không thu hình ảnh) có trong quote và được dịch đúng ('caméra augmentée' ≈ smart video), nhưng nguyên tắc viện dẫn nằm ngoài quote, nên B2=false.… |
| 23-1 | FIX | B1=PARTIAL; B2=false (adjacent); minor: evidence_fix_infeasible_3_quotes, residual_missing_after_fix | evidence_extend | evidence: 3 quote (trang 22, 22, 23; 291, 240, 233 ký tự) | Câu hỏi hỏi 'which obligations … exempted from', mà 2/3 nghĩa vụ nằm ngoài quote, nên chấm PARTIAL.… |
| 23-3 | FIX | B1=PARTIAL; B2=false (same_page) | evidence_add | evidence: 3 quote (trang 25, 25, 25; 157, 91, 236 ký tự) | Câu hỏi hỏi 'có tính là commercial exploitation không'. Quote không nói gạch đầu dòng này là monetisation, nên đáp án 'Yes' thiếu căn cứ trong quote, chấm PARTIAL.… |
| 24-1 | FIX | B1=PARTIAL; B2=false (adjacent) | evidence_extend | evidence: 2 quote (trang 112, 117; 174, 268 ký tự) | Thiếu tiêu đề và chủ thể của Art. 101, nên không có quote nào nối mức '3%/15 triệu' với 'Article 101' mà quote 1 nhắc tới.… |
| 24-3 | FIX | B1=PARTIAL; B2=false (adjacent) | evidence_extend | evidence: 1 quote (trang 117; 214 ký tự) | Câu hỏi có hai vế, vế thứ hai không có quote, nên chấm PARTIAL.… |
| 25-1 | FIX | B2=false (adjacent) | evidence_extend | evidence: 2 quote (trang 148, 152; 198, 222 ký tự) | Hai ý chính có trong quote, nhưng phần giải thích gắn với 'password locking' (cũng là tiền đề trong câu hỏi) nằm ngoài quote, nên B2=false.… |
| 26-1 | FIX | B2=false (adjacent) | evidence_extend | evidence: 2 quote (trang 40, 41; 263, 188 ký tự) | Cơ chế (inspector không biết case nào được chọn ngẫu nhiên) có trong quote, nhưng hệ quả chống automation bias nằm ngoài quote, nên B2=false.… |
| 27-1 | FIX | B2=false (adjacent) | evidence_extend | evidence: 2 quote (trang 32, 34; 243, 282 ký tự) | Hai ý chính có trong quote. Tuy nhiên, câu trả lời dùng 'international co-operation', mà ý này cùng đối tượng của 'this' nằm ngoài quote, nên B2=false.… |
| 27-2 | FIX | B1=PARTIAL; B2=false (adjacent) | evidence_extend | evidence: 2 quote (trang 37, 37; 68, 135 ký tự) | Muốn biết '31>40' là nhóm tuổi và '139/83' là huyết áp thì cần tiêu đề cột, mà tiêu đề nằm ngoài quote.… |
| 27-3 | FIX | B1=PARTIAL; B2=false (adjacent) | evidence_extend | evidence: 1 quote (trang 45; 227 ký tự) | Câu hỏi hỏi 'why', mà lý do gồm hai vế (luật chỉ bảo vệ một số đặc điểm, và AI gây phân biệt đối xử không tương quan với các đặc điểm đó); quote chỉ có vế đầu, nên chấm PARTIAL.… |
| 28-1 | FIX | B2=false (other_page) | evidence_extend, evidence_add | evidence: 3 quote (trang 27, 39, 40; 103, 58, 189 ký tự) | Hai con số chính có trong quote, nhưng việc chế tài nằm ở Art. 36 (p39, trước ngắt trang) thì ngoài quote, nên B2=false, missing_info_location=other_page.… |
| 29-1 | FIX | B1=PARTIAL; B2=false (same_page); minor: evidence_fix_infeasible_3_quotes, residual_missing_after_fix | evidence_extend, evidence_add | evidence: 3 quote (trang 88, 88, 89; 238, 216, 127 ký tự) | Không có tiêu đề cột thì quote 1 không cho biết 30 t thuộc BLOOM, và ba thành phần cộng thêm nằm ngoài quote, nên chấm PARTIAL.… |
| 30-1 | FIX | B1=PARTIAL; B2=false (same_page); minor: evidence_fix_infeasible_3_quotes, residual_missing_after_fix | evidence_extend | evidence: 3 quote (trang 18, 19, 19; 104, 103, 67 ký tự) | Không có tiêu đề cột thì '92/10/82' không gắn được với 'good/very good/fairly good', nên chấm PARTIAL.… |
| 30-2 | FIX | B1=PARTIAL; B2=false (adjacent); minor: length_31_words | evidence_extend | evidence: 2 quote (trang 19, 19; 281, 43 ký tự) | Không có tiêu đề cột thì không biết số nào là 'very good' hay 'total positive', nên chấm PARTIAL (cùng cách chấm với 14-2/27-2).… |
| 30-3 | FIX | B2=false (adjacent) | evidence_extend | evidence: 2 quote (trang 16, 16; 73, 200 ký tự) | Hai ý chính (3 năm; đồng tử) có trong quote và được dịch đúng, nhưng phần mô tả mục đích tiền chẩn đoán nằm ngoài quote, nên B2=false.… |
| 31-1 | FIX | B1=PARTIAL; B2=false (other_page) | evidence_add | evidence: 3 quote (trang 34, 35, 44; 127, 76, 133 ký tự) | Quote p35 không cho biết cái gì tăng 25% (năng suất), nên vế năng suất thiếu căn cứ trong quote, chấm PARTIAL (missing ở p34, other_page).… |
| 31-3 | FIX | B2=false (adjacent) | evidence_add | evidence: 2 quote (trang 40, 40; 118, 131 ký tự) | Con số chính (30 km) có trong quote và được dịch đúng, nhưng chi tiết triều đại trong câu trả lời nằm ngoài quote, nên B2=false.… |
| 32-2 | FIX | B1=PARTIAL; B2=false (adjacent) | evidence_extend | evidence: 1 quote (trang 6; 197 ký tự) | Câu hỏi hỏi 'which tier', mà tên tầng nằm ngoài quote, nên chấm PARTIAL.… |
| 32-3 | FIX | B2=false (adjacent) | evidence_extend | evidence: 1 quote (trang 13; 251 ký tự) | Hai con số chính có trong quote, nhưng câu thứ hai của câu trả lời nằm ngoài quote, nên B2=false.… |
| 33-1 | FIX | B2=false (adjacent) | evidence_extend | evidence: 2 quote (trang 7, 10; 226, 98 ký tự) | Hai ý chính có trong quote, nhưng phần 'confidentiality … IP' của câu trả lời nằm ngoài quote, nên B2=false.… |
| 33-2 | FIX | B2=false (adjacent) | evidence_extend | evidence: 2 quote (trang 12, 12; 123, 174 ký tự) | Cả hai vế đều có trong quote, nhưng mối nối 'khuyến nghị này dành cho mô hình không trực tiếp diễn giải được' nằm ở dòng tiêu đề ngoài quote, nên B2=false.… |

Các câu chỉ có `minor_fix` (vẫn KEEP, đề xuất rút gọn câu hỏi xuống ≤ 30 từ): 14-1.

## 4. So sánh nhãn tự động (`qa_review.csv`) với đánh giá của Claude

### `cross_block` (csv) × B7

| | EASY | MEDIUM | HARD | Tổng |
|---|---:|---:|---:|---:|
| cross_block=True | 0 | 5 | 48 | 53 |
| cross_block=False | 6 | 40 | 0 | 46 |

### `cross_page` (csv) × B7

| | EASY | MEDIUM | HARD | Tổng |
|---|---:|---:|---:|---:|
| cross_page=True | 0 | 1 | 42 | 43 |
| cross_page=False | 6 | 44 | 6 | 56 |

### `lexical_overlap` (csv) × B7

| | EASY | MEDIUM | HARD | Tổng |
|---|---:|---:|---:|---:|
| lex < 0.15 | 2 | 30 | 16 | 48 |
| 0.15 ≤ lex < 0.35 | 2 | 12 | 12 | 26 |
| lex ≥ 0.35 | 2 | 3 | 20 | 25 |

### `has_table` (csv) × B4

| | MULTI-PART | STRUCTURE | PARAPHRASED | SIMPLE | Tổng |
|---|---:|---:|---:|---:|---:|
| has_table=True | 2 | 3 | 0 | 0 | 5 |
| has_table=False | 32 | 27 | 33 | 2 | 94 |

### `n_evidence` (csv) × B4

| | MULTI-PART | STRUCTURE | PARAPHRASED | SIMPLE | Tổng |
|---|---:|---:|---:|---:|---:|
| n_evidence=1 | 0 | 2 | 25 | 2 | 29 |
| n_evidence=2 | 27 | 24 | 8 | 0 | 59 |
| n_evidence=3 | 7 | 4 | 0 | 0 | 11 |

### `n_evidence` (csv) × B2

| | True | False | Tổng |
|---|---:|---:|---:|
| n_evidence=1 | 14 | 15 | 29 |
| n_evidence=2 | 26 | 33 | 59 |
| n_evidence=3 | 3 | 8 | 11 |

### Các trường hợp lệch đáng chú ý

- **cross_block=True nhưng Claude chấm EASY**: 0 câu.
- **cross_block=False nhưng Claude chấm HARD**: 0 câu.
- **cross_page=True nhưng Claude không chấm HARD**: 1 câu:
  - 03-1 (csv: cross_block=True, cross_page=True, lex=0.5; Claude: MULTI-PART/MEDIUM): Cả hai ý chính đều được quote hỗ trợ, nhưng hai ý phụ của câu trả lời nằm ở câu liền kề, ngoài quote, nên B2=false.…
- **n_evidence=1 nhưng Claude chấm MULTI-PART**: 0 câu.
- **has_table=True nhưng Claude không chấm STRUCTURE**: 2 câu:
  - 29-1 (csv: cross_block=True, cross_page=True, lex=0.0; Claude: MULTI-PART/HARD): Không có tiêu đề cột thì quote 1 không cho biết 30 t thuộc BLOOM, và ba thành phần cộng thêm nằm ngoài quote, nên chấm PARTIAL.…
  - 30-1 (csv: cross_block=True, cross_page=True, lex=0.07; Claude: MULTI-PART/HARD): Không có tiêu đề cột thì '92/10/82' không gắn được với 'good/very good/fairly good', nên chấm PARTIAL.…
- **lexical_overlap ≥ 0.35 nhưng Claude chấm HARD**: 20 câu:
  - 01-1 (csv: cross_block=True, cross_page=True, lex=0.5; Claude: MULTI-PART/HARD): Hai ý của câu trả lời nằm ở hai trang (8, 9), cách nhau ~3.000 ký tự và 6 split_point, thuộc hai mục khác nhau (IV.1 và IV.3).…
  - 04-1 (csv: cross_block=True, cross_page=True, lex=0.62; Claude: MULTI-PART/HARD): Hai quote cách nhau 5 trang (~9.700 ký tự, 14 split_point), mỗi quote trả lời một nửa câu hỏi, nên đây là MULTI-PART thật và HARD.
  - 06-1 (csv: cross_block=True, cross_page=True, lex=0.44; Claude: MULTI-PART/HARD): Mọi ý đều có trong quote. Hai quote chỉ cách nhau 226 ký tự nhưng evidence vắt qua ranh giới block tại vị trí 3838 và 3858 (PageBreak).…
  - 06-2 (csv: cross_block=True, cross_page=True, lex=0.42; Claude: STRUCTURE/HARD): Thiếu 3/5 mục trong quote, nên chấm PARTIAL. Danh sách vắt qua ngắt trang (SP@5583/5603).…
  - 07-1 (csv: cross_block=True, cross_page=True, lex=0.36; Claude: MULTI-PART/HARD): Mọi ý đều có trong quote. Cả hai quote đều dùng đại từ 'It', trỏ về CodeCarbon ở tiêu đề B.2 (SP@6877).…
  - 08-1 (csv: cross_block=True, cross_page=True, lex=0.47; Claude: MULTI-PART/HARD): Phần trả lời chính (bài Mitchell et al. 2019; Google ra một toolkit năm 2020) được quote hỗ trợ, nhưng tên riêng 'Model Card Toolkit' nằm ngoài quote, nên B2=false.…
  - 09-1 (csv: cross_block=True, cross_page=True, lex=0.54; Claude: MULTI-PART/HARD): Mọi ý đều có trong quote. Câu nói về Disparate Impact bị PageBreak cắt đôi ('It' ở cuối p4, 'compares…' ở đầu p5).…
  - 10-1 (csv: cross_block=True, cross_page=True, lex=0.4; Claude: MULTI-PART/HARD): Hai con số chính có trong quote, còn phần so sánh '+70%, 500 triệu/năm' nằm ngoài, nên B2=false.…
  - 11-1 (csv: cross_block=True, cross_page=True, lex=0.5; Claude: MULTI-PART/HARD): Mọi ý đều có trong quote. B5: các chỗ khác nêu cùng mức $2,500/$7,500, không mâu thuẫn; chữ 'administrative' trong câu hỏi chỉ đúng §1798.155.…
  - 12-1 (csv: cross_block=True, cross_page=True, lex=0.4; Claude: MULTI-PART/HARD): Hai nghĩa vụ chính đều có trong quote, nhưng hai ý phụ (không né tránh biện pháp kỹ thuật; RFC 9309) nằm ngoài quote, nên B2=false.…
  - 12-2 (csv: cross_block=True, cross_page=True, lex=0.53; Claude: STRUCTURE/HARD): Phần trả lời chính đạt, nhưng ý thứ hai nằm ngoài quote, nên B2=false.…
  - 13-2 (csv: cross_block=True, cross_page=True, lex=0.37; Claude: STRUCTURE/HARD): Quy mô nhóm, thời gian và ngân sách (đúng các điều câu hỏi hỏi) có trong quote.…
  - 15-1 (csv: cross_block=True, cross_page=True, lex=0.69; Claude: MULTI-PART/HARD): Nếu không có cụm 'give effect to Article 3, paragraph 1, sub-paragraphs a and b', không có quote nào nói báo cáo phải bao gồm hoạt động của các chủ thể ở Art. 3(1)(a)/(b).…
  - 16-2 (csv: cross_block=True, cross_page=False, lex=0.5; Claude: STRUCTURE/HARD): Câu trả lời 'LED, which is Directive (EU) 2016/680' và việc dùng fn 5 để mô tả 'that directive' dựa vào kiến thức ngoài (LED = Law Enforcement Directive = 2016/680), nên chấm PARTIAL với missing_info_location=not_in_docu…
  - 17-1 (csv: cross_block=True, cross_page=True, lex=0.46; Claude: MULTI-PART/HARD): Mọi ý đều có trong quote. Quote p2 thuộc phần giới thiệu về EDPS của thông cáo báo chí; đây là nội dung thật, không phải header/footer, nên B8=false.…
  - 21-1 (csv: cross_block=True, cross_page=True, lex=0.42; Claude: MULTI-PART/HARD): Ba ý chính có trong quote, nhưng căn cứ 'Regulation 1025/2012 requires ESOs' của câu trả lời nằm ngoài quote, nên B2=false.…
  - 23-1 (csv: cross_block=True, cross_page=True, lex=0.38; Claude: MULTI-PART/HARD): Câu hỏi hỏi 'which obligations … exempted from', mà 2/3 nghĩa vụ nằm ngoài quote, nên chấm PARTIAL.…
  - 23-2 (csv: cross_block=True, cross_page=False, lex=0.47; Claude: STRUCTURE/HARD): Mọi ý đều có trong quote. '1025 FLOP' là 10^25 bị mất chỉ số trên khi parse; câu trả lời ghi 10^25, đây là cách đọc hợp lý.…
  - 28-1 (csv: cross_block=True, cross_page=True, lex=0.4; Claude: MULTI-PART/HARD): Hai con số chính có trong quote, nhưng việc chế tài nằm ở Art. 36 (p39, trước ngắt trang) thì ngoài quote, nên B2=false, missing_info_location=other_page.…
  - 28-2 (csv: cross_block=True, cross_page=True, lex=0.5; Claude: STRUCTURE/HARD): Mọi ý đều có trong quote. Câu 'The Directive creates general obligations…' bị PageFooter cắt đôi, và nửa ở p36 (chứa 4 mức) không nhắc tên Directive.…

Tóm tắt: trong 53 câu `cross_block=True`, Claude chấm HARD 48 câu (90.6%). Các câu `cross_block=True` nhưng chấm MEDIUM: 02-3, 03-1, 15-2, 19-2, 33-2.

[Suy luận] Mức đồng thuận cao, nhưng **không phải kiểm chứng độc lập**: khi chấm B7, Claude đã dùng `cross_block`, `cross_page`, `lexical_overlap` và vị trí split_point ở Phần A làm đầu vào. `cross_block=True` mà chấm MEDIUM thường là các quote nằm sát nhau, chỉ bị PageBreak hoặc một tiêu đề ngắn chen giữa. `lexical_overlap` gần như không dự đoán được B7 (xem bảng), vì độ khó của MULTI-PART đến từ khoảng cách giữa các đoạn, chứ không từ việc ít trùng từ.

## 5. Khuyến nghị

Kết quả: KEEP=42, FIX=57, DROP=0 (trên 99 câu).

**Chọn (b): tạo `qa_new_frozen_v2.jsonl` cùng manifest mới, áp dụng các FIX và bỏ các DROP.**

- Số câu còn lại trong v2: **99**. Số tài liệu còn đủ 3 câu: 33/33; không có tài liệu nào thiếu câu.
- 54/57 câu FIX chỉ cần sửa evidence (mở rộng hoặc thêm quote nguyên văn đã kiểm khớp đúng 1 lần), giữ nguyên câu hỏi và câu trả lời. Cách sửa này chi phí thấp và quan trọng, vì harness dùng vị trí quote làm ground truth cho Recall/nDCG.
- Câu FIX có đổi câu hỏi: 05-2, 11-3, 16-2. Câu FIX có đổi câu trả lời: 16-2.
  - Đổi vì câu trả lời dựa vào thông tin không có trong tài liệu (`not_in_document`): 16-2.
  - Đổi vì B5 AMBIGUOUS: 11-3.
  - Ngoại lệ `evidence_fix_infeasible_3_quotes`: 23-1, 29-1, 30-1. Các câu này cần nhiều hơn 3 quote (hoặc quote dài hơn 300 ký tự) mới phủ đủ câu trả lời. Theo quyết định của người dùng, chỉ sửa evidence (tối đa 3 quote, phủ được nhiều ý nhất), giữ nguyên câu hỏi và câu trả lời. Các câu vẫn còn thiếu ý sau khi sửa (`residual_missing_after_fix`): 23-1, 29-1, 30-1. Ý còn thiếu được ghi trong `reasoning` của bản ghi revision 2.
- `minor_fix` (rút gọn câu hỏi xuống ≤ 30 từ, không ảnh hưởng verdict): 03-1, 08-2, 14-1, 15-1, 30-2. Bạn quyết định có áp dụng vào v2 hay không. Nếu áp dụng thì dùng `minor_fix`, vì bản này đã gộp cả evidence đã sửa.
- Tỉ lệ EASY dự kiến của v2: 6.1%, so với 6.1% ở bản frozen. [Suy luận] Tỉ lệ EASY thấp nên nguy cơ bão hoà do câu dễ là thấp. Tuy vậy, B7 mới chỉ là suy luận.

Nhắc lại hai nguyên tắc:

1. v2 là **file mới** (`qa_new_frozen_v2.jsonl` + manifest mới có sha256 riêng). **Không ghi đè** `qa_new_frozen.jsonl`.
2. Mọi thay đổi phải làm và freeze **trước khi** xem bất kỳ kết quả retrieval nào trên bộ QA này.

## 6. Hạn chế

- **Người review (Claude) cùng họ model với model đã sinh QA** (generator trong manifest: “Claude Opus 5.5, claude.ai, 03/11/2026”). Vì vậy đánh giá có thể thiên về hướng chấm dễ, hoặc bỏ qua cùng loại lỗi mà generator mắc phải. Người dùng không chấm tay; thay vào đó dự kiến đối chiếu với một người chấm độc lập khác họ model (GPT-4.1, chạy bằng script riêng). `human_spot_check.csv` vẫn được giữ làm mẫu phân tầng tham khảo.
- **B7 `chunking_sensitivity` là suy luận**, chưa được kiểm chứng bằng retrieval thật. Các quy tắc đã dùng: hai đoạn xa nhau hoặc khác trang có ranh giới block → HARD; đại từ hoặc từ viết tắt có referent ở block khác → HARD; bảng nằm trọn trong một block → MEDIUM; một đoạn ngắn trùng nhiều từ với câu hỏi → EASY; quote liền nhau nhưng có split_point ở giữa → không chấm EASY. Các quy tắc này dùng chính các tín hiệu của validator (cross_block, cross_page, split_point), nên mức đồng thuận ở mục 4 không phải bằng chứng độc lập.
- Quy tắc chấm nghiêm (đã thống nhất): B2=false khi bất kỳ ý nào của câu trả lời, kể cả ý phụ, nằm ngoài quote. Vì vậy tỉ lệ FIX cao, nhưng phần lớn chỉ cần sửa evidence.
- Ngôn ngữ tài liệu (`doc_language`) và kiểm tra tiếng Anh ở Phần A đều dùng heuristic đếm stopword.
- B5 dựa trên tìm từ khoá trong `full_text`. Các đoạn diễn đạt khác nhưng cùng nghĩa có thể bị bỏ sót. Một số đoạn lặp nguyên văn ở phần tóm tắt (21-1, 21-2) không làm đáp án mơ hồ, nhưng có thể khiến Recall theo vị trí quote phạt oan retriever, nên đã được ghi vào `minor_issues`.
- Ràng buộc của prompt (≤ 3 quote, mỗi quote ≤ 300 ký tự) khiến một số câu không sửa được chỉ bằng evidence. Các câu này dùng ngoại lệ ở mục 5.

## 7. Mẫu chấm tay (`human_spot_check.csv`)

Chọn bằng `random.Random(2026)`, phân tầng 5 câu cho mỗi domain theo thứ tự Ada → Norma → Sophia, dùng `rng.sample` trên danh sách `qa_id` đã sắp xếp của từng domain: 02-1, 04-2, 06-2, 07-3, 03-2, 22-3, 23-1, 21-3, 18-3, 22-1, 31-3, 33-3, 31-1, 32-1, 30-3.
Verdict của Claude trong mẫu: KEEP=4, FIX=11, DROP=0.

## 8. Kiểm tra tính toàn vẹn

- sha256 `qa_new_frozen.jsonl` lúc đầu: `dca47521d0d5922b034f376496c55f960b85c604ef74d1923a31a1ccb6c8bb31`
- sha256 `qa_new_frozen.jsonl` lúc cuối: `dca47521d0d5922b034f376496c55f960b85c604ef74d1923a31a1ccb6c8bb31`, khớp

