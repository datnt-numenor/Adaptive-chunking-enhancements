# Nhiệm vụ: Kiểm tra bộ 99 QA mới đã freeze (CHỈ ĐỌC, KHÔNG SỬA)

## Bối cảnh

- Thư mục làm việc là bản clone repo `ekimetrics/adaptive-chunking` (paper "Adaptive Chunking", LREC 2026). Mọi đường dẫn bên dưới tính từ thư mục gốc này. Máy chạy Windows.
- Mình đã sinh bộ 99 QA mới (3 câu × 33 tài liệu CLAIR) để chạy lại Table 5 (đánh giá RAG). Mục đích của bộ mới là tránh việc điểm số bão hoà như với bộ QA cũ của nhóm.
- Cách sinh: mỗi tài liệu được sinh bằng một đoạn chat claude.ai độc lập, theo prompt `runs/qa_new/prompt.txt`. Prompt yêu cầu 3 loại câu hỏi, mỗi loại một câu:
  - MULTI-PART: cần ít nhất 2 chỗ trong tài liệu.
  - STRUCTURE: cần đọc bảng, danh sách, định nghĩa hoặc từ viết tắt.
  - PARAPHRASED: hỏi chi tiết nhưng không dùng lại từ ngữ của tài liệu.
- Mỗi câu có evidence là các quote nguyên văn kèm số trang.
- Bộ này đã qua `runs/validate_qa.py`: quote xuất hiện đúng 1 lần trên trang, schema đúng. Sau đó đã freeze vào `runs/qa_new/qa_new_frozen.jsonl`.
- Một lỗi đã biết: file export cũ để lộ nhãn persona trong tên tài liệu (ví dụ "Ada (technique)"), nên một số câu hỏi chứa "Ada…". Mình đã sửa tay các cụm này trước khi freeze.

**Mục tiêu:** đánh giá chất lượng ngữ nghĩa của từng câu (đúng, đủ, rõ, khó). Validator chỉ kiểm tra hình thức nên chưa làm được việc này. Kết quả dùng để quyết định giữ nguyên bộ QA, hay sửa và freeze lại thành v2 **trước khi** chạy retrieval.

## Dữ liệu

| File | Nội dung |
|---|---|
| `runs/qa_new/qa_new_frozen.jsonl` | 99 dòng: `qa_id` (dạng `01-1`), `doc_name`, `question`, `answer`, `evidence: [{page, quote}]` |
| `runs/qa_new/qa_new_frozen_manifest.json` | `sha256` của file trên, `n_qa`, `n_docs`, `generator`, `prompt_sha256`, `frozen_at` |
| `runs/qa_new/qa_review.csv` | Nhãn tự động: `n_evidence`, `cross_page`, `cross_block`, `has_table`, `lexical_overlap`, `notes` |
| `runs/qa_new/export_index.csv`, `runs/qa_new/raw/*.json` | Bảng ánh xạ file → tài liệu, và JSON gốc từ chat |
| `runs/qa_new/prompt.txt` | Prompt sinh QA |
| `data/clair/adi_parsed/<doc_name>.json` | `pages` (dict "số trang" → text), `full_text`, `split_points` (ranh giới block của parser), `titles` |

- Tên tài liệu có tiền tố persona: `Ada (technique)_…` (9 tài liệu), `Norma (légal)_…` (16), `Sophia (sciences sociales)_…` (8). Dùng tiền tố này làm nhóm lĩnh vực.
- Một số tài liệu gốc bằng tiếng Pháp, nhưng câu hỏi và câu trả lời viết bằng tiếng Anh.

## Quy tắc bắt buộc

1. **Không sửa, không ghi đè, không xoá** bất kỳ file nào trong `runs/qa_new/` (ngoài thư mục `runs/qa_new/review/`) hoặc trong `data/`. Tính sha256 của `qa_new_frozen.jsonl` khi bắt đầu và khi kết thúc. Nếu hai giá trị khác nhau, hoặc khác sha trong manifest, thì DỪNG và báo mình.
2. Chỉ ghi vào `runs/qa_new/review/`. Tạo thư mục nếu chưa có. Nếu một file output đã tồn tại, hỏi mình trước khi ghi đè. Riêng `qa_audit.jsonl` thì được nối thêm (xem quy tắc 6).
3. Không gọi API trả phí (OpenAI, gateway…). Không mở, đọc hay in file `.env`. Không cần mạng.
4. Không bịa thông tin. Mọi nhận định phải ghi rõ là **[Bằng chứng]** (trích từ file, kèm `qa_id`, trang, đoạn text) hay **[Suy luận]** (đánh giá của bạn). Nếu không kiểm chứng được, ghi đúng câu: "Không tìm thấy bằng chứng trong dữ liệu."
5. Chạy Python bằng `.venv\Scripts\python.exe -X utf8` nếu có thư mục `.venv`. Nếu không có, dùng `py -3.11 -X utf8`. Dùng UTF-8 khi đọc và ghi file.
6. Làm việc có thể tiếp tục khi bị ngắt: ghi `runs/qa_new/review/qa_audit.jsonl` sau **mỗi lô**. Khi bắt đầu lại, bỏ qua các `qa_id` đã có trong file.
7. **Không tự sửa câu hỏi.** Chỉ đề xuất cách sửa trong output. Mình sẽ quyết định có tạo bản v2 hay không.

## Phần A: Kiểm tra tự động (viết script, chạy, báo kết quả)

Viết `runs/qa_new/review/check_qa_integrity.py`, chỉ đọc dữ liệu, kiểm tra các mục sau và in PASS/FAIL cho từng mục:

1. sha256 của `qa_new_frozen.jsonl` khớp với manifest.
2. Có đúng 99 dòng, 33 tài liệu, mỗi tài liệu 3 câu. `qa_id` không trùng. Câu hỏi không trùng nguyên văn.
3. Mỗi `doc_name` có file tương ứng trong `data/clair/adi_parsed/`. Mỗi `page` tồn tại trong `pages`.
4. Mỗi quote xuất hiện **đúng 1 lần** trên trang đã ghi, so khớp chính xác vì quote đã được chuẩn hoá khi freeze. Dài 40–300 ký tự, nếu ngoài khoảng thì chỉ cảnh báo, không tính FAIL.
5. Câu hỏi không chứa nhãn persona hay nhãn lĩnh vực: `Ada`, `Norma`, `Sophia`, `technique`, `légal`, `sciences sociales`, so khớp theo từ và không phân biệt hoa thường. Không chứa cụm mơ hồ `this document|the report|the text|the paper|above`.
6. Câu hỏi dài tối đa 30 từ. Câu trả lời 1–3 câu (chỉ cảnh báo).
7. Câu trả lời bằng tiếng Anh. Dùng heuristic đơn giản và ghi rõ đó là heuristic.
8. Câu hỏi gần trùng nhau trong cùng tài liệu: Jaccard trên tập từ ≥ 0,6 thì cảnh báo.
9. Vị trí của evidence trong `full_text`, cùng số `split_points` nằm giữa các quote của cùng một câu. Số này cho biết evidence có trải qua nhiều block hay không.

Ghi `runs/qa_new/review/integrity_report.json` và in tóm tắt. **Sau Phần A, DỪNG lại, cho mình xem kết quả và chờ mình xác nhận.**

## Phần B: Review ngữ nghĩa từng câu (bạn tự đọc và đánh giá)

Với mỗi QA, chỉ đọc: câu hỏi, câu trả lời, các quote, và **ngữ cảnh ±1.500 ký tự quanh mỗi quote trên trang đó**. Đọc thêm trang khác chỉ khi cần cho tiêu chí B5. Không nạp nguyên tài liệu vào context.

Gán nhãn theo **đúng các giá trị liệt kê** dưới đây:

| Mã | Tiêu chí | Nhãn |
|---|---|---|
| B1 `grounded` | Câu trả lời có được các quote hỗ trợ không? Chỉ xét quote, không dùng kiến thức ngoài | `SUPPORTED` / `PARTIAL` / `UNSUPPORTED` / `CONTRADICTED` |
| B2 `evidence_complete` | Mọi ý trong câu trả lời đều nằm trong ít nhất một quote? Nếu một ý chỉ có ở ngữ cảnh quanh quote, ghi ý đó vào `notes` | `true` / `false` |
| B3 `self_contained` | Người chưa xem tài liệu có hiểu câu hỏi đang hỏi về luật, tổ chức hay báo cáo nào không? Câu có tự nhiên, đúng ngữ pháp không (lưu ý các câu đã sửa tay)? | `true` / `false` |
| B4 `actual_type` | Loại thật của câu hỏi. Ghi thêm `expected_type` theo vị trí (`-1` MULTI-PART, `-2` STRUCTURE, `-3` PARAPHRASED). Đây chỉ là thông tin, vì prompt không bắt buộc thứ tự. Câu MULTI-PART phải có ≥ 2 quote ở hai chỗ khác nhau | `MULTI-PART` / `STRUCTURE` / `PARAPHRASED` / `SIMPLE` |
| B5 `unique_answer` | Trong cùng tài liệu, có đoạn khác trả lời khác hoặc mâu thuẫn không? Tìm từ khoá chính trong `full_text` để kiểm tra | `UNIQUE` / `AMBIGUOUS` |
| B6 `translation` | Nếu quote là tiếng Pháp: câu trả lời tiếng Anh có dịch đúng ý không? | `OK` / `ISSUE` / `N/A` |
| B7 `chunking_sensitivity` [Suy luận] | Câu hỏi có phân biệt được chunker tốt và chunker kém không? `EASY`: một đoạn ngắn, trùng nhiều từ với câu hỏi, chunker nào cũng giữ nguyên. `HARD`: cần ≥ 2 đoạn xa nhau, hoặc bảng kèm tiêu đề của bảng, hoặc đại từ hay từ viết tắt định nghĩa ở chỗ khác. Dùng thêm `cross_page`, `cross_block`, `lexical_overlap` và khoảng cách giữa các quote từ Phần A | `EASY` / `MEDIUM` / `HARD` |
| B8 `forbidden_source` | Câu hỏi có dựa vào header, footer hoặc mục lục không? | `true` / `false` |
| `verdict` | Kết luận | `KEEP` / `FIX` / `DROP` |

Mỗi QA ghi **1 dòng JSON** vào `runs/qa_new/review/qa_audit.jsonl`:

```json
{"qa_id": "01-1", "doc_name": "...", "domain": "Ada|Norma|Sophia",
 "grounded": "...", "evidence_complete": true, "self_contained": true,
 "expected_type": "...", "actual_type": "...", "unique_answer": "...",
 "translation": "...", "chunking_sensitivity": "...", "forbidden_source": false,
 "verdict": "KEEP|FIX|DROP",
 "evidence_for_verdict": "[Bằng chứng] trích ngắn, có trang",
 "reasoning": "[Suy luận] 1-3 câu",
 "proposed_fix": {"question": "...", "answer": "...", "evidence": [{"page": 0, "quote": "..."}]} | null}
```

Quy tắc cho `verdict`:

- **DROP:** `grounded` là `UNSUPPORTED` hoặc `CONTRADICTED`, hoặc `forbidden_source` là `true`, mà không sửa được bằng cách đổi quote.
- **FIX:** có bất kỳ vấn đề nào khác (`PARTIAL`, `evidence_complete=false`, `self_contained=false`, `AMBIGUOUS`, dịch sai, MULTI-PART chỉ có 1 quote…).
  - Câu sửa phải tuân thủ `runs/qa_new/prompt.txt`.
  - Mọi quote trong `proposed_fix` phải là **chuỗi chép nguyên văn** từ trang. Tự kiểm tra bằng Python rằng quote xuất hiện đúng 1 lần trên trang đó trước khi ghi.
- **KEEP:** mọi tiêu chí đều đạt. Nếu `chunking_sensitivity` là `EASY`, vẫn giữ nhưng đánh dấu để thống kê.

Thứ tự làm:

1. **Smoke:** làm 2 tài liệu đầu tiên (6 QA). Ghi vào file, in ra bảng gọn, rồi **DỪNG chờ mình xác nhận** cách chấm.
2. Sau đó làm theo lô **5 tài liệu**. Ghi file sau mỗi lô và báo tiến độ trong 1–2 dòng. Không cần dừng giữa các lô, trừ khi gặp vấn đề hệ thống, ví dụ nhiều quote không khớp.

## Phần C: Tổng hợp

1. Viết script `runs/qa_new/review/summarize_audit.py` đọc `qa_audit.jsonl`, rồi ghi `runs/qa_new/review/qa_audit_summary.md` (bằng tiếng Việt). Mọi con số trong summary phải sinh từ script, không được tự gõ tay.
   - Số lượng mỗi nhãn của B1–B8 và `verdict`: tổng, theo domain (Ada/Norma/Sophia), theo vị trí câu (-1/-2/-3).
   - Phân bố `chunking_sensitivity`, cùng tỉ lệ `EASY`. Đây là chỉ báo nguy cơ bão hoà.
   - Bảng mọi câu `FIX` và `DROP`, gồm `qa_id`, vấn đề, đề xuất sửa.
   - So sánh nhãn tự động trong `qa_review.csv` với đánh giá của bạn, ví dụ `cross_block=True` mà bạn chấm `EASY`.
2. Viết `runs/qa_new/review/human_spot_check.csv`:
   - 15 QA chọn ngẫu nhiên với `random.Random(2026)`, phân tầng 5 câu cho mỗi domain.
   - Cột: `qa_id`, `question`, `answer`, `quotes`, `claude_verdict`, cùng hai cột để trống `human_verdict` và `human_notes` để mình tự chấm.
3. Ở cuối summary, viết phần **Khuyến nghị**, chọn một trong ba:
   - (a) Giữ nguyên bộ QA đã freeze.
   - (b) Tạo `qa_new_frozen_v2.jsonl` cùng manifest mới, áp dụng các FIX và bỏ các DROP. Ghi rõ số câu còn lại, và mỗi tài liệu có còn đủ 3 câu không.
   - (c) Sinh lại câu hỏi cho một số tài liệu.

   Nhắc lại hai nguyên tắc: v2 là file mới, không ghi đè bản cũ; và mọi thay đổi phải làm **trước khi** xem bất kỳ kết quả retrieval nào.
4. Phần **Hạn chế** phải nêu:
   - Người review (Claude) cùng họ model với model đã sinh QA, nên có thể thiên vị theo hướng chấm dễ. Vì vậy cần đối chiếu với phần mình tự chấm tay.
   - B7 là suy luận, chưa kiểm chứng bằng retrieval thật.
5. Kết thúc:
   - Tính lại sha256 của `qa_new_frozen.jsonl` và so với giá trị lúc bắt đầu.
   - Liệt kê mọi file đã tạo.
   - Báo các lỗi hoặc ngoại lệ đã gặp, kể cả các câu bỏ qua.
