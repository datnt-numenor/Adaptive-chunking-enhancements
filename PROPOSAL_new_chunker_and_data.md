# Đề xuất: Chunker mới theo cấu trúc tài liệu và mở rộng bộ dữ liệu đánh giá

*Bản nháp để thảo luận với trưởng nhóm và thầy cô. Ngày: 04/11/2026.*

Quy ước trích dẫn theo `AGENTS.md`:

- **[Paper]**: điều paper nói.
- **[Code/Artifact]**: điều code, log hoặc kết quả đã lưu cho thấy, kèm đường dẫn.
- **[Suy luận]**: diễn giải hoặc giả thuyết của người viết, chưa được kiểm chứng.

Câu chuẩn dùng khi không có bằng chứng: *"Không tìm thấy bằng chứng trong paper/code."*

---

## 0. Tóm tắt

1. Hướng của thầy cô là **tạo một chunker mới cho số liệu tốt hơn Adaptive Chunking**. Hướng này khả thi, với một điều kiện: phải định nghĩa trước "tốt hơn" là tốt hơn ở **số liệu nào**.
2. Đề xuất:
   - Thước đo chính là **retrieval trên bộ QA đã freeze**: nDCG@10, Recall@k, MRR@10.
   - 5 metric nội tại của paper chỉ là thước đo phụ.
   - Mốc phải vượt là **phương pháp cố định tốt nhất chọn bằng cross-validation**, không chỉ Adaptive.
3. Ứng viên đầu tiên là **gom block theo cấu trúc** (structure-aware block packing). Phương pháp này tất định, chạy trên CPU và rẻ. Dù kết quả dương hay âm thì đều đưa được vào bài.
4. 33 tài liệu chỉ đủ phát hiện mức chênh nDCG@10 khoảng 4–5 điểm trở lên. Đề xuất bổ sung một **tập test độc lập** gồm tài liệu mới, freeze trước khi thử phương pháp. Bắt đầu bằng một pilot 5 tài liệu.

---

## 1. Câu hỏi cần thầy cô trả lời trước

> "Tốt hơn về số liệu" là tốt hơn ở **(a) 5 metric nội tại** (Table 3), **(b) retrieval** (nDCG/Recall/MRR), hay **(c) chất lượng câu trả lời** (Table 5)?

Câu trả lời quyết định thiết kế toàn bộ thí nghiệm. Tài liệu này đề xuất chọn (b) làm thước đo chính, (a) và (c) làm thước đo phụ. Lý do ở mục 2.

---

## 2. Vì sao không nên chỉ nhắm vào 5 metric nội tại

### 2.1 Bẫy 1: metric nội tại dễ bị "luyện đề"

**[Paper]**
- Adaptive Chunking chọn, cho từng tài liệu, phương pháp có điểm trung bình cao nhất trên 5 metric nội tại, với trọng số bằng nhau.
- 5 metric gồm: RC (Relative Cohesion), ICC (Intra-chunk Cohesion), DCC (Document Contextual Coherence), BI (Block Integrity), SC (Size Compliance).

**[Code/Artifact]**
- Adaptive của nhóm đạt 90,95 trên 4 ứng viên đã khóa. Con số này thấp hơn paper khoảng 0,12 điểm. Hàng `best = 91,06` trong CLI cũ không dùng được, vì khi đó cả 8 phương pháp cùng tham gia lựa chọn. Nguồn: `research/week3/PAPER_EXPERIMENT_COMPARISON_REPORT.md:72`, `research/week2/RESULTS.md:155`.
- 1.730/1.732 ranh giới trang (99,88%) trùng với `split_points` của parser trong phạm vi 5 ký tự. Bỏ BI làm thay đổi lựa chọn ở 9/33 tài liệu: `page` giảm từ 15 xuống 9, `our_recurs_1100` tăng từ 13 lên 20. Nguồn: `research/week3/HANDOFF.md:126-129`. Báo cáo audit cá nhân ghi phát hiện này là F6.

**[Suy luận]** Có thể viết một chunker đạt điểm nội tại cao hơn Adaptive mà chunk không thật sự tốt hơn:

| Metric | Cách "luyện đề" |
|---|---|
| BI | Chỉ cắt đúng tại `split_points` của parser, khi đó BI ≈ 100% theo định nghĩa |
| SC | Gom block cho đến khi chunk nằm trong khoảng 100–1.100 token, khi đó SC ≈ 100% |
| RC | Các cặp thực thể–đại từ đã có sẵn trong `mentions/`; chỉ cần tránh cắt giữa chúng |

Một chunker như vậy gần như chắc chắn thắng Table 3, nhưng đó là tối ưu thẳng vào thước đo, không chứng minh được chunk tốt hơn.

### 2.2 Metric nội tại cao không kéo theo retrieval tốt hơn

**[Code/Artifact]** Kết quả retrieval của nhóm trên 99 QA cũ, hybrid retrieval, top-10, nguồn `research/week3/HANDOFF.md:131-132, 292-296`:

| Hệ thống | nDCG@10 (Phase J) |
|---|---|
| Adaptive gốc | 0,7521 |
| Adaptive BI-neutral | 0,7690 |
| raw page | 0,7270 |
| raw LangChain recursive (default) | 0,7798 |
| **Best-fixed (CV)** | **0,8025** |

- Ở Phase H, hiệu Adaptive − raw `langch_recurs_1100` là −0,0504, với 95% CI bootstrap theo tài liệu là [−0,1006; −0,0035]. Sau hiệu chỉnh Holm, p = 0,3266, nên không có ý nghĩa thống kê.
- Ở Phase J, hiệu BI-neutral − Adaptive là +0,0169, CI [−0,0097; +0,0457], Holm p = 0,5589, cũng không có ý nghĩa thống kê.

**[Suy luận]** Adaptive đứng đầu Table 3 nhưng không đứng đầu retrieval. Vậy metric nội tại không phải là đại diện đáng tin cho chất lượng retrieval trên bộ dữ liệu này.

### 2.3 Bẫy 2: "thắng Adaptive" là mốc so sánh quá thấp

- **Về retrieval [Code/Artifact]:** trên dữ liệu của nhóm, Adaptive đã thấp hơn best-fixed CV về điểm trung bình. Nên một phương pháp chỉ thắng Adaptive thì chưa chứng minh được gì. `AGENTS.md` cũng bắt buộc so sánh với "the best fixed chunker selected without test-set information".
- **Về metric nội tại [Suy luận]:** Adaptive lấy điểm cao nhất trong các ứng viên. Nếu đưa phương pháp mới vào làm ứng viên thứ 5, Adaptive luôn ≥ phương pháp đó theo cách xây dựng. Vì vậy phải so sánh phương pháp mới **đứng riêng**, và báo cáo riêng biến thể "Adaptive + ứng viên mới".

---

## 3. Thiết kế đánh giá đề xuất

### 3.1 Thước đo

| Vai trò | Thước đo |
|---|---|
| **Chính** | nDCG@10 trên bộ QA đã freeze, dùng đúng định nghĩa source-span của harness `research/week3/phase_g_runner.py` |
| Phụ (retrieval) | Recall@1/3/5/10, Hit@k, MRR@10 |
| Phụ (nội tại) | 5 metric của paper, báo cáo cả bản gốc và bản BI page-neutral |
| Phụ (downstream) | Chất lượng câu trả lời (Retrieval Completeness và G-Eval correctness) cho 2–3 cấu hình mạnh nhất, nếu còn ngân sách |
| Chi phí | Số chunk, độ dài chunk, thời gian chunking và index |

### 3.2 Mốc so sánh bắt buộc

1. Adaptive gốc: 4 ứng viên, trọng số bằng nhau.
2. Best-fixed chọn bằng cross-validation theo tài liệu. Đây là mốc chính cần vượt.
3. Oracle theo tài liệu, chỉ dùng làm cận trên.
4. Các baseline cố định của paper: raw page, raw LangChain recursive.

### 3.3 Thống kê

- Đơn vị thống kê là **tài liệu**: lấy trung bình các câu hỏi trong cùng tài liệu trước.
- Bootstrap theo tài liệu với 10.000 mẫu, báo cáo 95% CI của hiệu.
- Wilcoxon signed-rank hai phía (xử lý giá trị 0 theo Pratt), kèm hiệu chỉnh Holm trên họ các so sánh.
- Báo cáo **cả effect size lẫn CI**, không chỉ p-value.
- Mọi tham số (kích thước tối đa, ngưỡng, trọng số) **chỉ được chọn bằng `StratifiedGroupKFold` theo tài liệu** (seed 2026, fold trong `prepared/folds.jsonl`). Không chỉnh tham số sau khi đã xem kết quả trên toàn bộ 33 tài liệu.

### 3.4 Giả thuyết (ghi trước khi chạy)

- **H1:** Chunker chỉ cắt tại ranh giới block của parser, giữ tiêu đề đi cùng nội dung và không cắt bảng, đạt nDCG@10 ≥ best-fixed CV.
- **H2:** Nếu H1 sai nhưng chunker vẫn có điểm nội tại ≥ Adaptive, thì đó là bằng chứng rằng bộ metric nội tại bị "luyện đề" được. Điều này nối trực tiếp với F6.
- **Điểm yếu nhắm tới [Suy luận, từ audit]:**
  - Recursive cắt theo số ký tự nên có thể cắt giữa bảng hoặc tách tiêu đề khỏi nội dung.
  - `page` cắt theo trang nên có thể tách một đoạn văn kéo dài qua hai trang.
  - `llm_regex` có fallback im lặng (F3).

---

## 4. Các phương pháp ứng viên

| # | Phương pháp | Ý tưởng | Dựa trên | Chi phí | Độ mới |
|---|---|---|---|---|---|
| 1 | **Gom block theo cấu trúc** | Chỉ cắt tại `split_points`; tiêu đề luôn dính với block nội dung theo sau; không cắt bảng; gom tham lam các block liên tiếp đến khi đạt `max_tokens`, gộp chunk dưới `min_tokens` vào chunk kề | Cấu trúc mà BI đo, nhưng kiểm tra bằng retrieval | Rẻ: tất định, CPU | Trung bình |
| 2 | **Cắt theo cấu trúc, điểm cắt theo ngữ nghĩa** | Ứng viên điểm cắt chỉ là `split_points`; chọn tập điểm cắt bằng quy hoạch động, tối đa hoá độ tương đồng embedding trong chunk trừ phạt lệch kích thước | Semantic chunking nhưng không cắt giữa block, nên tránh chunk quá dài *(cần trích lại độ dài chunk semantic lớn nhất từ `check_chunks.py`)* | Trung bình: cần embedding, chạy được trên Kaggle | Cao hơn |
| 3 | **LLM-regex bền hơn** | Chấp nhận khối \`\`\`; kiểm tra regex có thật sự cắt được tài liệu không; nếu không thì có chuỗi phương án dự phòng | F3 | Rẻ: phản hồi đã có cache | Thấp, vì chỉ là sửa phương pháp cũ |

**Thứ tự đề xuất:** làm 1 trước. Chỉ làm 2 nếu 1 cho tín hiệu dương trên tập phát triển. 3 chỉ là phụ lục hoặc sửa lỗi.

### 4.1 Kế hoạch cho phương pháp 1

1. Viết splitter mới cắm vào pipeline chunking hiện có. Code đặt trong `research/`, không sửa upstream.
2. Tham số: `max_tokens ∈ {600, 800, 1100}`, `min_tokens = 100`. Chọn bằng CV theo tài liệu, tiêu chí là nDCG@10 trên fold train.
3. Chạy smoke test trên 2 tài liệu trước khi chạy đủ 33 tài liệu.
4. Chấm metric nội tại (bản gốc và bản page-neutral), rồi chạy index → retrieve → evaluate bằng harness `phase_g_runner.py`. Giữ nguyên bộ QA đã freeze, fold, model và revision.
5. Báo cáo theo mục 3, so sánh với cả 4 nhóm mốc ở mục 3.2.

### 4.2 Các kết quả có thể xảy ra, và kết quả nào cũng dùng được

| Kết quả | Ý nghĩa cho bài báo |
|---|---|
| Thắng best-fixed về retrieval | Đóng góp chính; phương pháp 2 là bản nâng cấp |
| Thắng Adaptive nhưng thua best-fixed | Cải tiến có giới hạn; báo cáo trung thực |
| Chỉ thắng ở metric nội tại | Bằng chứng mạnh rằng metric nội tại "luyện đề" được; đưa vào phần audit |
| Không thắng ở đâu | Kết quả âm tính; vẫn báo cáo, kèm phân tích lỗi |

---

## 5. Liên hệ với kế hoạch của nhóm

- Hướng này **thay thế Đóng góp 1** (selector hiệu chỉnh theo downstream) trong `AGENTS.md`. Nó vẫn đúng nguyên tắc "thay một thành phần", vì chỉ thay chunker, không xây hệ thống RAG mới. Cần trưởng nhóm đồng ý.
- **Dùng lại nguyên vẹn** hạ tầng hiện có: bộ QA đã freeze, fold chia theo tài liệu, model đã pin revision, harness `research/week3`.
- Mỗi phương pháp mới tốn khoảng 1 lần index và retrieval, ước tính vài USD thuê GPU hoặc chạy trên Kaggle *[Suy luận, từ chi phí các lần chạy trước]*.
- **Đóng góp 2** (thêm tiêu đề vào chunk, V0/V1/V2) vẫn kết hợp được. Ranh giới chunk và cách biểu diễn chunk là hai biến khác nhau, nên phải đo riêng từng biến trước khi kết hợp.

---

## 6. Mở rộng bộ dữ liệu

### 6.1 Vì sao cần

**[Code/Artifact]** Mọi kiểm định của nhóm lấy 33 tài liệu làm đơn vị. CI của hiệu nDCG@10 ở Phase J rộng khoảng 0,055 (từ −0,0097 đến +0,0457).

**[Suy luận]**
- Với 33 tài liệu, chỉ phát hiện ổn định được mức chênh khoảng 4–5 điểm nDCG@10 trở lên.
- Cả 33 tài liệu đều đã được xem khi phân tích. Vì vậy, nếu thiết kế chunker trên chúng thì không còn tập nào "chưa dùng lúc thiết kế", trái với yêu cầu của `AGENTS.md`.

### 6.2 Cần bao nhiêu tài liệu (ước lượng thô)

Cách tính:
- Từ CI Phase J: SE ≈ 0,0277 / 1,96 ≈ 0,014.
- Độ lệch chuẩn của hiệu theo tài liệu: σ_d ≈ 0,014 × √33 ≈ 0,08.
- Cỡ mẫu cặp đôi: n ≈ ((z₀,₉₇₅ + z₀,₈) · σ_d / Δ)² = (2,8 × 0,08 / Δ)².

| Mức chênh nDCG@10 muốn phát hiện (Δ) | Số tài liệu cần (power 80%, α = 0,05, một so sánh) |
|---|---|
| 0,05 | ≈ 20 |
| 0,03 | ≈ 55 |
| 0,02 | ≈ 125 |
| 0,01 | ≈ 500 |

Lưu ý [Suy luận]:
- σ_d ước từ một cặp so sánh duy nhất, nên bản thân con số này cũng không chắc chắn.
- Wilcoxon cần thêm khoảng 5% so với xấp xỉ t.
- Với nhiều so sánh và hiệu chỉnh Holm, số tài liệu cần còn lớn hơn.
- Muốn chứng minh mức hơn khoảng 2 điểm thì cần khoảng **100–150 tài liệu**.

### 6.3 Hai cách tăng dữ liệu (nên kết hợp)

| Cách | Tác dụng | Giới hạn |
|---|---|---|
| Thêm câu hỏi cho 33 tài liệu cũ (từ 3 lên 8–10 câu mỗi tài liệu) | Rẻ (không cần parse, chunk hay tính metric lại); giảm nhiễu trong từng tài liệu | **Không** tăng số đơn vị thống kê |
| Thêm tài liệu mới | Tăng số đơn vị thống kê thật sự; tạo được tập test độc lập | Tốn công parse, tính metric và review QA |

### 6.4 Nguồn tài liệu mới

| Nguồn | Ưu điểm | Nhược điểm |
|---|---|---|
| Benchmark có câu hỏi do người viết kèm bằng chứng (ví dụ Qasper, FinanceBench, CUAD) | Không tốn công sinh và review QA; tránh thiên lệch do LLM sinh; tự nhiên thành tập test độc lập | Lĩnh vực khác CLAIR; phải tự lấy PDF gốc và parse; **chưa xác minh license, quy mô và việc có PDF gốc hay không** |
| PDF công khai cùng lĩnh vực CLAIR (văn bản luật, tiêu chuẩn, báo cáo chính sách AI) | So sánh trực tiếp được với CLAIR | Phải tự sinh QA và review thủ công. Ở bộ QA cũ, nhóm giữ nguyên 62 câu và phải sửa 37 câu (`research/week3/HANDOFF.md:236`) |

### 6.5 Khó khăn cần lường trước

- **Parser:**
  - [Paper] dùng Azure Document Intelligence kèm pipeline Markdown riêng.
  - [Code] repo có `DoclingParser` miễn phí.
  - [Suy luận] Parser khác cho `split_points` khác, nên BI không so sánh trực tiếp được giữa hai bộ dữ liệu. Cách xử lý: coi bộ mới là một tập đánh giá riêng và báo cáo theo từng bộ. Kết quả trên bộ mới cũng trả lời câu hỏi "F6 có lặp lại với parser khác không".
- **RC:** cần maverick-coref. Gói này không cài được trên Windows của người viết (lỗi decode cp1252), nên phải chạy trên Kaggle hoặc Linux. RC cũng chỉ áp dụng cho tài liệu tiếng Anh.
- **Độ dài tài liệu:** một tài liệu rất dài (khoảng 390k token) đã chiếm khoảng 1/3 thời gian tính metric trong lần tái lập. Nên chọn tài liệu khoảng 5k–50k token.

### 6.6 Chi phí ước tính cho 100 tài liệu mới

*[Suy luận: ngoại suy từ các lần chạy trước, giả sử độ dài tương tự CLAIR. Cần đo lại bằng pilot ở mục 6.7.]*

| Hạng mục | Ước tính |
|---|---|
| Sinh QA bằng GPT-4.1 | Vài USD |
| LLM-regex | Khoảng $2–3 |
| Metric nội tại trên Kaggle | Khoảng 30 giờ GPU (khoảng 1 tuần quota) |
| Index và retrieval | Vài USD thuê GPU |
| Review QA thủ công | **Tốn thời gian nhất** |

Tổng ngân sách API của dự án là USD 60–80 (`AGENTS.md`). Mọi lần chạy trả phí cần ước tính và được trưởng nhóm duyệt trước.

### 6.7 Quy trình đề xuất

1. **Tập phát triển:** 33 tài liệu CLAIR. Dùng để thiết kế và tinh chỉnh chunker, chạy CV thoải mái.
2. **Tập test:** bộ tài liệu mới. Thu thập, parse, sinh QA và review, rồi **freeze trước khi thử bất kỳ phương pháp nào**. Chỉ chạy **một lần ở cuối**, và báo cáo kể cả khi kết quả âm tính.
3. **Pilot trước:** chạy khoảng 5 tài liệu mới qua toàn bộ pipeline (Docling → chunking → metric nội tại → QA → retrieval) để đo thời gian và chi phí thật trước khi mở rộng.

---

## 7. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Mức cải thiện 1–2 điểm nDCG không đạt ý nghĩa thống kê với 33 tài liệu | Báo trước cho thầy cô; ưu tiên effect size + CI; bổ sung tập test |
| Chỉnh tham số theo kết quả toàn bộ 33 tài liệu (overfit tập phát triển) | Chỉ chọn tham số bằng CV; tập test freeze và chỉ chạy một lần |
| Bộ dữ liệu mới làm thay đổi bộ QA đã freeze | Giữ nguyên bộ cũ; bộ mới là tập riêng với sha256 và manifest riêng |
| Retrieval bão hoà (điểm quá cao, khó phân biệt các phương pháp) | Dùng bộ QA khó hơn (đang thử với bộ 99 QA mới); báo cáo cả Recall@1/3 |
| Vượt ngân sách API hoặc quota GPU | Ước tính trước mỗi lần chạy; pilot nhỏ; cache mọi phản hồi |

---

## 8. Cần quyết định

1. **Thầy cô:** "số liệu" là metric nội tại, retrieval hay chất lượng câu trả lời?
2. **Trưởng nhóm:** hướng chunker mới có thay thế Đóng góp 1 (selector) không?
3. **Trưởng nhóm và Member 1 (phụ trách dữ liệu):** có mở rộng dữ liệu không? Lấy nguồn nào? Quy mô bao nhiêu? Ngân sách bao nhiêu?

## 9. Việc tiếp theo nếu được đồng ý

- [ ] Viết splitter "gom block theo cấu trúc" trong `research/`, kèm unit test và script chấm metric nội tại.
- [ ] Smoke test 2 tài liệu, rồi chạy đủ 33 tài liệu: metric nội tại, sau đó retrieval qua harness `phase_g_runner.py`.
- [ ] Phân tích thống kê theo mục 3.3 so với Adaptive, best-fixed CV và oracle.
- [ ] (Song song, nếu được duyệt) script parse 5 PDF bằng Docling sang định dạng JSON của repo, rồi chạy pilot mục 6.7.
