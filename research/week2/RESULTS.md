# Kết quả tái lập Week 2

## 1. Trạng thái và phạm vi

Quy trình tái lập từ các tài liệu CLAIR đã parse sẵn đến kết quả intrinsic
metrics đã hoàn thành cho 33 tài liệu và tám phương pháp chunking. Kết quả cuối
được lưu tại [`artifacts/final_table3/`](artifacts/final_table3/).

Phạm vi đã hoàn thành:

- khôi phục và xác thực bảy phương pháp không dùng semantic GPU;
- chạy semantic chunking đúng cấu hình upstream trên RunPod;
- hợp nhất đủ tám phương pháp và ba stage xử lý;
- tính processed metrics và raw metrics bằng model Jina chạy local trên Kaggle;
- tái tạo các bảng phân tích do CLI gọi là Table 1, Table 2 và fixed-method
  Table 3;
- kiểm tra schema, số tài liệu, phương pháp, số hàng và duplicate keys;
- chạy bộ kiểm thử cuối: `63 passed, 1 skipped`.

Chưa thực hiện trong Week 2:

- RAG retrieval/downstream answer-quality evaluation;
- document-grouped train/validation/test cho selector học được;
- Block Integrity page-neutral ablation;
- tái tính statistical significance hoặc confidence intervals.

Vì vậy, báo cáo này chỉ kết luận về việc tái lập chunking và intrinsic metrics;
không kết luận rằng một selector cải thiện chất lượng RAG downstream.

## 2. Nguồn bằng chứng

Các kết luận dưới đây được kiểm tra từ artifact thay vì chỉ dựa vào console:

- commit upstream: `ea87ce8e1a97888f3f179e7f1359ff7f43fb179d`;
- merge manifest: [`artifacts/final_table3/metrics_merge_manifest.json`](artifacts/final_table3/metrics_merge_manifest.json);
- validation report: [`artifacts/final_table3/validation_report.json`](artifacts/final_table3/validation_report.json);
- analysis/Table 3 log: [`artifacts/final_table3/analysis_table3.log`](artifacts/final_table3/analysis_table3.log);
- semantic manifest: [`artifacts/semantic_output/semantic_run_manifest.json`](artifacts/semantic_output/semantic_run_manifest.json);
- audit paper--README--code: [`paper_code_audit.md`](paper_code_audit.md);
- trạng thái và lịch sử lỗi: [`HANDOFF.md`](HANDOFF.md).

Các giá trị `paper mean` trong Table 3 bên dưới là reference values mà
`adaptive_chunking.paper.replicate` dùng để so sánh. Chúng không phải số được
tính lại từ artifact local.

## 3. Cấu hình tái lập

### 3.1 Dataset và code

| Thành phần | Cấu hình |
|---|---|
| Corpus | 33 tài liệu CLAIR đã parse sẵn |
| Phương pháp | page, sentence, LangChain recursive default/1100, recursive 600/1100 của paper, semantic, LLM-regex |
| Source commit | `ea87ce8e1a97888f3f179e7f1359ff7f43fb179d` |
| Processed stage dùng cho processed metrics | `small_merged` |
| Raw methods dùng cho raw metrics | page, sentence, semantic, LangChain recursive default/1100 |
| Intrinsic metrics | RC, ICC, DCC, BI, SC; trọng số bằng nhau 20% |
| API secrets trong metric runs | không dùng `JINA_API_KEY` hoặc `OPENAI_API_KEY` |

Manifest semantic lưu SHA256 và số token của từng tài liệu. Không tìm thấy một
dataset version ID độc lập trong paper/code; các hash trong manifest là định
danh dữ liệu thực tế của lần chạy này.

### 3.2 Semantic chunking

| Thành phần | Giá trị |
|---|---|
| Thiết bị | NVIDIA RTX A5000, compute capability 8.6 |
| Hệ điều hành | Linux 6.8, glibc 2.35 |
| Python | 3.11.10 |
| PyTorch | 2.6.0+cu124 |
| Model | `Qwen/Qwen3-Embedding-0.6B` |
| Model revision | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| Attention / dtype | FlashAttention 2 / bfloat16 |
| Batch size | 16 |
| Hoàn thành | 33/33 tài liệu |
| Runner time | 235.42 giây |

Pod RunPod đã được terminate sau khi archive local và Parquet framing được xác
thực. Chi phí suy ra từ uptime khoảng USD 0.19; billing record chưa settle tại
lần kiểm tra cuối, nên đây là ước lượng chứ không phải hóa đơn cuối.

### 3.3 Metric computation

| Thành phần | Giá trị |
|---|---|
| Thiết bị | Kaggle Tesla T4, compute capability 7.5 |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| Embedding model | `jinaai/jina-embeddings-v3` |
| Model revision | `ab036b023d30b4d1138c4c3bfa9f0c445ab455d6` |
| Remote-code repository | `jinaai/xlm-roberta-flash-implementation` |
| Remote-code revision | `bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3` |
| Processed batch size | 4 |
| Raw batch size | 1 cho shard 00--04; 4 cho shard 05 |
| Số run hợp lệ | 6 processed + 6 raw |

Batch size raw được giảm vì các semantic chunks raw rất dài. Thay đổi này giữ
nguyên tài liệu, chunk boundaries, model, revision và metric definitions; nó là
một điều chỉnh vận hành để vừa VRAM T4. Sai khác floating-point rất nhỏ do batch
shape vẫn có thể tồn tại và không được khẳng định là bằng bit.

## 4. Kiểm tra tính toàn vẹn

### 4.1 Chunk artifacts

| Stage | Rows | Documents | Methods |
|---|---:|---:|---:|
| `raw` | 21,253 | 33 | 8 |
| `no_oversizing` | 22,675 | 33 | 8 |
| `small_merged` | 19,473 | 33 | 8 |

Bốn page chunks rỗng tồn tại ở hai stage trung gian `raw` và
`no_oversizing`; không còn chunk rỗng tại `small_merged`. Validator chấp nhận
trường hợp này theo stage.

### 4.2 Metric artifacts

| Kind | Rows | Documents | Methods | Metrics |
|---|---:|---:|---:|---:|
| Processed | 2,640 | 33 | 8 | 10 |
| Raw | 1,650 | 33 | 5 | 10 |

Các key `(doc_name, chunking_method, metric_name)` không bị trùng và số hàng
đúng bằng tích của documents, methods và metrics.

Null scores được giữ nguyên, không điền giá trị giả:

- processed: 16 RC null và 6 DCC null;
- raw: 10 RC null và 2 DCC null.

RC có thể không xác định khi tài liệu không có entity--pronoun pairs. Nguyên
nhân cụ thể của toàn bộ DCC null chưa được audit riêng; không tìm thấy bằng
chứng trong artifact hiện tại để khẳng định thêm. Khi tính weighted mean, code
hiện tại bỏ metric null và chuẩn hóa lại trên các metric còn lại; paper không
mô tả rõ hành vi này.

## 5. Kết quả phân tích

### 5.1 Bảng phân tích processed metrics (CLI “Table 1”)

Giá trị là mean ± standard deviation theo tài liệu. `final` là trung bình có
trọng số bằng nhau của năm intrinsic metrics.

| Method | SC | BI | ICC | DCC | RC | Final |
|---|---:|---:|---:|---:|---:|---:|
| page | 99.9±0.4 | 99.9±0.3 | 69.2±3.6 | 86.4±4.6 | 97.2±3.5 | 90.52 |
| sentence | 100.0±0.0 | 68.3±7.8 | 77.2±2.7 | 75.4±5.4 | 89.5±9.7 | 82.07 |
| langch_recurs_default | 98.7±2.9 | 95.2±2.4 | 65.5±3.6 | 89.0±2.1 | 96.1±4.3 | 88.89 |
| langch_recurs_1100 | 99.4±1.7 | 98.8±1.3 | 64.6±3.5 | 88.3±2.8 | 98.6±3.1 | 89.93 |
| our_recurs_1100 | 100.0±0.1 | 98.1±1.9 | 66.6±3.8 | 89.8±2.5 | 99.0±1.8 | **90.68** |
| our_recurs_600 | 100.0±0.0 | 94.8±3.6 | 69.5±4.1 | 84.7±3.1 | 97.2±2.9 | 89.24 |
| semantic | 98.9±5.8 | 92.5±2.9 | 69.1±3.8 | 84.6±3.6 | 97.7±3.6 | 88.57 |
| llm_regex | 99.6±1.1 | 98.8±1.4 | 71.3±5.0 | 82.7±6.0 | 97.5±3.9 | 89.99 |
| best | 99.9±0.3 | 99.4±1.2 | 68.6±4.6 | 88.3±4.4 | 99.0±1.5 | **91.06** |

Lưu ý quan trọng: lệnh analysis truyền cả tám methods vào code chọn `best`.
Paper đánh dấu bốn ứng viên Adaptive bằng `*`. Vì vậy, hàng `best = 91.06`
phản ánh hành vi code hiện tại, không phải bằng chứng tái lập protocol selector
bốn ứng viên của paper.

### 5.2 Meta metrics (CLI “Table 2”)

| Method | Avg tokens | Max tokens | Min tokens | Std tokens | Total chunks |
|---|---:|---:|---:|---:|---:|
| page | 663 | 1,146 | 72 | 235 | 1,780 |
| sentence | 172 | 606 | 100 | 58 | 6,882 |
| langch_recurs_default | 773 | 1,364 | 118 | 127 | 1,556 |
| langch_recurs_1100 | 742 | 1,146 | 69 | 294 | 1,593 |
| our_recurs_1100 | 878 | 1,141 | 104 | 217 | 1,345 |
| our_recurs_600 | 496 | 691 | 102 | 101 | 2,381 |
| semantic | 625 | 1,140 | 96 | 320 | 1,891 |
| llm_regex | 577 | 1,144 | 76 | 344 | 2,045 |
| best | 711 | 1,146 | 86 | 256 | 1,660 |

Tên “Table 1/2” ở CLI không ánh xạ hoàn toàn với numbering trong paper. Khi
trích dẫn báo cáo, nên mô tả nội dung bảng thay vì chỉ dùng số thứ tự.

### 5.3 Tái lập fixed-method Table 3

`*` dùng `small_merged`; `†` dùng raw chunks. RC, ICC, DCC, BI và SC là số
local; `paper mean` là reference value; `delta = local mean - paper mean`.

| Method | Tag | RC | ICC | DCC | BI | SC | Local mean | Paper mean | Delta |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| our recursive (s=1100) | * | 99.0±1.8 | 66.6±3.8 | 89.8±2.5 | 98.1±1.9 | 100.0±0.1 | 90.68 | 90.68 | -0.00 |
| our recursive (s=600) | * | 97.2±2.9 | 69.5±4.1 | 84.7±3.1 | 94.8±3.6 | 100.0±0.0 | 89.24 | 89.24 | +0.00 |
| page (post-processed) | * | 97.2±3.5 | 69.2±3.6 | 86.4±4.6 | 99.9±0.3 | 99.9±0.4 | 90.52 | 90.52 | -0.00 |
| LLM regex | * | 97.5±3.9 | 71.3±5.0 | 82.7±6.0 | 98.8±1.4 | 99.6±1.1 | 89.99 | 89.80 | +0.19 |
| LC recursive (s=1100) | † | 98.4±3.1 | 64.7±3.4 | 86.8±2.8 | 98.6±1.4 | 93.3±7.2 | 88.36 | 88.37 | -0.01 |
| LC recursive (default) | † | 96.1±4.3 | 65.6±3.6 | 88.8±2.4 | 95.0±2.8 | 97.7±6.3 | 88.62 | 88.62 | -0.00 |
| page (raw) | † | 97.1±3.5 | 69.3±3.4 | 86.1±5.0 | 100.0±0.0 | 92.7±9.7 | 89.03 | 89.03 | -0.00 |
| semantic | † | 96.9±3.9 | 69.2±3.9 | 76.4±7.2 | 91.4±3.6 | 48.9±16.8 | 76.56 | 76.49 | +0.07 |
| sentence | † | 86.6±10.4 | 78.6±2.1 | 72.0±5.9 | 60.9±9.2 | 61.7±21.5 | 71.96 | 73.26 | **-1.30** |

Bảy trong chín rows lệch không quá 0.07 điểm phần trăm so với reference. Hai
rows còn lại là LLM-regex (+0.19) và sentence raw (-1.30). Đây là mô tả định
lượng; chưa có document-level uncertainty/test nên không diễn giải các delta
nhỏ này là khác biệt có ý nghĩa thống kê.

## 6. Lỗi, recovery và thay đổi cần ghi nhận

1. Raw shard 04 version 1 hết VRAM ở tài liệu GDPR. Một semantic chunk dài
   10,113 token và batch size 4 khiến attention cố cấp thêm 8.01 GiB trên T4.
   Version 2 chạy batch size 1 và validation đạt `status: ok`.
2. Trước các raw shard tiếp theo, độ dài semantic chunk lớn nhất được audit.
   Raw shards 00--03 chạy batch size 1; shard 03 chứa chunk dài 17,146 token và
   vẫn hoàn thành.
3. Processed shard 05 là pilot hợp lệ. Kernel cũ bị đánh dấu lỗi sau khi metric
   đã lưu xong vì NumPy ABI ở bước post-validation. Parquet được tải về và xác
   thực local: 400 rows, 5 documents, 8 methods, 10 metrics.
4. Analysis ban đầu lỗi vì pandas/NumPy trả `DataFrame.values` read-only.
   `paper/analysis.py` được sửa tương thích bằng assignment qua `DataFrame.iat`.
   Công thức correlation không thay đổi. Sau sửa, analysis và Table 3 kết thúc
   với `All methods computed locally.`
5. Cảnh báo `FigureCanvasAgg is non-interactive` chỉ cho biết backend headless
   không mở cửa sổ Figure 1; correlation computation hoàn thành.

Không có failed/partial output nào được trộn vào final merge. Mỗi shard chỉ
được đưa vào merge sau khi validation đạt yêu cầu.

## 7. Đánh giá bằng chứng

### Paper nói gì

- Table 3 so sánh bốn phương pháp post-processed (`*`) với năm phương pháp raw
  (`†`) bằng năm intrinsic metrics.
- Adaptive dùng trung bình bằng trọng số của năm metrics; paper mô tả trọng số
  bằng nhau là heuristic.
- Paper báo Wilcoxon significance cho các so sánh liên quan.

### Code thực sự làm gì

- Fixed-method Table 3 lấy đúng bốn processed rows và năm raw rows từ local
  Parquet, sau đó in reference means để so sánh.
- CLI analysis tạo hàng `best` từ mọi method được truyền vào; đường chạy hiện
  tại truyền cả tám methods, không lọc về bốn starred candidates.
- Với metric null, code tính NaN-aware weighted mean và tái chuẩn hóa trên các
  metric còn lại.
- Không tìm thấy implementation Wilcoxon/bootstrap/Holm trong source/tests;
  Table 3 command không tái tính statistical significance.

### Suy luận của người phân tích

- Fixed-method intrinsic metrics được tái lập rất sát reference Table 3 và có
  provenance đủ tốt để dùng làm baseline cho bước nghiên cứu tiếp theo.
- Sai lệch sentence raw -1.30 đáng được phân tích theo tài liệu trước khi gán
  nguyên nhân cho model, dependency hoặc data drift.
- Hàng `best` và các claims downstream chưa đủ để chứng minh Adaptive tốt hơn
  best-fixed hoặc cải thiện RAG. Cần sửa candidate filter, đóng băng QA/evidence
  labels và đánh giá theo document-level splits.
- BI gần hoàn hảo của page chunking phù hợp với audit về co-derived structural
  bias; cần page-neutral ablation trước khi dùng BI để tối ưu selector mới.

## 8. Giới hạn và bước tiếp theo

Các giới hạn chính:

- bắt đầu từ parsed JSON; repository không có original PDF corpus, Azure raw
  responses hoặc parser-output lock để tái lập end-to-end parsing;
- dependency lock upstream chưa đầy đủ, dù model/code revisions quan trọng đã
  được ghim trong harness;
- chưa có confidence intervals hoặc document-level significance tests;
- chưa có frozen evidence labels cho Recall@k, MRR và nDCG;
- chưa tái lập protocol-faithful Adaptive/RAG vì candidate và baseline paths
  trong code audit còn mismatch.

Thứ tự tiếp theo được khuyến nghị:

1. đóng băng QA/evidence schema và document IDs;
2. sửa experiment harness để Adaptive chỉ xét bốn starred candidates và Table
   5 dùng raw baselines đúng protocol;
3. tạo best-fixed và oracle baselines không rò rỉ test information;
4. chạy document-grouped retrieval evaluation trước mọi paid LLM evaluation;
5. triển khai BI page-neutral ablation;
6. chỉ gửi khoảng 5--6 cấu hình mạnh nhất sang answer-quality evaluation có
   cache, checkpoint và budget estimate.
