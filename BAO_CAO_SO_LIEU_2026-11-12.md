# Báo cáo số liệu bổ sung: SDP-B tối ưu (`sw0_rc1`) và phân tích dư địa Table 3

Ngày 12/11/2026. Bổ sung cho `BAO_CAO_SO_LIEU_2026-11-10.md` (file đó giữ nguyên). Mọi số liệu có nhãn **[Paper]**,
**[Code/Artifact]** (kèm đường dẫn) hoặc **[Suy luận]**. Thống kê: đơn vị là tài liệu, Wilcoxon (Pratt), CI bootstrap
10.000 lần theo tài liệu, seed 2026.

## 0. Tóm tắt

1. **Số thật, CLAIR 33 tài liệu:** bản SDP-B tối ưu (`sw0_rc1`) hơn Adaptive **+1,50** điểm trung bình 5 metric,
   CI 95 % [1,23; 1,77], cao hơn ở 31/33 tài liệu, p = 3,3e-9. So với SDP-B: +0,60 [0,42; 0,78], 32/33.
2. **Mức tăng nằm ở ICC (+9,42) và RC (+0,84).** DCC giảm 3,49. BI và SC đã chạm trần.
3. **So với khoảng ± của paper** (độ lệch chuẩn giữa các tài liệu): ICC và cột trung bình vượt ±; RC, BI, SC không thể
   vượt vì Adaptive cách 100 ít hơn chính ± của nó.
4. **Trạng thái:** hai thay đổi của cấu hình được tìm bằng thăm dò trên toàn bộ 33 tài liệu. Chọn bằng 5 fold giữa 6 cấu hình
   có điểm thật cho cùng kết quả (+1,50, mục 2b); với 8 cấu hình cũng vậy (mục 11b).
5. **Xác nhận trên bộ test (mục 11a, đăng ký trước, lần nhìn thứ hai):** `sw0_rc1` − Adaptive = **+1,97** [1,85; 2,08],
   cao hơn ở 128/129 tài liệu, p = 6,7e-23.
6. **Retrieval (mục 11c, bản chạy thử cục bộ trên CLAIR):** `sw0_rc1` có recall@10 **thấp hơn** Adaptive
   (0,812 so với 0,907; Δ −0,095 [−0,182; −0,013], Holm p = 0,58). Điểm Table 3 cao nhất không đi cùng retrieval tốt nhất.
   Chưa chạy Table 5 cho cấu hình mới; mục 6 cho thấy chưa có bằng chứng Table 5 tăng theo Table 3.

## 1. Cấu hình `sw0_rc1`

Hai thay đổi so với SDP-B (`runs\sdp\B_default`), trọng số ICC/DCC giữ nguyên 1:1:

| Thay đổi | Tham số | Ý nghĩa |
|---|---|---|
| `rc1` | phạt λ = 1 cho điểm cắt tách cặp thực thể–đại từ | lấy cặp từ `data\clair\mentions` (cùng dữ liệu metric RC dùng) |
| `sw0` | `struct_weight = 0` | bỏ các ưu tiên cấu trúc của bản A (thưởng cắt trước tiêu đề, phạt chunk mở đầu bằng chú thích trang, phạt cắt giữa đoạn) |

- [Code/Artifact] Mã: `research\structure_chunker\headroom\proxy_sweep.py` (lớp `RCScorer`, danh sách `SWEEP2`).
  Chunk: `runs\sdp_sweep_v2\sw0_rc1\chunks\small_merged\chunks.parquet` (2.042 chunk, hậu xử lý giống paper).
- [Code/Artifact] Chấm bằng `score_full.py run runs\sdp_sweep_v2\sw0_rc1`: code metric gốc, jina-v3 revision `ab036b0`,
  bf16, RTX 4060, 3.545 giây (`metrics\environment.json`).

## 2. Kết quả Table 3 trên CLAIR (33 tài liệu)

[Code/Artifact] `runs\sw0_rc1_clair_report_v1\compare_output.txt`
(= `compare.py runs\sdp\B_default runs\sdp_sweep_v2\sw0_rc1`). T là trung bình của năm trung bình metric.

| Phương pháp | RC | ICC | DCC | BI | SC | **T** |
|---|---:|---:|---:|---:|---:|---:|
| Adaptive, paper **[Paper]** | 99,0 | 68,2 | 88,8 | 99,4 | 99,9 | 91,07 |
| Adaptive, chạy lại | 98,96 | 68,35 | 88,63 | 99,42 | 99,93 | 91,06 |
| SDP-B | 97,81 | 75,98 | 85,99 | 99,94 | 99,96 | 91,94 |
| **SDP-B tối ưu (`sw0_rc1`)** | **99,80** | **77,76** | 85,15 | 99,94 | 99,96 | **92,52** |

So sánh theo cặp từng tài liệu ([Code/Artifact] `runs\sw0_rc1_clair_report_v1\paired_by_metric.txt`):

| So sánh | Trung bình 5 metric | ICC | DCC | RC |
|---|---|---|---|---|
| `sw0_rc1` − Adaptive | **+1,50** [1,23; 1,77], 31/33, p = 3,3e-9 | +9,42 [7,76; 10,83], 31/33 | −3,49 [−4,75; −2,19], 4/32 | +0,84 [0,39; 1,38], 13/31 |
| `sw0_rc1` − SDP-B | +0,60 [0,42; 0,78], 32/33, p = 2,9e-7 | +1,79 [0,88; 2,57], 28/33 | −0,54 [−1,18; 0,13], 9/32 | +1,99 [1,07; 3,06], 18/31 |

- +1,50 là trung bình của hiệu theo từng tài liệu (92,45 so với 90,95). Giá trị tăng thêm của T là 1,47
  (92,524 − 91,057). Hai số lệch nhau vì 2 tài liệu không tính được RC và 1 tài liệu không tính được DCC.
- Tài liệu thiếu RC (cả ba phương pháp): "Recommandations de sécurité pour un système d'IA générative",
  "Entrée en vigueur du règlement européen sur l'IA - CNIL". Tài liệu thiếu DCC (chỉ ở `sw0_rc1`): "EDPS Generative AI
  guidelines" (523 token, gộp thành một chunk).
- Hai tài liệu `sw0_rc1` thấp hơn Adaptive: "EDPS Generative AI guidelines" (−0,82), "Standards - Transparency" (−0,40).
- [Code/Artifact] Kiểm tra lại độc lập (script không dùng chung hàm, đọc thẳng file metric): mọi số trong bảng trùng;
  33 tài liệu trùng đúng bộ baseline; 2.042 chunk khớp file chunk. `runs\sw0_rc1_clair_report_v1\verify_independent.txt`.
  Cận trên CI ra 1,76 hoặc 1,77 tùy cách rút mẫu bootstrap.

### 2b. Chọn cấu hình bằng 5 fold (bổ sung cuối ngày 12/11)

[Code/Artifact] `runs\sdp\tune_with_sw0_rc1_v1\` (`tune.py`, fold theo tài liệu từ `runs\table5_v2\prepared\folds.jsonl`,
tiêu chí: điểm Table 3 trung bình trên fold huấn luyện). Ứng viên là 6 cấu hình đã có điểm thật: SDP-A t600, t800, t1000,
SDP-B, `B_qwen`, `sw0_rc1`.

- `sw0_rc1` được chọn ở **5/5 fold**. Điểm trên fold kiểm tra: 92,52; 92,74; 92,30; 92,50; 92,14.
- Điểm held-out trên 33 tài liệu − Adaptive = **+1,50** [1,22; 1,76], 31/33. Trùng với mục 2, vì mọi fold chọn cùng một cấu hình.
- **Giới hạn của phép kiểm này:** hai thay đổi `rc1` và `sw0` được tìm ra bằng thăm dò trên toàn bộ 33 tài liệu, nên việc
  chia fold chỉ loại trừ sai lệch do *chọn giữa các ứng viên*, không loại trừ sai lệch do *thiết kế ứng viên*.
  `keepcont_rc1` và `noart_rc1` chưa có điểm thật nên chưa nằm trong tập ứng viên.

## 3. So với khoảng ± của paper

[Paper] Table 3 ghi "% mean ± % st. dev.": ± là độ lệch chuẩn giữa 33 tài liệu. Cột mean của paper không có ±;
giá trị ±1,1 dưới đây tính từ bản chạy lại. [Code/Artifact] `runs\headroom_gain_vs_sd_v2\report.txt`, `gain_vs_sd.csv`.

| Metric | Adaptive (mean ± sd) | Chỗ còn lại tới 100 | SDP-B − Adaptive | `sw0_rc1` − Adaptive | Vượt ±? |
|---|---:|---:|---:|---:|---|
| RC | 98,96 ± 1,5 | 1,04 | −1,15 | +0,84 | Không thể (chỗ còn lại < ±) |
| ICC | 68,35 ± 4,7 | 31,65 | +7,63 | **+9,42** | **Có** |
| DCC | 88,63 ± 3,6 | 11,37 | −2,64 | −3,49 | Không, giảm |
| BI | 99,42 ± 1,2 | 0,58 | +0,52 | +0,52 | Không thể |
| SC | 99,93 ± 0,3 | 0,07 | +0,03 | +0,03 | Không thể |
| Trung bình theo tài liệu | 90,95 ± 1,1 | 9,05 | +0,90 | **+1,50** | **Có** (SDP-B thì chưa) |

- [Paper] Theo cùng tiêu chí, Adaptive của paper hơn phương pháp đơn tốt nhất +0,39 (91,07 so với 90,68) và paper kết luận
  bằng kiểm định Wilcoxon theo cặp, không bằng ±.
- [Suy luận] ± đo mức khác nhau giữa các tài liệu, không phải độ bất định của mức tăng. Độ bất định của mức tăng là CI
  theo cặp ở mục 2.

## 4. Dư địa Table 3 và các phép thăm dò (proxy)

Các số ở mục này là **thăm dò**: SC/BI/RC tính bằng code metric thật trên CPU; ICC/DCC dự đoán từ embedding block đã cache
(`runs\sdp\_unit_emb\jina_max1100`). Chạy trên toàn bộ tài liệu, không chia fold.

- [Code/Artifact] `runs\headroom_analysis_v1\report.txt`. Giá trị cao nhất từng metric trong 11 cấu hình đã có điểm thật:
  SC 100,00; BI 99,94; RC 99,29; ICC 76,81; DCC 91,06. Nếu đạt cả năm cùng lúc: +1,49 so với SDP-B, không cấu hình nào đạt.
- [Code/Artifact] ICC và DCC đánh đổi qua kích thước chunk: trong cùng tài liệu, ρ(kích thước, ICC) = −0,79,
  ρ(kích thước, DCC) = +0,89, ρ(ICC, DCC) = −0,74 (âm ở 31/32 tài liệu). Trên `clair_ext`: −0,59; +0,94; −0,54.
- [Code/Artifact] `runs\sdp_sweep_v1\report.txt`. Đổi trọng số ICC/DCC không tăng điểm: DCC ×1,5 cho +0,10 [−0,06; +0,29];
  ICC ×1,5 cho −0,04 [−0,15; +0,05] so với SDP-B.
- [Code/Artifact] Chọn theo tài liệu trong các cấu hình đã có điểm thật: +1,03 so với Adaptive (chỉ +0,13 trên SDP-B).

Dự đoán proxy so với SDP-B (CLAIR: `runs\sdp_sweep_v2\report.txt`, `runs\sdp_sweep_v4\report.txt`;
bộ test: `runs\sdp_sweep_ext_v1\report.txt`, 4 cấu hình chốt từ CLAIR):

| Cấu hình | CLAIR (dự đoán) | `clair_ext` 129 tài liệu (dự đoán) |
|---|---|---|
| `rc1` | +0,23 [0,06; 0,42] | +0,56 [0,44; 0,70] |
| `sw0` | +0,49 [0,34; 0,65] | +0,43 [0,33; 0,54] |
| `sw0_rc1` | +0,74 [0,56; 0,91] → **thật +0,60** | +1,02 [0,89; 1,15] |
| `noart_rc1` (chỉ bỏ phạt chú thích trang) | +0,56 [0,38; 0,76] | chưa chạy |
| `keepcont_rc1` (chỉ giữ phạt cắt giữa đoạn) | +0,68 [0,51; 0,85] | chưa chạy |

- RC là số thật ở cả hai bộ: `rc1` đưa RC từ 97,81 lên 99,81 (CLAIR) và từ 94,63 lên 98,45 (`clair_ext`).
- [Code/Artifact] Vòng 4 cho thấy phần lớn mức tăng ICC của `sw0` đến từ việc bỏ **mức phạt chunk mở đầu bằng chú thích
  trang** (`artifact_penalty`); bỏ phạt cắt giữa đoạn chỉ thêm khoảng 0,04.
- **Proxy so với thật** (`sw0_rc1`, CLAIR): ICC dự đoán 77,77, thật 77,76; DCC dự đoán 85,82, thật 85,15; mức tăng so với
  Adaptive dự đoán +1,64, thật +1,50. Sai số theo tài liệu (thật − dự đoán): ICC 0,00 (sd 0,58), DCC −0,67 (sd 0,98).

## 5. Chất lượng điểm cắt

[Code/Artifact] `runs\headroom_analysis_v1\cut_quality.txt`, `runs\sdp_sweep_v4\cut_quality.txt` (CLAIR, tỉ lệ trên số điểm cắt):

| Cấu hình | Cắt ngay trước tiêu đề | Cắt giữa đoạn văn | Chunk mở đầu bằng chú thích trang | Điểm cắt ngay sau một tiêu đề | Cặp tham chiếu bị tách |
|---|---:|---:|---:|---:|---:|
| SDP-A t600 | 67 % | 3 (0 %) | 19 (1 %) | 21 % | 133/7.796 |
| SDP-B | 49 % | 3 (0 %) | 28 (1 %) | 25 % | 185/7.796 |
| `rc1` | 48 % | 3 (0 %) | 34 (2 %) | 26 % | 43/7.796 |
| **`sw0_rc1`** | 23 % | 133 (7 %) | 566 (28 %) | 34 % | 43/7.796 |
| `keepcont_rc1` | 24 % | 3 (0 %) | 605 (30 %) | 33 % | 42/7.796 |
| `noart_rc1` | 38 % | 3 (0 %) | 509 (26 %) | 29 % | 41/7.796 |

[Suy luận] `sw0_rc1` đổi một phần cấu trúc lấy điểm ICC. `keepcont_rc1` và `noart_rc1` giữ 0 % điểm cắt giữa đoạn và được
dự đoán gần bằng `sw0_rc1`, nhưng **chưa có điểm thật**.

## 6. Table 5 và retrieval: điểm Table 3 có dự báo được không

[Code/Artifact] `runs\headroom_analysis_v1\report.txt`, `retrieval_miss_overlap.txt`. Dùng kết quả retrieval và Table 5 đã có
của Adaptive, SDP-A, SDP-B (báo cáo 10/11); **chưa chạy retrieval hay Table 5 cho `sw0_rc1`**.

- Trên `clair_ext`, Adaptive đạt 76,4 điểm Correctness (độ phủ) và 87,8 nếu chỉ tính câu có trả lời. Trong 57 câu từ chối,
  47 câu có recall@10 = 0.
- Khi recall@10 = 1, Correctness của câu có trả lời là 88–91 ở mọi chunker, cả hai bộ.
- Trên `clair_ext`, 345/438 câu có recall@10 giống nhau ở cả 5 chunker; 29 câu trượt ở mọi chunker.
- Tương quan theo tài liệu giữa ΔICC và Δrecall@10 (so với Adaptive): ρ từ −0,13 đến +0,09, không có ý nghĩa ở cả hai bộ.
  ΔDCC: ρ = +0,23 đến +0,43 trên CLAIR, −0,07 đến −0,02 trên `clair_ext`.
- [Suy luận] Chưa có bằng chứng mức tăng Table 3 chuyển thành retrieval hay câu trả lời tốt hơn.

## 7. Khả năng tái lập

| Thành phần | Mức giống nhau | Bằng chứng |
|---|---|---|
| SC, BI, RC | trùng (lệch 1e-14) | [Code/Artifact] RC tính trên CPU so với bản chấm GPU |
| Chunker, cùng embedding | chunk giống hệt | [Code/Artifact] cấu hình `base` chạy 12/11 trùng `runs\sdp\B_default` (09/11) |
| ICC, DCC giữa các máy | lệch tối đa 0,1 điểm | [Code/Artifact] báo cáo 10/11 mục 5 |
| Chunker, embedding lệch nhẹ (mô phỏng trên CPU) | 97,5–99 % điểm cắt giữ nguyên; điểm đổi ≤ 0,011 mỗi tài liệu | [Code/Artifact] `runs\headroom_analysis_v1\noise_sensitivity.txt` |

Chưa làm: đẩy mã `headroom\` lên repo; file ghim phiên bản (máy này: torch 2.6.0+cu124, transformers 4.57.6,
sentence-transformers 5.1.2); bỏ đường dẫn cứng `D:/Project/...` trong 4 script; chia sẻ bộ test.

## 8. Hạn chế

1. Hai thay đổi của `sw0_rc1` được thiết kế bằng thăm dò trên **toàn bộ 33 tài liệu**; 5 fold (mục 2b) không loại trừ được
   sai lệch này. Chỉ một lần chạy trên `clair_ext` mới là xác nhận độc lập.
2. Chưa có điểm ICC/DCC thật của `sw0_rc1` trên `clair_ext`.
3. **Tính vòng tròn:** chunker dùng embedding jina-v3 (metric ICC/DCC) và dữ liệu coreference (metric RC) để chọn điểm cắt.
   Với ICC đã có đối chứng `B_qwen` (+0,78 với embedding độc lập); với RC chưa có.
4. DCC giảm 3,49, gần bằng ±3,6 của Adaptive.
5. `sw0` làm cấu trúc điểm cắt kém đi (mục 5).
6. Chưa có retrieval / Table 5 cho cấu hình mới.

## 9. Việc tiếp theo

1. Chấm thật `keepcont_rc1` và `noart_rc1` (`runs\sdp_sweep_v4`, khoảng 1 giờ mỗi bản trên RTX 4060).
2. Chọn cấu hình bằng `tune.py` 5 fold với điểm thật.
3. Một lần xác nhận có đăng ký trước trên `clair_ext` (cần pod để tính ICC/DCC; ghi rõ đây là lần thứ hai dùng bộ test).
   Dự đoán hiện có: `runs\headroom_predict_ext_v1\report.txt`.
4. Retrieval cho cấu hình cuối (bản chạy thử cục bộ; đầu vào đã chuẩn bị ở `runs\table5_proxy_prep_v3`).
5. Đóng gói tái lập (mục 7).

## 11. Bổ sung chiều 12/11: xác nhận trên bộ test, 5 fold đủ ứng viên, retrieval

Các mục 8 và 9 ở trên viết trước khi có phần này; những điểm đã được giải quyết ghi ở 11d.

### 11a. Bộ test `clair_ext` (129 tài liệu), lần nhìn thứ hai, có đăng ký trước

- Đăng ký trước: `research\data_expansion\PREREGISTRATION_2026-11-12_table3_second_look.md` (viết trước khi có ICC/DCC thật).
  Giả thuyết chính duy nhất H4: `sw0_rc1` > Adaptive. Dự đoán ghi trước: +1,84 (khoảng +1,70 đến +1,97).
- [Code/Artifact] Chấm trên pod 2× H100 bằng `run_metrics_remote_v2.sh`; kết quả ở
  `runs\gpu_metrics_ext_v2_results\extracted_verified\metrics_out` (sha256 file nén khớp với pod: `1895d929…`).
  Bước `validate` lệch tối đa 8,36e-4 so với Kaggle. Chunk được chấm trùng sha256 đã khoá. 129/129 tài liệu ở cả 4 cấu hình.
  Tổng thời gian GPU: 32,7 phút.
- [Code/Artifact] Thống kê: `runs\ext_second_look_analysis_v1\report.txt` (`analyze_second_look.py`).
  Adaptive và SDP-B dùng lại điểm của lần nhìn thứ nhất (`runs\ext_metrics_merged_v1`).

| Phương pháp | RC | ICC | DCC | BI | SC | **T** | − Adaptive [CI 95 %] | Thắng/hòa/thua | Holm p |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| Adaptive | 96,23 | 69,27 | 89,30 | 98,42 | 99,96 | 90,63 | — | — | — |
| SDP-B | 94,63 | 76,96 | 86,80 | 99,63 | 99,92 | 91,59 | +0,96 [0,81; 1,10] | 114/0/15 | (lần 1) |
| **`sw0_rc1` (H4, chính)** | 98,47 | 78,73 | 86,27 | 99,63 | 99,92 | **92,60** | **+1,97** [1,85; 2,08] | 128/0/1 | 6,7e-23 |
| `keepcont_rc1` (phụ) | 98,47 | 78,62 | 86,30 | 99,63 | 99,92 | 92,59 | +1,95 [1,84; 2,07] | 128/0/1 | 4,7e-22 |
| `noart_rc1` (phụ) | 98,47 | 78,05 | 86,41 | 99,63 | 99,92 | 92,50 | +1,86 [1,74; 1,98] | 128/0/1 | 4,7e-22 |
| `rc1` (phụ) | 98,45 | 76,51 | 86,38 | 99,63 | 99,92 | 92,18 | +1,54 [1,43; 1,66] | 128/0/1 | 4,7e-22 |

- **H4 được xác nhận.** H5 (`sw0_rc1` − SDP-B): +1,01 [0,87; 1,16], 122/1/6, Holm p = 6,8e-22.
- Mức tăng của `sw0_rc1` theo metric: ICC +9,46 [8,98; 9,93] (129/129 tài liệu); RC +2,24 [1,79; 2,74]; BI +1,21;
  SC −0,04; **DCC −3,03** [−3,48; −2,58].
- So với ± của Adaptive trên bộ test (RC 5,0; ICC 4,3; DCC 2,6; BI 1,6; SC 0,4; trung bình 1,3): ICC và trung bình vượt ±.
  **DCC giảm nhiều hơn ± của nó** (3,03 so với 2,6).
- Dự đoán so với thật: dự đoán +1,84, thật +1,97. ICC dự đoán 78,63, thật 78,73; DCC dự đoán 85,73 (đã hiệu chỉnh theo CLAIR), thật 86,27.
  Hiệu chỉnh DCC lấy từ CLAIR đã trừ quá tay trên bộ test.
- Đây là lần thứ hai dùng 129 tài liệu này. SC, BI, RC của `rc1` và `sw0_rc1` trên bộ test đã biết trước khi chạy pod.

### 11b. CLAIR: điểm thật của `keepcont_rc1`, `noart_rc1`; 5 fold đủ ứng viên; đối chiếu hai máy

[Code/Artifact] Điểm chấm trên pod, gộp ở `runs\sdp_pod_clair_v1\`; thống kê trong `runs\ext_second_look_analysis_v1\report.txt`.

| Cấu hình | RC | ICC | DCC | BI | SC | T | − Adaptive [CI 95 %] | Thắng |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `sw0_rc1` (pod) | 99,80 | 77,75 | 85,14 | 99,94 | 99,96 | 92,52 | +1,50 [1,22; 1,77] | 31/33 |
| `keepcont_rc1` (pod) | 99,80 | 77,34 | 85,35 | 99,94 | 99,96 | 92,48 | +1,46 [1,19; 1,72] | 31/33 |
| `noart_rc1` (pod) | 99,82 | 76,64 | 85,87 | 99,94 | 99,96 | 92,45 | +1,42 [1,14; 1,69] | 31/33 |

- Proxy đã dự đoán +1,58 cho `keepcont_rc1` và +1,46 cho `noart_rc1`.
- [Code/Artifact] 5 fold với 8 ứng viên có điểm thật (SDP-A t600/t800/t1000, SDP-B, `B_qwen`, `sw0_rc1`, `keepcont_rc1`,
  `noart_rc1`): `sw0_rc1` vẫn được chọn ở 5/5 fold, held-out +1,50 [1,22; 1,76]. `runs\sdp\tune_all_v2_pod\`.
  Hai ứng viên mới dùng điểm chấm trên pod, các ứng viên khác chấm trên laptop.
- **Đối chiếu hai máy (cùng file chunk `sw0_rc1`, RTX 4060 so với H100):** SC, BI, RC trùng; ICC lệch tối đa 2,7e-3,
  DCC tối đa 3,0e-3 (thang 0–1, cùng ở tài liệu "Standards - Transparency"), trung bình 3e-4. Trung bình 5 metric 92,454 so với 92,450.
  Mức lệch tối đa này lớn hơn con số 1,1e-3 ghi ở báo cáo 10/11.

### 11c. Retrieval: bản chạy thử cục bộ trên CLAIR (99 câu, Qwen3-0.6B thay cho 4B)

[Code/Artifact] `runs\proxy\r4_sdpb_variants\`, thống kê `runs\proxy\r4_sdpb_variants_analysis\`. Đây là bản chạy thử trên
laptop (đã kiểm với A5000 ở báo cáo 10/11), trên tập phát triển; **thăm dò, chưa xác nhận bằng harness chính thức**.

| Hệ thống | recall@10 | recall@5 | nDCG@10 | − Adaptive (recall@10) [CI 95 %] | p | Holm p |
|---|---:|---:|---:|---|---:|---:|
| Adaptive | 0,907 | 0,792 | 0,701 | — | — | — |
| LangChain mặc định | 0,844 | 0,757 | 0,681 | | | |
| SDP-B | 0,837 | 0,750 | 0,675 | | | |
| SDP-B + `rc1` | 0,869 | 0,806 | 0,693 | −0,038 [−0,089; 0,010] | 0,33 | 1,0 |
| `noart_rc1` | 0,874 | 0,794 | 0,688 | −0,033 [−0,092; 0,023] | 0,55 | 1,0 |
| `keepcont_rc1` | 0,814 | 0,736 | 0,663 | −0,093 [−0,172; −0,018] | 0,049 | 0,58 |
| **`sw0_rc1`** | 0,812 | 0,722 | 0,649 | −0,095 [−0,182; −0,013] | 0,049 | 0,58 |

- Không chênh lệch nào có ý nghĩa sau Holm. Nhưng hai cấu hình có điểm Table 3 cao nhất (`sw0_rc1`, `keepcont_rc1`) có
  ước lượng recall@10 thấp nhất, CI chưa hiệu chỉnh không chứa 0.
- `rc1` và `noart_rc1` (giữ thưởng cắt trước tiêu đề) cao hơn SDP-B +0,03 đến +0,04 (không có ý nghĩa).
- [Suy luận] Bỏ ưu tiên cắt theo tiêu đề làm tăng ICC nhưng có thể làm retrieval kém đi. Nếu cần **một** chunker vừa tăng
  Table 3 vừa không kéo retrieval xuống, `noart_rc1` là ứng viên hợp lý hơn `sw0_rc1`
  (Table 3: +1,42 trên CLAIR, +1,86 trên bộ test). Lựa chọn này đưa ra **sau khi** thấy kết quả bộ test và retrieval trên
  CLAIR, nên là thăm dò; kết quả chính đã đăng ký trước vẫn là `sw0_rc1`.

### 11d. Cập nhật cho mục 8 và 9

- Hạn chế 2 (chưa có điểm thật trên bộ test): đã có (11a). Hạn chế 6 (chưa có retrieval): đã có bản chạy thử (11c);
  chưa có harness chính thức với Qwen3-4B và chưa có Table 5.
- Việc tiếp theo 1–3: đã xong. Việc 5 (đóng gói tái lập): đã có `research\structure_chunker\REPRODUCE.md`,
  `requirements-freeze-2026-11-12.txt`, biến `AC_MODELS` / `AC_HARNESS`; chưa đẩy lên repo.
- Còn lại: xác nhận retrieval của `sw0_rc1` và `noart_rc1` bằng harness chính thức (Qwen3-4B, cần GPU 24 GB), cả trên
  bộ test; Table 5 nếu cần (tốn tiền API); đối chứng độc lập cho RC (coreference khác với metric).

## 10. Danh sách file

| Nội dung | Vị trí |
|---|---|
| Điểm thật `sw0_rc1` | `runs\sdp_sweep_v2\sw0_rc1\metrics\` |
| Bảng compare, so sánh theo cặp, kiểm tra độc lập | `runs\sw0_rc1_clair_report_v1\` |
| Bảng so với ± | `runs\headroom_gain_vs_sd_v2\` (`_v1` là bản chạy lỗi, không dùng) |
| Phân tích dư địa, điểm cắt, độ nhạy, retrieval trượt | `runs\headroom_analysis_v1\` |
| Các vòng quét proxy | `runs\sdp_sweep_v1`, `_v2`, `_v4`, `runs\sdp_sweep_ext_v1` |
| Script | `research\structure_chunker\headroom\` |
| Đăng ký trước lần nhìn thứ hai, gói pod | `research\data_expansion\PREREGISTRATION_2026-11-12_table3_second_look.md`, `runs\gpu_metrics_ext_v2\` |
| Kết quả pod (đã kiểm sha256) | `runs\gpu_metrics_ext_v2_results\extracted_verified\` |
| Thống kê bộ test + CLAIR trên pod | `runs\ext_second_look_analysis_v1\` |
| Điểm CLAIR chấm trên pod, 5 fold đủ ứng viên | `runs\sdp_pod_clair_v1\`, `runs\sdp\tune_all_v2_pod\` |
| Retrieval chạy thử cục bộ | `runs\proxy\r4_sdpb_variants\`, `runs\proxy\r4_sdpb_variants_analysis\` |
| Hướng dẫn tái lập | `research\structure_chunker\REPRODUCE.md` |
