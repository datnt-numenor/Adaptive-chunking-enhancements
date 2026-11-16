# Structure-aware DP Chunker (SDP)

Hướng cải tiến mục 5.3 trong `BAO_CAO.md`: thay vì chọn một phương pháp chunking cố định cho cả tài liệu
(Adaptive), chunker mới **chỉ cắt tại ranh giới block của parser** và **chọn điểm cắt bằng quy hoạch động**,
nên kích thước chunk thay đổi theo nội dung ngay trong một tài liệu.

Code nằm ở đây, không sửa `src/` upstream. Metric được chấm bằng **code gốc** (`compute_metrics_per_origin`).

## Thành phần

| File | Vai trò |
|---|---|
| `sdp_chunker.py` | Bản A: unit = block parser (block > 1.100 token được cắt trước bằng recursive splitter); DP tối đa `Σ(−τ + điểm ranh giới)`; ràng buộc cứng ≤ 1.100 token, phạt mềm < 100 token |
| `sdp_embed.py` | Bản B: nhúng mỗi block một lần; proxy ICC/DCC tính O(d) mỗi ứng viên nhờ prefix-sum; **Dinkelbach** để tối đa *trung bình* theo chunk (ICC/DCC là trung bình, không phải tổng) |
| `run_sdp.py` | Chạy chunker trên corpus, hậu xử lý giống `replicate.py` bước 1c (`merge_small_chunks_to_neighbours`, min 100, max 1.150), chấm SC/BI/RC trên CPU |
| `score_full.py` | Chấm đủ 5 metric bằng jina-v3 pin revision như lần Kaggle; `validate` để so lại với số Kaggle |
| `compare.py` | Bảng so với 4 ứng viên, Adaptive, "Adaptive + SDP"; bootstrap CI + Wilcoxon theo tài liệu |

Điểm ranh giới (bản A, cũng dùng trong B với trọng số `struct_weight`):

- Cắt trước tiêu đề Markdown: thưởng (cấp 1: 0,6 … cấp ≥4: 0,3).
- Cắt trước block mở đầu bằng chữ thường (đoạn văn bị ngắt qua trang, 708 block trong corpus): phạt 3.
- Cắt trước block chỉ chứa `<!-- PageBreak/PageHeader/PageFooter -->`: phạt 2 (giữ chúng ở chunk trước).
- Cắt trước đại từ / sau dấu `:` / giữa hai mục danh sách: phạt nhỏ.

## Lệnh

```bash
# bản A (CPU, ~2 giây cho 33 tài liệu)
python -X utf8 research/structure_chunker/run_sdp.py --out runs/sdp/A_default
# bản B (GPU, jina-v3; embedding block được cache ở runs/sdp/_unit_emb/)
HF_HUB_OFFLINE=1 python -X utf8 research/structure_chunker/run_sdp.py --out runs/sdp/B_default --method sdp_b --embed
# chấm đủ 5 metric, rồi so sánh
HF_HUB_OFFLINE=1 python -X utf8 research/structure_chunker/score_full.py run runs/sdp/B_default
python -X utf8 research/structure_chunker/compare.py runs/sdp/A_default runs/sdp/B_default
# test
python -m pytest -q research/structure_chunker
```

Trong cmd (Windows):

```bat
python -X utf8 research\structure_chunker\run_sdp.py --out runs\sdp\A_target800 --method sdp_a_t800 --set target_tokens=800
set HF_HUB_OFFLINE=1
python -X utf8 research\structure_chunker\score_full.py run runs\sdp\A_target800
python -X utf8 research\structure_chunker\tune.py --out runs\sdp\tune_A_targets ^
  runs\sdp\A_target600 runs\sdp\A_target800 runs\sdp\A_target1000
```

`tune.py` chọn cấu hình theo điểm Table 3 trên fold huấn luyện (`runs\table5_v2\prepared\folds.jsonl`,
5 fold, seed 2026), lấy điểm trên fold kiểm tra, rồi so 33 điểm held-out với Adaptive / our_recurs_1100 / page
(bootstrap CI theo tài liệu, Wilcoxon, Holm).

Ghi chú: CLAUDE.md §6 nhắc tới `research\struct_chunker\struct_chunker.py`, nhưng file đó không có trên đĩa
(kiểm tra 09/11/2026). Thư mục này là bản triển khai thay thế; `--set target_tokens=800` tương ứng `--target 800`.

## Retrieval (bước 3 trong CLAUDE.md) — metric chính: **recall@10** (người dùng chốt 09/11/2026)

Thư mục `retrieval\`:

| File | Vai trò |
|---|---|
| `prepare_sdp_retrieval.py` | Tạo thư mục đầu vào **mới**: chép nguyên `runs\table5_v2\{prepared,qa-full,qa-smoke}` (kiểm sha256), thêm file hệ thống SDP bằng chính hàm `_find_chunk_spans` của harness. `--cv-system` ghép hệ thống chọn theo CV từ `tune.py` |
| `sdp_launcher.py` | Gọi harness nhóm, thêm tên hệ thống SDP vào `ALL_SYSTEMS` trong bộ nhớ (không sửa file harness, không đụng `FIXED_SYSTEMS`) |
| `make_sdp_bundle.py`, `run_sdp_retrieval_remote.sh` | Đóng gói và chạy trên A5000/A6000 (`check → setup → smoke → full → pack`), cấu hình giống `runs\gpu\run_table5_v2_remote.sh` |
| `analyze_sdp_retrieval.py` | Thống kê theo tài liệu: recall@10 chính; kèm `top10_tokens` và `recall@budget` (cắt danh sách xếp hạng ở cùng ngân sách token) vì recall theo độ phủ ký tự thiên vị chunk to |

Đã kiểm tra trên máy local: qua launcher, `evaluate-retrieval` trên kết quả A5000 cũ cho sai lệch **0,0** so với
`runs\table5_v2_results`; recall@10 tính lại trong `analyze_sdp_retrieval.py` khớp harness (sai lệch 0).

**Kết quả (RunPod A5000, 09/11/2026)** — `runs\table5_sdp_results\table5_sdp`, thống kê ở `runs\table5_sdp_analysis`.
3 hệ thống đối chứng khớp tuyệt đối (sai lệch 0 trên 297 cặp) với `runs\table5_v2_results`.
sdp_a = A_target800 (CV chọn), sdp_b = B_default, sdp_bq = B_qwen.

| Hệ thống | recall@10 | recall@budget (7.929 tok) | nDCG@10 | top10_tokens |
|---|---:|---:|---:|---:|
| adaptive | 0,900 | 0,879 | 0,704 | 7.844 |
| LC default | 0,854 | 0,854 | 0,689 | 7.792 |
| page | 0,845 | 0,835 | 0,662 | 6.932 |
| sdp_a | 0,854 | 0,847 | 0,661 | 8.857 |
| sdp_b | 0,840 | 0,825 | 0,677 | 6.801 |
| sdp_bq | 0,862 | 0,862 | 0,683 | 6.849 |

Mọi chênh lệch SDP − Adaptive đều âm, CI chứa 0, Holm p = 1,0. Ở các câu SDP thua, evidence **không** bị cắt đôi
(nằm trọn trong 1 chunk ở cả hai hệ thống) → thua do xếp hạng, không do ranh giới. Điểm Table 3 tăng **không**
chuyển thành retrieval tốt hơn.

## Kiểm chứng môi trường local

`score_full.py validate --docs 2 --methods our_recurs_1100 page` trên RTX 4060 (torch 2.6.0, bf16):
lệch tối đa so với Kaggle **7,5e-4** (0,075 điểm/100). Cần trỏ `refs/main` trong cache HF về đúng commit đã pin
(xem docstring `score_full.py`), vì code tuỳ biến của jina tải tokenizer và module con theo nhánh `main`.

## Lưu ý về tính hợp lệ (phải ghi khi báo cáo)

1. **Bản B tối ưu thẳng vào embedder của metric (jina-v3)** → điểm ICC/DCC có thể bị thổi phồng.
   Cần biến thể `--embed-model qwen0.6b` (embedder độc lập) và kiểm tra retrieval.
2. Tham số mặc định được đặt **trước** khi xem điểm 33 tài liệu; smoke 8 tài liệu nhỏ nhất chỉ dùng để
   kiểm tra code chạy đúng. Mọi tinh chỉnh tiếp theo phải qua CV theo tài liệu.
3. So SDP **đứng riêng** với Adaptive. Hàng "Adaptive + SDP" (SDP là ứng viên thứ 5) luôn ≥ cả hai theo định nghĩa,
   chỉ báo cáo riêng.
4. BI của SDP ≈ 100 theo thiết kế (chỉ cắt tại `split_points`) — đây là lợi thế "theo định nghĩa", không phải bằng chứng
   chunk tốt hơn (xem F6).
