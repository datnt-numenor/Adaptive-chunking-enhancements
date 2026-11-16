# Tái lập kết quả Table 3 của SDP-B và SDP-B tối ưu (`sw0_rc1`)

Viết ngày 12/11/2026. Mọi lệnh chạy từ thư mục gốc của repo `adaptive-chunking` (bản gốc `ekimetrics/adaptive-chunking`,
commit `ea87ce8`), với thư mục `research\` này chép vào. Kết quả cần khớp: xem cuối file.

## 1. Môi trường

- Python 3.11.9. Bản ghim đầy đủ: `requirements-freeze-2026-11-12.txt` (pip freeze của máy đã chạy).
  Các gói quyết định con số: `torch==2.6.0+cu124`, `transformers==4.57.6`, `sentence-transformers==5.1.2`,
  `numpy==2.4.6`, `pandas==3.0.6`, `scipy==1.17.1`, `tiktoken==0.14.0`.
- Cài gói gốc ở chế độ editable: `pip install -e .`
- pandas 3: máy đã chạy có một thay đổi cục bộ trong `src\adaptive_chunking\paper\analysis.py` của repo gốc
  (`git diff src/adaptive_chunking/paper/analysis.py`). Các script ở đây không gọi file đó.
- GPU: 8 GB đủ cho CLAIR (khoảng 1 giờ mỗi cấu hình trên RTX 4060). Bộ `clair_ext` cần ≥ 24 GB
  (`research\data_expansion\run_metrics_remote_v2.sh`).

### Ghim jina-v3 (bắt buộc)

Mã tuỳ biến của `jinaai/jina-embeddings-v3` tải tokenizer và `jinaai/xlm-roberta-flash-implementation` theo nhánh `main`,
bỏ qua `revision`. Phải tải đúng hai commit rồi trỏ `refs/main` của cache về chúng, sau đó chạy với `HF_HUB_OFFLINE=1`:

```python
from pathlib import Path
from huggingface_hub import snapshot_download
for repo, rev in [("jinaai/jina-embeddings-v3", "ab036b023d30b4d1138c4c3bfa9f0c445ab455d6"),
                  ("jinaai/xlm-roberta-flash-implementation", "bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3")]:
    snap = Path(snapshot_download(repo, revision=rev))
    (snap.parents[1] / "refs").mkdir(exist_ok=True)
    (snap.parents[1] / "refs" / "main").write_text(rev)
```

Kiểm tra: `python -X utf8 research\structure_chunker\score_full.py validate --docs 2 --methods our_recurs_1100 page`
phải lệch không quá 1e-3 so với `runs\full_kaggle\results\chunking_metrics.parquet`.

### Biến môi trường thay cho đường dẫn cứng

| Biến | Mặc định | Dùng ở |
|---|---|---|
| `AC_MODELS` | `D:/Project/models` | `run_sdp.py --embed-model qwen0.6b`, `retrieval\proxy_retrieval.py` (cần thư mục con `Qwen3-Embedding-0.6B@97b0c61`, `snowflake-arctic-embed-l-v2.0@ac6544c`) |
| `AC_HARNESS` | `D:/Project/Adaptive-chunking-enhancements` | `retrieval\proxy_retrieval.py`, `data_expansion\make_retrieval_bundle.py` (repo nhóm, commit `2110947`) |

Table 3 không cần hai biến này; chúng chỉ dùng cho retrieval.

## 2. Dữ liệu cần có

| Dữ liệu | Nguồn |
|---|---|
| `data\clair\adi_parsed`, `data\clair\mentions` | có sẵn trong git của repo gốc |
| `runs\full_kaggle\` (chunk và metric của 4 ứng viên Adaptive) | bản tái lập trên Kaggle; `llm_regex` do LLM sinh nên **không tái tạo chính xác được**, phải dùng đúng file này |
| `runs\table5_v2\prepared\folds.jsonl` | chia 5 fold theo tài liệu, seed 2026 |

## 3. Các bước (cmd, Windows)

```bat
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1

:: SDP-B: nhúng block (GPU, ~4,5 phút) + quy hoạch động + hậu xử lý của paper
python -X utf8 research\structure_chunker\run_sdp.py --out runs\sdp\B_default --method sdp_b --embed

:: SDP-B tối ưu: chạy lại quy hoạch động từ embedding đã cache (CPU, ~15 giây cho 12 cấu hình)
python -X utf8 research\structure_chunker\headroom\proxy_sweep.py --out runs\sdp_sweep_v2 --round 2

:: chấm đủ 5 metric bằng code gốc của paper (GPU)
python -X utf8 research\structure_chunker\score_full.py run runs\sdp\B_default
python -X utf8 research\structure_chunker\score_full.py run runs\sdp_sweep_v2\sw0_rc1

:: bảng, so sánh theo cặp, 5 fold, bảng so với ± của paper
python -X utf8 research\structure_chunker\compare.py runs\sdp\B_default runs\sdp_sweep_v2\sw0_rc1
python -X utf8 research\structure_chunker\tune.py --out runs\sdp\tune_check runs\sdp\B_default runs\sdp_sweep_v2\sw0_rc1
python -X utf8 research\structure_chunker\headroom\gain_vs_sd.py --out runs\headroom_gain_vs_sd_check
```

Mỗi script từ chối ghi vào thư mục đã tồn tại; đổi tên `--out` nếu đã có.

Lưu ý: `proxy_sweep.py` bước 1 (hiệu chỉnh proxy) đọc điểm thật của `runs\sdp\{A_default, A_target600, A_target800,
A_target1000, B_default, B_qwen}`. Nếu chỉ cần chunk của `sw0_rc1`, các lần chạy đó vẫn phải có sẵn; cách tạo chúng ở `README.md`.

## 4. Con số phải ra (CLAIR, 33 tài liệu)

| | RC | ICC | DCC | BI | SC | T |
|---|---:|---:|---:|---:|---:|---:|
| Adaptive (4 ứng viên) | 98,96 | 68,35 | 88,63 | 99,42 | 99,93 | 91,06 |
| SDP-B | 97,81 | 75,98 | 85,99 | 99,94 | 99,96 | 91,94 |
| SDP-B tối ưu (`sw0_rc1`) | 99,80 | 77,76 | 85,15 | 99,94 | 99,96 | 92,52 |

- `sw0_rc1` − Adaptive = +1,50, CI 95 % [1,23; 1,77], 31/33 tài liệu, Wilcoxon p = 3,3e-9.
- SDP-B − Adaptive = +0,90 [0,64; 1,15], 31/33.

### Sai khác được phép

| Thành phần | Mức giống nhau đã đo |
|---|---|
| SC, BI, RC | trùng (lệch 1e-14); không dùng GPU |
| ICC, DCC trên GPU khác | lệch tối đa 1e-3 trên thang 0–1 (laptop, Kaggle, H100) |
| Điểm cắt khi embedding block tính trên GPU khác | mô phỏng: 97,5–99 % điểm cắt giữ nguyên, điểm đổi ≤ 0,011 mỗi tài liệu. Muốn trùng tuyệt đối thì dùng lại file chunk: sha256 của `runs\sdp_sweep_v2\sw0_rc1\chunks\small_merged\chunks.parquet` bắt đầu bằng `b47b8975ece48d34` |

## 5. Không tái lập chính xác được

- **Table 5** (câu trả lời và chấm bằng LLM): chạy lại cùng prompt qua gateway chỉ cho 63–71/99 câu trả lời giống hệt.
- **`llm_regex`**: do LLM sinh (gpt-4o); Adaptive chọn nó cho 6/33 tài liệu.
- **Bộ test `clair_ext`**: PDF không nằm trong repo. Tải lại bằng `research\data_expansion\download.py` với các file
  `candidates_*.tsv`; RC ở bộ này phụ thuộc bước coreference chạy lại.
