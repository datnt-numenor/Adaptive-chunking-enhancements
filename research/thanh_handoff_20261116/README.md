# Bàn giao phần việc của Thanh — 16/11/2026

Tác giả commit: Thanh <thanhnguyencong2005@gmail.com>.

## Nội dung

- Mã chunker cấu trúc/SDP và phân tích headroom: `research/structure_chunker/`.
- Mở rộng dữ liệu và đăng ký thí nghiệm: `research/data_expansion/`, `data/clair_ext/`, `data/clair_ext_set/`.
- Công cụ Table 5 nhiều hệ thống: `research/table5_ext/` và `runs/gw/`.
- Báo cáo: `BAO_CAO_SO_LIEU_2026-11-10.md`, `BAO_CAO_SO_LIEU_2026-11-12.md`, các báo cáo audit và PDF tại gốc repo.
- QA, folds, chunks, metric, retrieval, đầu vào/đầu ra sinh câu trả lời, chấm điểm và phân tích: `runs/`.
- Bản vá harness `prepare` và nhận diện câu từ chối: `research/week3/`, kèm `LOCAL_PATCH_*.json`.
- Thay đổi pandas 3 của repo gốc được giữ dưới dạng `pandas3_analysis.patch`; không áp dụng vào `src/` của repo nhóm.

## Nguồn và phạm vi

[Code/Artifact] Nguồn chính: bản làm việc `D:/Project/adaptive-chunking`, HEAD `ea87ce8e1a97888f3f179e7f1359ff7f43fb179d` cùng các tệp chưa commit. Bản bàn giao nối tiếp nhánh `thanh`, HEAD trước khi nhập `cee3c42`.

[Code/Artifact] `inventory.json` ghi đường dẫn nguồn, đích, kích thước và SHA-256 của 11.872 tệp được xét; 11.558 tệp được đưa vào hoặc đã có sẵn. 314 tệp nén/embedding/index nhị phân có thể tạo lại được giữ ở máy nguồn, kèm lý do loại trừ trong manifest. Đây không phải bản sao toàn bộ ổ đĩa. Môi trường Python, tệp `.env`, khóa và cache Python không thuộc bản bàn giao.

[Code/Artifact] Nếu kết quả trên nhánh đã có nội dung khác bản cục bộ, bản cục bộ được đặt ở `local_snapshot/` thay vì ghi đè. Không sửa các kết quả nguồn đã đóng băng. Không đưa bản AGENTS.md cá nhân thay thế hướng dẫn của nhóm.

## Kiểm tra và tái lập

[Code/Artifact] Đã chạy kiểm thử cục bộ `test_sdp_chunker.py` và `test_sdp_embed.py`: 11 passed. Không chạy lại thí nghiệm GPU hoặc API trả phí trong lần bàn giao này. Kết quả lưu trữ phải được diễn giải theo báo cáo và metadata của từng lần chạy; không xem mọi thư mục là một thí nghiệm đã hoàn tất.

Hướng dẫn tái lập: `research/structure_chunker/REPRODUCE.md`. Một số script lưu đường dẫn Windows của máy gốc; cần chỉnh đường dẫn hoặc đặt `AC_MODELS`, `AC_HARNESS` theo hướng dẫn. API chính thức và gateway là hai điều kiện riêng, không gộp kết quả.
