# Shard 01 failure audit

## Paper nói gì

Không tìm thấy bằng chứng trong paper/code rằng paper quy định cách lập lịch
request API hoặc cơ chế phục hồi rate limit như một phần của chunking method.

## Code thực sự làm gì

- Upstream `split_documents_from_dir` tạo một task LLM-regex cho mỗi document và
  gửi đồng thời bằng `asyncio.gather` trước khi lưu raw parquet.
- Lần chạy shard 01 đầu tiên thất bại với OpenAI HTTP 429: giới hạn 30.000 TPM đã
  dùng hết và request tiếp theo cần 11.423 token.
- Helper sau đó chuyển sang chạy tuần tự từng document và lưu checkpoint.
- Document 01 đã chạy upstream thành công với return code 0 và đủ bảy method ở
  cả ba stage. Tuy nhiên, validator cũ đánh dấu thất bại vì method `page` có một
  chunk rỗng ở `raw` và `no_oversizing`. Chunk này không còn ở `small_merged`.

## Suy luận của người phân tích và biện pháp

- Shard 01 có các document đủ lớn để việc gửi đồng thời vượt giới hạn TPM.
- Chunk rỗng của `page` ở hai stage trung gian là dữ liệu do pipeline upstream
  tạo ra, không phải lỗi API hay thiếu output. Cấm mọi chunk rỗng ở mọi stage là
  một false negative của helper nghiên cứu.
- Helper tiếp tục giữ nguyên model, prompt, temperature, chunker và
  post-processing; chỉ thay lịch request và checkpoint. Validator mới chỉ cho
  phép chunk rỗng thuộc method `page` ở `raw`/`no_oversizing`, vẫn cấm chunk rỗng
  của method khác và cấm mọi chunk rỗng ở `small_merged`.
