# Nguyên tắc làm việc nhóm

Tài liệu này áp dụng cho toàn bộ thành viên làm việc trên repository
`datnt-numenor/Adaptive-chunking-enhancements`.

## 1. Vai trò của nhánh `main`

- `main` là nhánh tích hợp ổn định của cả nhóm, không phải nhánh cá nhân.
- Không commit hoặc push trực tiếp vào `main`.
- Mọi thay đổi phải đi qua Pull Request (PR).
- Trưởng nhóm `@datnt-numenor` review và quyết định thời điểm merge.
- Thành viên không tự merge PR của mình, kể cả khi giao diện GitHub cho phép.
- Không force-push hoặc xóa `main`.

## 2. Quy tắc đặt tên nhánh

Mỗi nhiệm vụ dùng một nhánh riêng theo mẫu:

```text
<ten-thanh-vien>/<nhiem-vu-ngan-gon>
```

Ví dụ:

```text
dat/phase-j-selector
member2/context-enrichment
member3/statistical-evaluation
```

Không dùng một nhánh cá nhân duy nhất cho nhiều nhiệm vụ không liên quan.

## 3. Bắt đầu một nhiệm vụ

Luôn tạo nhánh mới từ `main` mới nhất:

```powershell
git switch main
git pull --ff-only enhancements main
git switch -c <ten-thanh-vien>/<ten-nhiem-vu>
git push -u enhancements <ten-thanh-vien>/<ten-nhiem-vu>
```

Trước khi sửa code, đọc `AGENTS.md`, `LLM.md` và handoff của phase đang làm.
Không chạy lại các thực nghiệm đã hoàn thành nếu artifact chưa bị hỏng và chưa
được trưởng nhóm chấp thuận.

## 4. Phân chia công việc để tránh xung đột

- Mỗi PR chỉ giải quyết một nhiệm vụ rõ ràng.
- Thống nhất người sở hữu file/module trước khi hai người cùng sửa một khu vực.
- Không sửa file do thành viên khác đang phụ trách nếu chưa trao đổi.
- Không trộn refactor, format hàng loạt hoặc cập nhật dependency vào PR nghiên cứu.
- Nếu cần thay đổi file dùng chung, thông báo nhóm trước và merge thay đổi đó sớm.
- Thường xuyên cập nhật `main` vào nhánh cá nhân; không chờ đến cuối mới xử lý xung đột.

Cập nhật nhánh đang làm:

```powershell
git fetch enhancements
git switch <ten-thanh-vien>/<ten-nhiem-vu>
git merge enhancements/main
```

Nếu có conflict, người tạo PR phải xử lý trên nhánh của mình; không sửa trực
tiếp trên `main`.

## 5. Quy tắc commit và Pull Request

- Commit nhỏ, có nội dung rõ ràng và không chứa file tạm.
- Không commit `.env`, API key, token, cookie, credential hoặc dữ liệu bí mật.
- Không commit virtualenv, cache, model weights hay artifact lớn nếu chưa được duyệt.
- PR phải ghi: mục tiêu, thay đổi chính, cách kiểm thử, artifact liên quan và giới hạn còn lại.
- Nếu kết quả khác paper, tách rõ `Paper says`, `Code/artifact shows` và `Analyst inference`.
- Gắn `@datnt-numenor` làm reviewer.
- Sau khi có review, mọi commit mới phải được review lại.

Checklist trước khi mở PR:

```text
[ ] Nhánh được tạo từ main mới nhất
[ ] Không có secret hoặc file tạm
[ ] Test liên quan đã chạy và kết quả được ghi trong PR
[ ] Manifest/hash được cập nhật nếu tạo artifact nghiên cứu mới
[ ] Không ghi đè cache hoặc kết quả thực nghiệm đã khóa
[ ] Tài liệu/handoff được cập nhật nếu trạng thái phase thay đổi
```

## 6. Điều kiện để merge

Một PR chỉ được merge khi:

1. CI bắt buộc đã vượt qua;
2. không còn conversation hoặc review comment chưa giải quyết;
3. `@datnt-numenor` đã approve;
4. branch không xung đột với `main`;
5. kết quả kiểm thử và giới hạn được mô tả trung thực;
6. trưởng nhóm xác nhận đúng thời điểm tích hợp.

Ưu tiên **Squash and merge** để lịch sử `main` gọn và mỗi PR tương ứng một thay
đổi hoàn chỉnh. Sau khi merge, xóa branch đã hoàn thành.

## 7. Nguyên tắc riêng cho thực nghiệm nghiên cứu

- Pin commit, dependency, model revision, dataset/input hash và seed.
- Lưu command, device, thời gian, chi phí và trạng thái hoàn thành trong manifest.
- Dùng smoke test trước khi chạy toàn bộ hoặc phát sinh phí.
- Cache API trả phí và hỗ trợ resume; không retry mù quáng.
- Không trộn failed/partial output vào kết quả cuối.
- Không sửa hoặc thay thế QA, split hay baseline đã freeze để làm đẹp kết quả.
- Kết luận phải dựa trên artifact có thể kiểm tra, không dựa riêng vào log màn hình.

## 8. Trường hợp khẩn cấp

Nếu `main` bị lỗi nghiêm trọng hoặc lộ secret:

- dừng merge mới;
- thông báo ngay cho trưởng nhóm;
- tạo nhánh `hotfix/<mo-ta>` từ commit an toàn gần nhất;
- thu hồi/rotate secret bên ngoài Git trước khi sửa lịch sử;
- chỉ force-push khi trưởng nhóm xác nhận và đã thông báo toàn nhóm.

