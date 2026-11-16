# Cham cheo bo QA bang GPT-4.1

- Thoi gian: 2026-11-08T13:49:31.475594+00:00 -> 2026-11-08T13:49:31.784616+00:00
- Model yeu cau: `gpt-4.1-2025-04-14`; model tra ve: {'gpt-4.1-2025-04-14': 99}; system_fingerprint: {'fp_9dba188b17': 97, 'fp_2363a7d7d1': 2}
- sha256 v1: `dca47521d0d5922b034f376496c55f960b85c604ef74d1923a31a1ccb6c8bb31`; candidate_v2: `b8bb620c8889af6bb20e7c7f0d761ae7731e6f6b6da776e89caf70d891872f50`
- So yeu cau: 99; phan hoi hop le: 99; khong hop le: 0
- Chi phi thuc te (tinh tu usage): $0.5273
- GPT-4.1 cham mu: khong duoc thay nhan cua Claude.
- Kappa = n/a khi ca hai ben chi dung mot gia tri nhan (khong tinh duoc).

## v2 de xuat: GPT-4.1 co chap nhan ban da sua khong

KEEP 99, FIX 0, DROP 0 / 99

Do nhay voi chunking (B7) tren v2: Claude {'HARD': 48, 'MEDIUM': 45, 'EASY': 6}; GPT-4.1 {'MEDIUM': 38, 'EASY': 49, 'HARD': 12}; trung 26/99, kappa -0.00

## Hang doi phan xu: 8 cau (adjudication_queue.csv)

Gom: cau GPT-4.1 khong chap nhan tren v2, cau doi chu question/answer, cau thieu phan hoi hop le.

- 03-1: question_changed
- 05-2: question_changed
- 08-2: question_changed
- 11-3: question_changed
- 14-1: question_changed
- 15-1: question_changed
- 16-2: question_changed;answer_changed
- 30-2: question_changed

