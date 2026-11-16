# Cham cheo bo QA bang GPT-4.1

- Thoi gian: 2026-11-08T13:27:23.244470+00:00 -> 2026-11-08T13:35:00.739641+00:00
- Model yeu cau: `gpt-4.1-2025-04-14`; model tra ve: {'gpt-4.1-2025-04-14': 198}; system_fingerprint: {'fp_9dba188b17': 195, 'fp_2363a7d7d1': 3}
- sha256 v1: `dca47521d0d5922b034f376496c55f960b85c604ef74d1923a31a1ccb6c8bb31`; candidate_v2: `55bdabc29698985e8ffdf9ccd64f6adceb1f4029d990a7fed03de563719b97be`
- So yeu cau: 198; phan hoi hop le: 198; khong hop le: 0
- Chi phi thuc te (tinh tu usage): $1.0446
- GPT-4.1 cham mu: khong duoc thay nhan cua Claude.
- Kappa = n/a khi ca hai ben chi dung mot gia tri nhan (khong tinh duoc).

## v1: dong thuan giua Claude va GPT-4.1 (cung mot ban QA)

So cau so sanh duoc: 99

| Tieu chi | Trung nhan | Kappa |
|---|---:|---:|
| grounded | 81/99 | 0.33 |
| evidence_complete | 50/99 | 0.11 |
| self_contained | 98/99 | 0.00 |
| actual_type | 40/99 | 0.21 |
| unique_answer | 98/99 | 0.00 |
| translation | 97/99 | 0.93 |
| chunking_sensitivity | 27/99 | 0.00 |
| forbidden_source | 99/99 | n/a |
| verdict | 49/99 | 0.11 |

KEEP so voi can sua (FIX/DROP): trung 49/99, kappa 0.11

Bang cheo verdict (hang = Claude, cot = GPT-4.1):

| Claude \ GPT | KEEP | FIX | DROP |
|---|---:|---:|---:|
| KEEP | 42 | 0 | 0 |
| FIX | 50 | 7 | 0 |
| DROP | 0 | 0 | 0 |

## v2 de xuat: GPT-4.1 co chap nhan ban da sua khong

KEEP 98, FIX 1, DROP 0 / 99

Do nhay voi chunking (B7) tren v2: Claude {'HARD': 48, 'MEDIUM': 45, 'EASY': 6}; GPT-4.1 {'MEDIUM': 37, 'EASY': 49, 'HARD': 13}; trung 27/99, kappa 0.01

## Hang doi phan xu: 9 cau (adjudication_queue.csv)

Gom: cau GPT-4.1 khong chap nhan tren v2, cau doi chu question/answer, cau thieu phan hoi hop le.

- 03-1: question_changed
- 05-2: question_changed
- 08-2: question_changed
- 11-3: question_changed
- 14-1: question_changed
- 15-1: question_changed
- 16-2: question_changed;answer_changed
- 23-1: gpt_v2_FIX
- 30-2: question_changed

