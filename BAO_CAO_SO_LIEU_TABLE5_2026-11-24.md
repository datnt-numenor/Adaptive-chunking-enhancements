# Báo cáo kết quả Table 5: Adaptive Chunking và SDP Chunking

Ngày 24/11/2026. Đánh giá trên **438 câu hỏi thuộc 126 tài liệu**, sử dụng mô hình
`gpt-4o-mini-2024-07-18` qua gateway. Các hệ thống so sánh gồm Adaptive Chunking,
SDP-A (t600), SDP-B và LangChain recursive mặc định.

## 1. Kết quả thực nghiệm

**[Code/Artifact]** Số liệu được tổng hợp từ JSON kết quả thực nghiệm và
[descriptive.csv](runs/t5_analysis_v1/ext_gw/descriptive.csv), làm tròn đến hai chữ số thập phân.

| Chỉ số | Adaptive Chunking | SDP-A (t600) | SDP-B | LangChain recursive |
|---|---:|---:|---:|---:|
| Retrieval Completeness (%) | 81,74 | **92,12** | 92,01 | 90,87 |
| Answer Correctness (%) | 77,81 | **87,88** | 87,35 | 87,45 |
| **Mean hai chỉ số (%)** | **79,77** | **90,00** | **89,68** | **89,16** |
| Correctness trên toàn bộ QA (%) | 67,69 | **77,25** | 76,78 | 76,87 |
| Số câu có trả lời | 381 | 385 | 385 | 385 |
| Số câu từ chối | 57 | 53 | 53 | 53 |
| Tổng số câu hỏi | 438 | 438 | 438 | 438 |

**Cách tính:** Retrieval Completeness là điểm trung bình trên toàn bộ 438 câu;
Answer Correctness là điểm trung bình trên các câu có trả lời. Correctness trên toàn bộ QA
tính câu từ chối bằng 0. **Mean = (Retrieval Completeness + Answer Correctness) / 2**,
tính từ số chưa làm tròn.

## 2. So sánh với Adaptive Chunking

**[Code/Artifact]** Chênh lệch được tính bằng **điểm phần trăm** từ các giá trị chưa làm tròn.

| Phương pháp so với Adaptive | Δ Completeness | Δ Answer Correctness | Δ Mean | Số câu trả lời thêm |
|---|---:|---:|---:|---:|
| SDP-A (t600) | +10,39 | +10,07 | **+10,23** | 4 |
| SDP-B | +10,27 | +9,54 | **+9,91** | 4 |
| LangChain recursive | +9,13 | +9,64 | +9,38 | 4 |

SDP-A có Mean cao hơn SDP-B **0,32 điểm** và LangChain recursive **0,84 điểm**.
Hai cấu hình SDP đạt tỷ lệ trả lời **87,90%**, so với **86,99%** của Adaptive.

## 3. Nhận xét

**[Suy luận]** Trên bộ thực nghiệm này, **SDP-A (t600) đạt kết quả tổng thể tốt nhất**:
dẫn đầu về Retrieval Completeness, Answer Correctness và Mean, đồng thời trả lời nhiều hơn Adaptive 4 câu.

SDP-B cũng đạt điểm cao hơn Adaptive ở cả hai chỉ số chính. So với LangChain recursive,
SDP-B có Completeness cao hơn nhưng Answer Correctness thấp hơn nhẹ; Mean của SDP-B vẫn cao hơn.
Kết quả cho thấy hai cấu hình SDP đạt chất lượng truy xuất và trả lời cao hơn Adaptive trong phép đánh giá Table 5 này.

## 4. Nguồn số liệu

- [JSON final của Adaptive Chunking](runs/t5_ext_gw_4omini_adaptive/judge-full/adaptive@plain/adaptive_score_final.json).
- [JSON kết quả SDP-A (t600)](runs/t5_ext_gw_4omini_sdp_a_t600/judge-full/sdp_a_t600@plain/generated_questions_evaluation_results.json).
- [JSON kết quả SDP-B](runs/t5_ext_gw_4omini_sdp_b/judge-full/sdp_b@plain/generated_questions_evaluation_results.json).
- [JSON kết quả LangChain recursive](runs/t5_ext_gw_4omini_raw__langch_recurs_default/judge-full/raw__langch_recurs_default@plain/generated_questions_evaluation_results.json).
- [Bảng tổng hợp](runs/t5_analysis_v1/ext_gw/descriptive.csv) và [manifest](runs/t5_analysis_v1/ext_gw/manifest.json).
