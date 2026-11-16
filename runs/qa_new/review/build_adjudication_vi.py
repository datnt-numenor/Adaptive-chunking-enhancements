"""Tao runs/qa_new/review/adjudication_vi.md: bang phan xu song ngu Viet-Anh. CHI DOC du lieu goc.

Chay: .venv\\Scripts\\python.exe -X utf8 runs/qa_new/review/build_adjudication_vi.py [--overwrite]

- Van ban tieng Anh/Phap (cau hoi, cau tra loi, quote) luon doc truc tiep tu qa_new_frozen.jsonl (CU)
  va candidate_v2.jsonl (MOI); ban dich tieng Viet, ly do va cach tach y la du lieu viet tay ben duoi.
- Kiem tra bang script: (1) moi y tieng Anh la chuoi con cua cau tra loi MOI; (2) moi doan trich la chuoi
  con cua mot quote trong evidence MOI; (3) quote do xuat hien DUNG 1 lan tren trang. Sai thi khong ghi file.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
QA_DIR = ROOT / "runs" / "qa_new"
REVIEW = QA_DIR / "review"
FROZEN = QA_DIR / "qa_new_frozen.jsonl"
CAND = REVIEW / "candidate_v2.jsonl"
QUEUE = REVIEW / "adjudication_queue.csv"
AUDIT = REVIEW / "qa_audit.jsonl"
PARSED = ROOT / "data" / "clair" / "adi_parsed"
OUT = REVIEW / "adjudication_vi.md"

# --------------------------------------------------------------------------------------------
# Du lieu viet tay: ban dich (Claude dich), ly do (1-2 cau), tach y cho 23-1 / 29-1 / 30-1.
# q_new_vi / a_new_vi = None nghia la khong doi so voi ban cu.
# --------------------------------------------------------------------------------------------
ITEMS = {
"03-1": dict(
    q_old_vi="Theo chương Khả năng giải thích (Explicability) trong bộ tiêu chuẩn AI có trách nhiệm, ai phải đưa ra quyết định cuối cùng trong một hệ thống AI, và người dùng cần làm được gì khi họ cho rằng hệ thống đã sai?",
    q_new_vi="Trong chương Khả năng giải thích của bộ tiêu chuẩn AI có trách nhiệm, ai phải đưa ra quyết định cuối cùng của một hệ thống AI, và người dùng cần làm được gì khi họ nghĩ hệ thống đã sai?",
    a_old_vi="Con người luôn phải là bên đưa ra quyết định cuối cùng, vì hệ thống không bao giờ được hoàn toàn tự động. Người dùng cũng phải có sẵn một cách để báo lỗi khi họ cho rằng dự đoán hoặc quyết định là sai.",
    a_new_vi=None,
    reason="Câu hỏi cũ dài 31 từ, vượt giới hạn 30 từ của prompt; câu mới chỉ rút gọn cách diễn đạt, nghĩa không đổi. Evidence được mở rộng để chứa hai ý phụ 'không bao giờ hoàn toàn tự động' và 'có sẵn cách báo lỗi'."),
"05-2": dict(
    q_old_vi="Chương Độ bền vững (Robustness) của bộ tiêu chuẩn AI có trách nhiệm liệt kê các nguyên nhân gây lệch dữ liệu (data skew). Ngoài thiết kế dữ liệu huấn luyện, đặc trưng không có trong môi trường vận hành, phụ thuộc dữ liệu và thay đổi ở nguồn dữ liệu phía trước, nguyên nhân cuối cùng được nêu là gì?",
    q_new_vi="(Bản dịch giống câu cũ; chỉ sửa ngữ pháp tiếng Anh.)",
    a_old_vi="Tính toàn vẹn dữ liệu: những điểm không nhất quán trong dữ liệu không được phát hiện ở các hệ thống AI đã triển khai. Các nhóm cần phát hiện lỗi đặc trưng như giá trị thiếu, sai kiểu dữ liệu và giá trị nằm ngoài khoảng.",
    a_new_vi=None,
    reason="Sửa lỗi ngữ pháp tiếng Anh: chủ ngữ số ít 'chapter' phải đi với 'lists', không phải 'list'; nghĩa không đổi. Evidence được mở rộng để chứa câu về các lỗi đặc trưng."),
"08-2": dict(
    q_old_vi="Theo cấu trúc model card của Mitchell và cộng sự trong chương Minh bạch (Transparency) của bộ tiêu chuẩn AI có trách nhiệm, mục Dữ liệu huấn luyện (Training Data) nên chứa gì, và phương án dự phòng khi không có đầy đủ chi tiết là gì?",
    q_new_vi="Theo cấu trúc model card của Mitchell và cộng sự trong chương Minh bạch về AI có trách nhiệm, mục Dữ liệu huấn luyện nên chứa gì, và phương án dự phòng khi không có chi tiết là gì?",
    a_old_vi="Khi có thể, mục này nên giống mục Dữ liệu đánh giá (Evaluation Data), tức là nêu chi tiết về các bộ dữ liệu đã dùng. Nếu không thể, mục này cần đưa thông tin tối thiểu, ví dụ phân bố theo các yếu tố khác nhau trong bộ dữ liệu huấn luyện.",
    a_new_vi=None,
    reason="Câu hỏi cũ dài 32 từ; câu mới rút gọn ('standards chapter on Transparency' thành 'Transparency chapter', bỏ chữ 'full'), nghĩa không đổi. Evidence được mở rộng để chứa câu 'When possible, this section should mirror Evaluation Data'."),
"11-3": dict(
    q_old_vi="Theo luật CCPA của California, một công ty được dùng thêm bao nhiêu thời gian để trả lời một yêu cầu phức tạp của người tiêu dùng, và phải báo cho người đó về việc chậm trễ trong thời hạn bao lâu?",
    q_new_vi="Theo luật CCPA của California, khi các yêu cầu phức tạp hoặc nhiều, doanh nghiệp được kéo dài thời hạn trả lời tổng cộng tối đa bao lâu, và khi nào phải thông báo cho người tiêu dùng?",
    a_old_vi="Khi cần thiết, doanh nghiệp được kéo dài thời hạn trả lời tổng cộng tối đa 90 ngày. Doanh nghiệp phải báo cho người tiêu dùng về việc gia hạn, kèm lý do, trong vòng 45 ngày kể từ khi nhận yêu cầu.",
    a_new_vi=None,
    reason="Câu cũ hỏi 'thêm bao nhiêu thời gian' nên có hai đáp án hợp lệ: §1798.130 (trang 12) nói 'thêm 45 ngày', còn §1798.145(h) (trang 35) nói 'tổng cộng tối đa 90 ngày'. Câu mới hỏi 'tổng cộng' và 'phức tạp hoặc nhiều' nên chỉ khớp §1798.145(h); câu trả lời và evidence giữ nguyên."),
"14-1": dict(
    q_old_vi="Theo chương Minh bạch trong Bộ quy tắc thực hành cho mô hình AI đa dụng (GPAI) thuộc Đạo luật AI của EU, các phiên bản cũ của tài liệu mô hình phải được lưu bao lâu, và yêu cầu cung cấp thêm thông tin của các nhà cung cấp hạ nguồn phải được trả lời nhanh đến mức nào?",
    q_new_vi="Theo chương Minh bạch trong Bộ quy tắc thực hành GPAI của EU, các phiên bản cũ của tài liệu mô hình phải được lưu bao lâu, và yêu cầu thông tin của các nhà cung cấp hạ nguồn phải được trả lời nhanh đến mức nào?",
    a_old_vi="Các phiên bản trước của Tài liệu mô hình phải được lưu đến hết 10 năm sau khi mô hình được đưa ra thị trường. Thông tin bổ sung mà nhà cung cấp hạ nguồn yêu cầu phải được cung cấp trong thời hạn hợp lý, chậm nhất 14 ngày sau khi nhận yêu cầu, trừ trường hợp đặc biệt.",
    a_new_vi=None,
    reason="Câu hỏi cũ dài 32 từ; câu mới rút gọn (bỏ 'AI Act' và tiền tố 'extra-'), nghĩa không đổi. Claude chấm câu này KEEP, chỉ đổi câu hỏi theo minor_fix; evidence giữ nguyên."),
"15-1": dict(
    q_old_vi="Theo Công ước khung của Hội đồng châu Âu về Trí tuệ nhân tạo, mỗi Bên phải nộp báo cáo đầu tiên cho Hội nghị các Bên khi nào, và báo cáo phải bao gồm hoạt động AI của những chủ thể nào?",
    q_new_vi="Theo Công ước khung của Hội đồng châu Âu về Trí tuệ nhân tạo, mỗi Bên phải nộp báo cáo đầu tiên cho Hội nghị các Bên khi nào, và báo cáo đó phải bao gồm hoạt động AI của ai?",
    a_old_vi="Mỗi Bên phải báo cáo trong vòng hai năm đầu sau khi trở thành Bên, và định kỳ sau đó. Báo cáo bao gồm các hoạt động thực hiện Điều 3(1)(a) và (b): hoạt động trong vòng đời hệ thống AI do cơ quan công quyền hoặc tác nhân tư nhân hành động thay mặt họ thực hiện, và các rủi ro, tác động phát sinh từ hoạt động của các tác nhân tư nhân khác.",
    a_new_vi=None,
    reason="Câu hỏi cũ dài 31 từ và có cụm 'the report'; câu mới rút xuống 30 từ và dùng 'that report', nghĩa không đổi. Evidence được mở rộng để chứa cụm nối báo cáo với Điều 3(1)(a)–(b)."),
"16-2": dict(
    q_old_vi="Theo Tuyên bố 3/2024 của EDPB, đối với nhận dạng sinh trắc học từ xa, Đạo luật AI là luật chuyên ngành (lex specialis) so với chỉ thị nào, và chỉ thị đó điều chỉnh loại xử lý dữ liệu nào?",
    q_new_vi="Theo Tuyên bố 3/2024 của EDPB, trong việc sử dụng AI nào thì Đạo luật AI là luật chuyên ngành (lex specialis) so với LED, và Chỉ thị (EU) 2016/680 điều chỉnh loại xử lý dữ liệu nào?",
    a_old_vi="Đạo luật AI là luật chuyên ngành so với LED, tức Chỉ thị (EU) 2016/680. Chỉ thị này điều chỉnh việc cơ quan có thẩm quyền xử lý dữ liệu cá nhân để phòng ngừa, điều tra, phát hiện hoặc truy tố tội phạm, hoặc thi hành hình phạt.",
    a_new_vi="Đối với nhận dạng sinh trắc học từ xa, Đạo luật AI là luật chuyên ngành so với LED (Điều 5(1)(h) Đạo luật AI). Chỉ thị (EU) 2016/680 điều chỉnh việc cơ quan có thẩm quyền xử lý dữ liệu cá nhân để phòng ngừa, điều tra, phát hiện hoặc truy tố tội phạm.",
    reason="Tài liệu chỉ nhắc 'LED' đúng một lần, không định nghĩa và không nói LED chính là Chỉ thị 2016/680; câu cũ vì vậy dựa vào kiến thức ngoài tài liệu. Câu mới hỏi riêng hai ý đều có trong tài liệu (lĩnh vực lex specialis ở chú thích 7; phạm vi Chỉ thị 2016/680 ở chú thích 5) và bỏ ý 'thi hành hình phạt' nằm ngoài quote."),
"23-1": dict(
    q_old_vi="Theo hướng dẫn của Ủy ban châu Âu về Đạo luật AI của EU, nhà cung cấp mô hình AI đa dụng mã nguồn mở được miễn những nghĩa vụ nào, và vẫn phải thực hiện những nghĩa vụ nào?",
    q_new_vi=None,
    a_old_vi="Các nhà cung cấp mã nguồn mở có mô hình không thuộc loại có rủi ro hệ thống được miễn nghĩa vụ lập tài liệu kỹ thuật, nghĩa vụ cung cấp thông tin cho nhà cung cấp hệ thống AI hạ nguồn, và nghĩa vụ chỉ định đại diện được ủy quyền theo Điều 54. Họ vẫn phải có chính sách tuân thủ luật bản quyền của Liên minh châu Âu và công bố bản tóm tắt nội dung dùng để huấn luyện.",
    a_new_vi=None,
    reason="Câu hỏi và câu trả lời không đổi; chỉ đổi evidence. Bản v1 chỉ có quote cho Điều 54 nên thiếu hai nghĩa vụ được miễn và điều kiện 'không có rủi ro hệ thống'; bản r1 (GPT-4.1 đã chấm) vẫn thiếu ý 'bản tóm tắt nội dung huấn luyện', nay đã thêm quote thứ 4 nên phủ đủ 6/6 ý.",
    ideas=[
        ("Open-source providers whose models are not systemic-risk models",
         "Nhà cung cấp mã nguồn mở có mô hình không thuộc loại rủi ro hệ thống",
         ["unless their model is a general-purpose AI model with systemic risk"],
         "trừ khi mô hình là mô hình AI đa dụng có rủi ro hệ thống"),
        ("exempted from the technical documentation duty",
         "được miễn nghĩa vụ lập tài liệu kỹ thuật",
         ["the following obligations do not apply", "the obligation to 'draw up and keep up-to- date the technical documentation"],
         "các nghĩa vụ sau không áp dụng; nghĩa vụ lập và cập nhật tài liệu kỹ thuật"),
        ("the duty to inform downstream AI system providers",
         "(được miễn) nghĩa vụ cung cấp thông tin cho nhà cung cấp hệ thống AI hạ nguồn",
         ["make available information and documentation to providers of AI systems who intend to integrate the general-purpose AI model into their AI systems"],
         "cung cấp thông tin và tài liệu cho nhà cung cấp hệ thống AI muốn tích hợp mô hình đa dụng vào hệ thống của họ"),
        ("the duty to appoint an authorised representative under Article 54",
         "(được miễn) nghĩa vụ chỉ định đại diện được ủy quyền theo Điều 54",
         ["Article 54 AI Act, which concerns the obligation to appoint an authorised representative"],
         "Điều 54 Đạo luật AI, về nghĩa vụ chỉ định đại diện được ủy quyền"),
        ("put in place a policy to comply with Union copyright law",
         "(vẫn phải) có chính sách tuân thủ luật bản quyền của Liên minh châu Âu",
         ["are not exempted from the obligation laid down in Article 53(1), point (c), AI Act to put in place a policy to comply with Union copyright law"],
         "không được miễn nghĩa vụ ở Điều 53(1)(c): có chính sách tuân thủ luật bản quyền của Liên minh châu Âu"),
        ("publish a summary of the content used for training",
         "(vẫn phải) công bố bản tóm tắt nội dung dùng để huấn luyện",
         ["they are not exempted from the obligation laid down in Article 53(1), point (d), AI Act to produce a summary about the content used for training"],
         "không được miễn nghĩa vụ ở Điều 53(1)(d): lập bản tóm tắt về nội dung dùng để huấn luyện"),
    ]),
"30-2": dict(
    q_old_vi="Trong khảo sát của Ifop và Impact AI về cảm nhận của người Pháp đối với AI, bao nhiêu phần trăm người 18–24 tuổi có hình ảnh rất tốt về AI, và tổng tỉ lệ đánh giá tích cực của họ so với nhóm 35–49 tuổi thế nào?",
    q_new_vi="Trong khảo sát của Ifop và Impact AI về cảm nhận của người Pháp đối với AI, bao nhiêu phần trăm người 18–24 tuổi có hình ảnh rất tốt về AI, và đánh giá chung của họ so với nhóm 35–49 tuổi thế nào?",
    a_old_vi="12% người 18–24 tuổi có hình ảnh rất tốt về AI. Tổng tỉ lệ có hình ảnh tốt của họ là 82%, so với 68% ở nhóm 35–49 tuổi.",
    a_new_vi=None,
    reason="Câu hỏi cũ dài 31 từ; câu mới bỏ chữ 'positive' để còn 30 từ, nên 'overall rating' kém rõ nghĩa hơn một chút. Nếu muốn giữ chữ 'positive', có thể dùng: 'In the Ifop–Impact AI survey on French perceptions of AI, what share of 18–24-year-olds rated AI very good, and how did their overall positive rating compare with 35–49-year-olds?' (28 từ). Evidence được mở rộng để chứa tiêu đề cột của bảng."),
}
APPENDIX = {
"29-1": dict(
    q_old_vi="Trong sách trắng về AI tạo sinh của Data For Good, ước tính dấu chân carbon khi huấn luyện BLOOM đã thay đổi thế nào từ phép tính ban đầu đến ước tính mở rộng toàn phạm vi?",
    q_new_vi=None,
    a_old_vi="Ước tính ban đầu là 30 tấn CO2eq. Sau khi tính thêm điện năng tiêu thụ còn lại của trung tâm dữ liệu, việc sản xuất máy chủ và giai đoạn phát triển, con số tăng lên 123,82 tấn CO2eq, khoảng gấp bốn lần ước tính đơn giản ban đầu.",
    a_new_vi=None,
    reason="Câu hỏi và câu trả lời không đổi; chỉ đổi evidence. Bản v1 thiếu tiêu đề cột (cho biết 30 tấn là của BLOOM) và ba thành phần được cộng thêm; bản hiện tại có 4 quote, phủ đủ các ý.",
    ideas=[
        ("The initial estimate was 30 tonnes CO2eq", "Ước tính ban đầu (của BLOOM) là 30 tấn CO2eq",
         ["| Entraînement BLOOM |", "| Impact carbone | 552 tonnes CO2eq | 30 tonnes CO2eq |"],
         "cột 'Huấn luyện BLOOM'; dòng 'Tác động carbon: 552 tấn (GPT-3) | 30 tấn (BLOOM)'"),
        ("data center's residual consumption", "điện năng tiêu thụ còn lại của trung tâm dữ liệu",
         ["une estimation plus poussée a été menée, afin d'étendre le périmètre de calcul",
          "quantifier la vraie consommation résiduelle", "d'électricité du data center"],
         "một ước tính sâu hơn được thực hiện để mở rộng phạm vi tính; định lượng điện năng tiêu thụ còn lại thực tế … của trung tâm dữ liệu (câu bị ngắt trang giữa tr. 88 và 89)"),
        ("server manufacturing", "việc sản xuất máy chủ",
         ["introduire les impacts liés à la fabrication des serveurs de calcul"],
         "đưa vào tác động của việc sản xuất máy chủ tính toán"),
        ("the development phase", "giai đoạn phát triển",
         ["tenir compte de la phase de développement du modèle"],
         "tính đến giai đoạn phát triển mô hình"),
        ("it rose to 123.82 tonnes CO2eq, about four times the naive estimate",
         "tăng lên 123,82 tấn CO2eq, khoảng gấp bốn lần ước tính đơn giản",
         ["L'impact total du modèle calculé est maintenant de 123,82 tonnes CO2eq, soit 4 fois plus que la première estimation plus naïve."],
         "tổng tác động của mô hình nay là 123,82 tấn CO2eq, gấp 4 lần ước tính đầu tiên đơn giản hơn"),
    ]),
"30-1": dict(
    q_old_vi="Trong khảo sát của Ifop cho Microsoft và Impact AI, khảo sát được thực hiện khi nào, và bao nhiêu phần trăm nhân viên ngành CNTT và công nghệ mới có hình ảnh tốt về AI?",
    q_new_vi=None,
    a_old_vi="Khảo sát trực tuyến được thực hiện từ 16 đến 19/10/2018. Nhân viên ngành CNTT và công nghệ mới có hình ảnh về AI tốt nhất, với 92% có hình ảnh tốt (10% rất tốt, 82% khá tốt).",
    a_new_vi=None,
    reason="Câu hỏi và câu trả lời không đổi; chỉ đổi evidence. Bản v1 thiếu tiêu đề cột của bảng (không biết 92/10/82 là 'tốt/rất tốt/khá tốt') và ý 'trực tuyến'; bản hiện tại có 4 quote, phủ đủ các ý.",
    ideas=[
        ("online", "trực tuyến", ["Questionnaire auto-administré en ligne"], "bảng hỏi tự điền trực tuyến"),
        ("from 16 to 19 October 2018", "từ 16 đến 19/10/2018", ["Enquête menée du 16 au 19 octobre 2018"],
         "khảo sát thực hiện từ 16 đến 19/10/2018"),
        ("Employees in IT and new technologies had the best image of AI", "Nhân viên ngành CNTT và công nghệ mới có hình ảnh về AI tốt nhất",
         ["Ce sont les salariés de l'informatique et des nouvelles technologies qui ont la meilleure image de l'IA"],
         "chính nhân viên ngành tin học và công nghệ mới là những người có hình ảnh về AI tốt nhất"),
        ("92% holding a good image", "92% có hình ảnh tốt",
         ["TOTAL bonne image (%)", "| Informatique, nouvelles technologies | 92 |"],
         "cột 'TỔNG hình ảnh tốt (%)'; dòng 'Tin học, công nghệ mới': 92"),
        ("10% very good, 82% fairly good", "10% rất tốt, 82% khá tốt",
         ["Une très bonne image | Plutôt une bonne image", "| 92 | 10 | 82 |"],
         "cột 'Hình ảnh rất tốt | Hình ảnh khá tốt'; cùng dòng: 10 và 82"),
    ]),
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if OUT.exists() and "--overwrite" not in sys.argv:
        print(f"[DUNG] {OUT} da ton tai. Chay lai voi --overwrite neu muon ghi de.")
        return 2
    v1 = {r["qa_id"]: r for r in map(json.loads, FROZEN.read_text(encoding="utf-8").splitlines()) if r}
    v2 = {r["qa_id"]: r for r in map(json.loads, CAND.read_text(encoding="utf-8").splitlines()) if r}
    queue = list(csv.DictReader(QUEUE.open(encoding="utf-8", newline="")))
    latest = {}
    for l in AUDIT.read_text(encoding="utf-8").splitlines():
        if l.strip():
            o = json.loads(l)
            o.setdefault("revision", 1)
            if o["qa_id"] not in latest or o["revision"] >= latest[o["qa_id"]]["revision"]:
                latest[o["qa_id"]] = o

    errs = []
    queue_ids = [r["qa_id"] for r in queue]
    if set(queue_ids) != set(ITEMS):
        errs.append(f"adjudication_queue.csv ({queue_ids}) khac danh sach ITEMS ({list(ITEMS)})")
    n_ideas = n_snips = 0
    pages_cache = {}
    for qid, it in {**ITEMS, **APPENDIX}.items():
        if (it["q_new_vi"] is None) != (v2[qid]["question"] == v1[qid]["question"]):
            errs.append(f"{qid}: q_new_vi khong khop voi viec cau hoi co doi hay khong")
        if (it["a_new_vi"] is None) != (v2[qid]["answer"] == v1[qid]["answer"]):
            errs.append(f"{qid}: a_new_vi khong khop voi viec cau tra loi co doi hay khong")
        doc = v2[qid]["doc_name"]
        if doc not in pages_cache:
            pages_cache[doc] = json.loads((PARSED / f"{doc}.json").read_text(encoding="utf-8"))["pages"]
        for e in v2[qid]["evidence"]:
            if pages_cache[doc][str(e["page"])].count(e["quote"]) != 1:
                errs.append(f"{qid}: quote trang {e['page']} khong xuat hien dung 1 lan")
        for en, _vi, snips, _gloss in it.get("ideas", []):
            n_ideas += 1
            if en not in v2[qid]["answer"]:
                errs.append(f"{qid}: y '{en}' khong co trong cau tra loi")
            for s in snips:
                n_snips += 1
                if not any(s in e["quote"] for e in v2[qid]["evidence"]):
                    errs.append(f"{qid}: doan trich khong nam trong quote nao: {s!r}")
    if errs:
        print("[LOI] Khong ghi file:")
        for e in errs:
            print("  -", e)
        return 1

    qrow = {r["qa_id"]: r for r in queue}
    out = []
    w = out.append
    w("# Bảng phân xử song ngữ Việt–Anh (adjudication)")
    w("")
    w("> Sinh tự động bởi `runs/qa_new/review/build_adjudication_vi.py`. Văn bản tiếng Anh và tiếng Pháp được đọc thẳng từ "
      "`qa_new_frozen.jsonl` (bản **CŨ**, v1) và `candidate_v2.jsonl` (bản **MỚI**, đề xuất). Bản dịch tiếng Việt, lý do và "
      "cách tách ý do Claude viết. Script đã kiểm: mỗi ý tiếng Anh có thật trong câu trả lời, mỗi đoạn trích có thật trong một "
      f"quote của bản MỚI, và mỗi quote xuất hiện đúng 1 lần trên trang ({n_ideas} ý, {n_snips} đoạn trích: tất cả đạt).")
    w("")
    w(f"- `candidate_v2.jsonl` sha256: `{sha(CAND)}`. Đây là bản đã thêm quote cho 23-1, 29-1, 30-1. "
      "GPT-4.1 chấm bản trước đó, nay lưu ở `candidate_v2_r1.jsonl`.")
    w(f"- `qa_new_frozen.jsonl` sha256: `{sha(FROZEN)}`")
    w(f"- Gồm {len(ITEMS)} câu trong `adjudication_queue.csv`, cộng phụ lục 29-1 và 30-1. Hai câu phụ lục không nằm trong hàng đợi, "
      "nhưng có đổi evidence và được tách ý theo yêu cầu.")
    w("- Cách duyệt: đọc phần tiếng Việt, rồi ghi vào dòng **Duyệt** một trong các lựa chọn `ĐỒNG Ý MỚI` / `GIỮ CŨ` / `SỬA KHÁC: …`.")
    w("")
    reason_vi = {"question_changed": "câu hỏi đổi chữ", "answer_changed": "câu trả lời đổi chữ",
                 "gpt_v2_FIX": "GPT-4.1 chưa chấp nhận bản r1"}

    def block(qid, it, idx, in_queue):
        a, b = v1[qid], v2[qid]
        rec = latest[qid]
        w(f"## {idx}. {qid}")
        w("")
        doc_short = a["doc_name"].split("_", 1)[1]
        w(f"- Tài liệu: *{doc_short}*")
        if in_queue:
            q = qrow[qid]
            reasons = ", ".join(reason_vi.get(x, x) for x in q["reasons"].split(";"))
            w(f"- Vì sao vào hàng đợi: {reasons}. Claude chấm v1: **{q['claude_verdict_v1']}**. "
              f"GPT-4.1 chấm bản r1: **{q['gpt_verdict']}**.")
            if q.get("gpt_unsupported_parts"):
                w(f"  - GPT-4.1 chỉ ra phần chưa có trong quote (ở bản r1): “{q['gpt_unsupported_parts']}”.")
        else:
            w(f"- Không nằm trong hàng đợi. Claude chấm v1: **{rec['verdict']}**. Có đổi evidence theo quyết định cho phép 4–5 quote.")
        w("")
        w("**Câu hỏi**")
        w("")
        w(f"- CŨ (EN): {a['question']}")
        w(f"- CŨ (VI): {it['q_old_vi']}")
        if it["q_new_vi"] is None:
            w("- MỚI: *không đổi*")
        else:
            w(f"- MỚI (EN): {b['question']}")
            w(f"- MỚI (VI): {it['q_new_vi']}")
        w("")
        w("**Câu trả lời**")
        w("")
        w(f"- CŨ (EN): {a['answer']}")
        w(f"- CŨ (VI): {it['a_old_vi']}")
        if it["a_new_vi"] is None:
            w("- MỚI: *không đổi*")
        else:
            w(f"- MỚI (EN): {b['answer']}")
            w(f"- MỚI (VI): {it['a_new_vi']}")
        w("")
        w(f"**Lý do thay đổi:** {it['reason']}")
        w("")
        ev_old = ", ".join(f"tr.{e['page']}" for e in a["evidence"])
        ev_new = ", ".join(f"tr.{e['page']}" for e in b["evidence"])
        w(f"**Evidence:** CŨ có {len(a['evidence'])} quote ({ev_old}); MỚI có {len(b['evidence'])} quote ({ev_new})"
          + ("; không đổi." if a["evidence"] == b["evidence"] else "."))
        w("")
        if it.get("ideas"):
            w("**Đối chiếu từng ý của câu trả lời với quote** (đoạn trích là nguyên văn, đã kiểm bằng script)")
            w("")
            w("| # | Ý (EN) | Ý (VI) | Đoạn trích trong quote (nguyên văn) | Nghĩa tiếng Việt của đoạn trích | Trang |")
            w("|---|---|---|---|---|---|")
            for k, (en, vi, snips, gloss) in enumerate(it["ideas"], 1):
                pages = sorted({e["page"] for s in snips for e in b["evidence"] if s in e["quote"]})
                sn = " … ".join(f"«{s}»" for s in snips).replace("|", "\\|")
                w(f"| {k} | {en} | {vi} | {sn} | {gloss.replace('|', '/')} | {', '.join(map(str, pages))} |")
            w("")
        w("**Duyệt: ____**")
        w("")
        w("---")
        w("")

    for i, qid in enumerate(queue_ids, 1):
        block(qid, ITEMS[qid], i, True)
    w("# Phụ lục: 29-1 và 30-1 (đổi evidence, không nằm trong hàng đợi)")
    w("")
    for j, qid in enumerate(APPENDIX, len(queue_ids) + 1):
        block(qid, APPENDIX[qid], j, False)

    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"Da ghi {OUT.relative_to(ROOT)}: {len(queue_ids)} cau hang doi + {len(APPENDIX)} phu luc; "
          f"{n_ideas} y, {n_snips} doan trich da kiem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
