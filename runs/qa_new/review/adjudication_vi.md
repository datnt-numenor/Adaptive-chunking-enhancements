# Bảng phân xử song ngữ Việt–Anh (adjudication)

> Sinh tự động bởi `runs/qa_new/review/build_adjudication_vi.py`. Văn bản tiếng Anh và tiếng Pháp được đọc thẳng từ `qa_new_frozen.jsonl` (bản **CŨ**, v1) và `candidate_v2.jsonl` (bản **MỚI**, đề xuất). Bản dịch tiếng Việt, lý do và cách tách ý do Claude viết. Script đã kiểm: mỗi ý tiếng Anh có thật trong câu trả lời, mỗi đoạn trích có thật trong một quote của bản MỚI, và mỗi quote xuất hiện đúng 1 lần trên trang (16 ý, 22 đoạn trích: tất cả đạt).

- `candidate_v2.jsonl` sha256: `b8bb620c8889af6bb20e7c7f0d761ae7731e6f6b6da776e89caf70d891872f50`. Đây là bản đã thêm quote cho 23-1, 29-1, 30-1. GPT-4.1 chấm bản trước đó, nay lưu ở `candidate_v2_r1.jsonl`.
- `qa_new_frozen.jsonl` sha256: `dca47521d0d5922b034f376496c55f960b85c604ef74d1923a31a1ccb6c8bb31`
- Gồm 9 câu trong `adjudication_queue.csv`, cộng phụ lục 29-1 và 30-1. Hai câu phụ lục không nằm trong hàng đợi, nhưng có đổi evidence và được tách ý theo yêu cầu.
- Cách duyệt: đọc phần tiếng Việt, rồi ghi vào dòng **Duyệt** một trong các lựa chọn `ĐỒNG Ý MỚI` / `GIỮ CŨ` / `SỬA KHÁC: …`.

## 1. 03-1

- Tài liệu: *Standards - Explicability*
- Vì sao vào hàng đợi: câu hỏi đổi chữ. Claude chấm v1: **FIX**. GPT-4.1 chấm bản r1: **KEEP**.

**Câu hỏi**

- CŨ (EN): According to the responsible-AI standards chapter on Explicability, who must make the final decision in an AI system, and what should users be able to do when they believe it erred?
- CŨ (VI): Theo chương Khả năng giải thích (Explicability) trong bộ tiêu chuẩn AI có trách nhiệm, ai phải đưa ra quyết định cuối cùng trong một hệ thống AI, và người dùng cần làm được gì khi họ cho rằng hệ thống đã sai?
- MỚI (EN): In the responsible-AI standards chapter on Explicability, who must make an AI system's final decision, and what should users be able to do when they think it erred?
- MỚI (VI): Trong chương Khả năng giải thích của bộ tiêu chuẩn AI có trách nhiệm, ai phải đưa ra quyết định cuối cùng của một hệ thống AI, và người dùng cần làm được gì khi họ nghĩ hệ thống đã sai?

**Câu trả lời**

- CŨ (EN): A human should always make the final decision, since systems should never be fully autonomous. Users must also have a built-in way to report a mistake when they think the prediction or decision is wrong.
- CŨ (VI): Con người luôn phải là bên đưa ra quyết định cuối cùng, vì hệ thống không bao giờ được hoàn toàn tự động. Người dùng cũng phải có sẵn một cách để báo lỗi khi họ cho rằng dự đoán hoặc quyết định là sai.
- MỚI: *không đổi*

**Lý do thay đổi:** Câu hỏi cũ dài 31 từ, vượt giới hạn 30 từ của prompt; câu mới chỉ rút gọn cách diễn đạt, nghĩa không đổi. Evidence được mở rộng để chứa hai ý phụ 'không bao giờ hoàn toàn tự động' và 'có sẵn cách báo lỗi'.

**Evidence:** CŨ có 2 quote (tr.3, tr.4); MỚI có 2 quote (tr.3, tr.4).

**Duyệt: ____**

---

## 2. 05-2

- Tài liệu: *Standards - Robustness*
- Vì sao vào hàng đợi: câu hỏi đổi chữ. Claude chấm v1: **FIX**. GPT-4.1 chấm bản r1: **KEEP**.

**Câu hỏi**

- CŨ (EN): The responsible-AI standards chapter on Robustness list causes of data skew. Besides training design, unavailable production features, data dependencies and upstream changes, what final cause is given?
- CŨ (VI): Chương Độ bền vững (Robustness) của bộ tiêu chuẩn AI có trách nhiệm liệt kê các nguyên nhân gây lệch dữ liệu (data skew). Ngoài thiết kế dữ liệu huấn luyện, đặc trưng không có trong môi trường vận hành, phụ thuộc dữ liệu và thay đổi ở nguồn dữ liệu phía trước, nguyên nhân cuối cùng được nêu là gì?
- MỚI (EN): The responsible-AI standards chapter on Robustness lists causes of data skew. Besides training design, unavailable production features, data dependencies and upstream changes, what final cause is given?
- MỚI (VI): (Bản dịch giống câu cũ; chỉ sửa ngữ pháp tiếng Anh.)

**Câu trả lời**

- CŨ (EN): Data integrity: inconsistencies that go unnoticed in deployed AI systems. Teams need to detect feature errors such as missing values, type mismatches and range anomalies.
- CŨ (VI): Tính toàn vẹn dữ liệu: những điểm không nhất quán trong dữ liệu không được phát hiện ở các hệ thống AI đã triển khai. Các nhóm cần phát hiện lỗi đặc trưng như giá trị thiếu, sai kiểu dữ liệu và giá trị nằm ngoài khoảng.
- MỚI: *không đổi*

**Lý do thay đổi:** Sửa lỗi ngữ pháp tiếng Anh: chủ ngữ số ít 'chapter' phải đi với 'lists', không phải 'list'; nghĩa không đổi. Evidence được mở rộng để chứa câu về các lỗi đặc trưng.

**Evidence:** CŨ có 2 quote (tr.1, tr.2); MỚI có 2 quote (tr.1, tr.2).

**Duyệt: ____**

---

## 3. 08-2

- Tài liệu: *Standards - Transparency*
- Vì sao vào hàng đợi: câu hỏi đổi chữ. Claude chấm v1: **FIX**. GPT-4.1 chấm bản r1: **KEEP**.

**Câu hỏi**

- CŨ (EN): Under the Mitchell et al. model card structure in the responsible-AI standards chapter on Transparency, what should the Training Data section contain, and what is the fallback when full detail is unavailable?
- CŨ (VI): Theo cấu trúc model card của Mitchell và cộng sự trong chương Minh bạch (Transparency) của bộ tiêu chuẩn AI có trách nhiệm, mục Dữ liệu huấn luyện (Training Data) nên chứa gì, và phương án dự phòng khi không có đầy đủ chi tiết là gì?
- MỚI (EN): Under the Mitchell et al. model card structure in the responsible-AI Transparency chapter, what should the Training Data section contain, and what is the fallback when detail is unavailable?
- MỚI (VI): Theo cấu trúc model card của Mitchell và cộng sự trong chương Minh bạch về AI có trách nhiệm, mục Dữ liệu huấn luyện nên chứa gì, và phương án dự phòng khi không có chi tiết là gì?

**Câu trả lời**

- CŨ (EN): When possible, it should mirror the Evaluation Data section, which covers details on the datasets used. If that is not possible, it should give minimal information, such as the distribution over various factors in the training datasets.
- CŨ (VI): Khi có thể, mục này nên giống mục Dữ liệu đánh giá (Evaluation Data), tức là nêu chi tiết về các bộ dữ liệu đã dùng. Nếu không thể, mục này cần đưa thông tin tối thiểu, ví dụ phân bố theo các yếu tố khác nhau trong bộ dữ liệu huấn luyện.
- MỚI: *không đổi*

**Lý do thay đổi:** Câu hỏi cũ dài 32 từ; câu mới rút gọn ('standards chapter on Transparency' thành 'Transparency chapter', bỏ chữ 'full'), nghĩa không đổi. Evidence được mở rộng để chứa câu 'When possible, this section should mirror Evaluation Data'.

**Evidence:** CŨ có 2 quote (tr.2, tr.2); MỚI có 2 quote (tr.2, tr.2).

**Duyệt: ____**

---

## 4. 11-3

- Tài liệu: *CCPA Act*
- Vì sao vào hàng đợi: câu hỏi đổi chữ. Claude chấm v1: **FIX**. GPT-4.1 chấm bản r1: **KEEP**.

**Câu hỏi**

- CŨ (EN): Under the California CCPA, how much additional time may a company take to answer a complicated consumer request, and how soon must it notify the person of the delay?
- CŨ (VI): Theo luật CCPA của California, một công ty được dùng thêm bao nhiêu thời gian để trả lời một yêu cầu phức tạp của người tiêu dùng, và phải báo cho người đó về việc chậm trễ trong thời hạn bao lâu?
- MỚI (EN): Under the California CCPA, by up to what total period may a business extend its response when requests are complex or numerous, and when must it notify the consumer?
- MỚI (VI): Theo luật CCPA của California, khi các yêu cầu phức tạp hoặc nhiều, doanh nghiệp được kéo dài thời hạn trả lời tổng cộng tối đa bao lâu, và khi nào phải thông báo cho người tiêu dùng?

**Câu trả lời**

- CŨ (EN): The business may extend the response period by up to 90 days in total when necessary. It must tell the consumer about the extension, with the reasons, within 45 days of receiving the request.
- CŨ (VI): Khi cần thiết, doanh nghiệp được kéo dài thời hạn trả lời tổng cộng tối đa 90 ngày. Doanh nghiệp phải báo cho người tiêu dùng về việc gia hạn, kèm lý do, trong vòng 45 ngày kể từ khi nhận yêu cầu.
- MỚI: *không đổi*

**Lý do thay đổi:** Câu cũ hỏi 'thêm bao nhiêu thời gian' nên có hai đáp án hợp lệ: §1798.130 (trang 12) nói 'thêm 45 ngày', còn §1798.145(h) (trang 35) nói 'tổng cộng tối đa 90 ngày'. Câu mới hỏi 'tổng cộng' và 'phức tạp hoặc nhiều' nên chỉ khớp §1798.145(h); câu trả lời và evidence giữ nguyên.

**Evidence:** CŨ có 2 quote (tr.35, tr.35); MỚI có 2 quote (tr.35, tr.35); không đổi.

**Duyệt: ____**

---

## 5. 14-1

- Tài liệu: *Code of Practice for General Purpose - AI Models Transparency - Chapter 2*
- Vì sao vào hàng đợi: câu hỏi đổi chữ. Claude chấm v1: **KEEP**. GPT-4.1 chấm bản r1: **KEEP**.

**Câu hỏi**

- CŨ (EN): Under the EU AI Act Transparency Chapter of the GPAI Code of Practice, how long must old model documentation versions be kept, and how fast must downstream providers’ extra-information requests be answered?
- CŨ (VI): Theo chương Minh bạch trong Bộ quy tắc thực hành cho mô hình AI đa dụng (GPAI) thuộc Đạo luật AI của EU, các phiên bản cũ của tài liệu mô hình phải được lưu bao lâu, và yêu cầu cung cấp thêm thông tin của các nhà cung cấp hạ nguồn phải được trả lời nhanh đến mức nào?
- MỚI (EN): Under the Transparency Chapter of the EU GPAI Code of Practice, how long must old model documentation versions be kept, and how fast must downstream providers' information requests be answered?
- MỚI (VI): Theo chương Minh bạch trong Bộ quy tắc thực hành GPAI của EU, các phiên bản cũ của tài liệu mô hình phải được lưu bao lâu, và yêu cầu thông tin của các nhà cung cấp hạ nguồn phải được trả lời nhanh đến mức nào?

**Câu trả lời**

- CŨ (EN): Previous versions of the Model Documentation must be kept until 10 years after the model was placed on the market. Additional information requested by downstream providers must be provided within a reasonable timeframe, no later than 14 days after the request, except in exceptional circumstances.
- CŨ (VI): Các phiên bản trước của Tài liệu mô hình phải được lưu đến hết 10 năm sau khi mô hình được đưa ra thị trường. Thông tin bổ sung mà nhà cung cấp hạ nguồn yêu cầu phải được cung cấp trong thời hạn hợp lý, chậm nhất 14 ngày sau khi nhận yêu cầu, trừ trường hợp đặc biệt.
- MỚI: *không đổi*

**Lý do thay đổi:** Câu hỏi cũ dài 32 từ; câu mới rút gọn (bỏ 'AI Act' và tiền tố 'extra-'), nghĩa không đổi. Claude chấm câu này KEEP, chỉ đổi câu hỏi theo minor_fix; evidence giữ nguyên.

**Evidence:** CŨ có 2 quote (tr.5, tr.6); MỚI có 2 quote (tr.5, tr.6); không đổi.

**Duyệt: ____**

---

## 6. 15-1

- Tài liệu: *Council of Europe Framework Convention on Artificial Intelligence and Human Rights, Democracy and the Rule of Law*
- Vì sao vào hàng đợi: câu hỏi đổi chữ. Claude chấm v1: **FIX**. GPT-4.1 chấm bản r1: **KEEP**.

**Câu hỏi**

- CŨ (EN): Under the Council of Europe Framework Convention on Artificial Intelligence, when must each Party first report to the Conference of the Parties, and which actors’ AI activities must the report cover?
- CŨ (VI): Theo Công ước khung của Hội đồng châu Âu về Trí tuệ nhân tạo, mỗi Bên phải nộp báo cáo đầu tiên cho Hội nghị các Bên khi nào, và báo cáo phải bao gồm hoạt động AI của những chủ thể nào?
- MỚI (EN): Under the Council of Europe Framework Convention on Artificial Intelligence, when must each Party first report to the Conference of the Parties, and whose AI activities must that report cover?
- MỚI (VI): Theo Công ước khung của Hội đồng châu Âu về Trí tuệ nhân tạo, mỗi Bên phải nộp báo cáo đầu tiên cho Hội nghị các Bên khi nào, và báo cáo đó phải bao gồm hoạt động AI của ai?

**Câu trả lời**

- CŨ (EN): Each Party must report within the first two years after becoming a Party, and periodically afterwards. The report covers activities giving effect to Article 3(1)(a) and (b): AI lifecycle activities by public authorities or private actors acting on their behalf, and risks and impacts from other private actors’ activities.
- CŨ (VI): Mỗi Bên phải báo cáo trong vòng hai năm đầu sau khi trở thành Bên, và định kỳ sau đó. Báo cáo bao gồm các hoạt động thực hiện Điều 3(1)(a) và (b): hoạt động trong vòng đời hệ thống AI do cơ quan công quyền hoặc tác nhân tư nhân hành động thay mặt họ thực hiện, và các rủi ro, tác động phát sinh từ hoạt động của các tác nhân tư nhân khác.
- MỚI: *không đổi*

**Lý do thay đổi:** Câu hỏi cũ dài 31 từ và có cụm 'the report'; câu mới rút xuống 30 từ và dùng 'that report', nghĩa không đổi. Evidence được mở rộng để chứa cụm nối báo cáo với Điều 3(1)(a)–(b).

**Evidence:** CŨ có 3 quote (tr.8, tr.3, tr.3); MỚI có 3 quote (tr.8, tr.3, tr.3).

**Duyệt: ____**

---

## 7. 16-2

- Tài liệu: *EDPB statement on data protection authorities*
- Vì sao vào hàng đợi: câu hỏi đổi chữ, câu trả lời đổi chữ. Claude chấm v1: **FIX**. GPT-4.1 chấm bản r1: **KEEP**.

**Câu hỏi**

- CŨ (EN): According to EDPB Statement 3/2024, the AI Act is lex specialis for remote biometric identification relative to which directive, and what processing does that directive govern?
- CŨ (VI): Theo Tuyên bố 3/2024 của EDPB, đối với nhận dạng sinh trắc học từ xa, Đạo luật AI là luật chuyên ngành (lex specialis) so với chỉ thị nào, và chỉ thị đó điều chỉnh loại xử lý dữ liệu nào?
- MỚI (EN): According to EDPB Statement 3/2024, for which AI use is the AI Act lex specialis vis-à-vis the LED, and what processing does Directive (EU) 2016/680 govern?
- MỚI (VI): Theo Tuyên bố 3/2024 của EDPB, trong việc sử dụng AI nào thì Đạo luật AI là luật chuyên ngành (lex specialis) so với LED, và Chỉ thị (EU) 2016/680 điều chỉnh loại xử lý dữ liệu nào?

**Câu trả lời**

- CŨ (EN): It is lex specialis relative to the LED, which is Directive (EU) 2016/680. That directive covers the processing of personal data by competent authorities for preventing, investigating, detecting or prosecuting criminal offences, or executing criminal penalties.
- CŨ (VI): Đạo luật AI là luật chuyên ngành so với LED, tức Chỉ thị (EU) 2016/680. Chỉ thị này điều chỉnh việc cơ quan có thẩm quyền xử lý dữ liệu cá nhân để phòng ngừa, điều tra, phát hiện hoặc truy tố tội phạm, hoặc thi hành hình phạt.
- MỚI (EN): For remote biometric identification, the AI Act is lex specialis vis-à-vis the LED (Art. 5(1)(h) AI Act). Directive (EU) 2016/680 governs the processing of personal data by competent authorities for preventing, investigating, detecting or prosecuting criminal offences.
- MỚI (VI): Đối với nhận dạng sinh trắc học từ xa, Đạo luật AI là luật chuyên ngành so với LED (Điều 5(1)(h) Đạo luật AI). Chỉ thị (EU) 2016/680 điều chỉnh việc cơ quan có thẩm quyền xử lý dữ liệu cá nhân để phòng ngừa, điều tra, phát hiện hoặc truy tố tội phạm.

**Lý do thay đổi:** Tài liệu chỉ nhắc 'LED' đúng một lần, không định nghĩa và không nói LED chính là Chỉ thị 2016/680; câu cũ vì vậy dựa vào kiến thức ngoài tài liệu. Câu mới hỏi riêng hai ý đều có trong tài liệu (lĩnh vực lex specialis ở chú thích 7; phạm vi Chỉ thị 2016/680 ở chú thích 5) và bỏ ý 'thi hành hình phạt' nằm ngoài quote.

**Evidence:** CŨ có 2 quote (tr.2, tr.2); MỚI có 2 quote (tr.2, tr.2).

**Duyệt: ____**

---

## 8. 23-1

- Tài liệu: *Provider Guidelines - European Commission*
- Vì sao vào hàng đợi: GPT-4.1 chưa chấp nhận bản r1. Claude chấm v1: **FIX**. GPT-4.1 chấm bản r1: **FIX**.
  - GPT-4.1 chỉ ra phần chưa có trong quote (ở bản r1): “and publish a summary of the content used for training”.

**Câu hỏi**

- CŨ (EN): Under the European Commission's EU AI Act guidelines, which obligations are open-source general-purpose AI model providers exempted from, and which must they still fulfil?
- CŨ (VI): Theo hướng dẫn của Ủy ban châu Âu về Đạo luật AI của EU, nhà cung cấp mô hình AI đa dụng mã nguồn mở được miễn những nghĩa vụ nào, và vẫn phải thực hiện những nghĩa vụ nào?
- MỚI: *không đổi*

**Câu trả lời**

- CŨ (EN): Open-source providers whose models are not systemic-risk models are exempted from the technical documentation duty, the duty to inform downstream AI system providers, and the duty to appoint an authorised representative under Article 54. They must still put in place a policy to comply with Union copyright law and publish a summary of the content used for training.
- CŨ (VI): Các nhà cung cấp mã nguồn mở có mô hình không thuộc loại có rủi ro hệ thống được miễn nghĩa vụ lập tài liệu kỹ thuật, nghĩa vụ cung cấp thông tin cho nhà cung cấp hệ thống AI hạ nguồn, và nghĩa vụ chỉ định đại diện được ủy quyền theo Điều 54. Họ vẫn phải có chính sách tuân thủ luật bản quyền của Liên minh châu Âu và công bố bản tóm tắt nội dung dùng để huấn luyện.
- MỚI: *không đổi*

**Lý do thay đổi:** Câu hỏi và câu trả lời không đổi; chỉ đổi evidence. Bản v1 chỉ có quote cho Điều 54 nên thiếu hai nghĩa vụ được miễn và điều kiện 'không có rủi ro hệ thống'; bản r1 (GPT-4.1 đã chấm) vẫn thiếu ý 'bản tóm tắt nội dung huấn luyện', nay đã thêm quote thứ 4 nên phủ đủ 6/6 ý.

**Evidence:** CŨ có 3 quote (tr.22, tr.23, tr.23); MỚI có 4 quote (tr.22, tr.22, tr.23, tr.23).

**Đối chiếu từng ý của câu trả lời với quote** (đoạn trích là nguyên văn, đã kiểm bằng script)

| # | Ý (EN) | Ý (VI) | Đoạn trích trong quote (nguyên văn) | Nghĩa tiếng Việt của đoạn trích | Trang |
|---|---|---|---|---|---|
| 1 | Open-source providers whose models are not systemic-risk models | Nhà cung cấp mã nguồn mở có mô hình không thuộc loại rủi ro hệ thống | «unless their model is a general-purpose AI model with systemic risk» | trừ khi mô hình là mô hình AI đa dụng có rủi ro hệ thống | 22 |
| 2 | exempted from the technical documentation duty | được miễn nghĩa vụ lập tài liệu kỹ thuật | «the following obligations do not apply» … «the obligation to 'draw up and keep up-to- date the technical documentation» | các nghĩa vụ sau không áp dụng; nghĩa vụ lập và cập nhật tài liệu kỹ thuật | 22 |
| 3 | the duty to inform downstream AI system providers | (được miễn) nghĩa vụ cung cấp thông tin cho nhà cung cấp hệ thống AI hạ nguồn | «make available information and documentation to providers of AI systems who intend to integrate the general-purpose AI model into their AI systems» | cung cấp thông tin và tài liệu cho nhà cung cấp hệ thống AI muốn tích hợp mô hình đa dụng vào hệ thống của họ | 22 |
| 4 | the duty to appoint an authorised representative under Article 54 | (được miễn) nghĩa vụ chỉ định đại diện được ủy quyền theo Điều 54 | «Article 54 AI Act, which concerns the obligation to appoint an authorised representative» | Điều 54 Đạo luật AI, về nghĩa vụ chỉ định đại diện được ủy quyền | 22 |
| 5 | put in place a policy to comply with Union copyright law | (vẫn phải) có chính sách tuân thủ luật bản quyền của Liên minh châu Âu | «are not exempted from the obligation laid down in Article 53(1), point (c), AI Act to put in place a policy to comply with Union copyright law» | không được miễn nghĩa vụ ở Điều 53(1)(c): có chính sách tuân thủ luật bản quyền của Liên minh châu Âu | 23 |
| 6 | publish a summary of the content used for training | (vẫn phải) công bố bản tóm tắt nội dung dùng để huấn luyện | «they are not exempted from the obligation laid down in Article 53(1), point (d), AI Act to produce a summary about the content used for training» | không được miễn nghĩa vụ ở Điều 53(1)(d): lập bản tóm tắt về nội dung dùng để huấn luyện | 23 |

**Duyệt: ____**

---

## 9. 30-2

- Tài liệu: *Livre blanc - Impact AI*
- Vì sao vào hàng đợi: câu hỏi đổi chữ. Claude chấm v1: **FIX**. GPT-4.1 chấm bản r1: **KEEP**.

**Câu hỏi**

- CŨ (EN): In the Ifop–Impact AI survey on French perceptions of AI, what share of 18–24-year-olds had a very good image of AI, and how did their overall positive rating compare with 35–49-year-olds?
- CŨ (VI): Trong khảo sát của Ifop và Impact AI về cảm nhận của người Pháp đối với AI, bao nhiêu phần trăm người 18–24 tuổi có hình ảnh rất tốt về AI, và tổng tỉ lệ đánh giá tích cực của họ so với nhóm 35–49 tuổi thế nào?
- MỚI (EN): In the Ifop–Impact AI survey on French perceptions of AI, what share of 18–24-year-olds had a very good image of AI, and how did their overall rating compare with 35–49-year-olds?
- MỚI (VI): Trong khảo sát của Ifop và Impact AI về cảm nhận của người Pháp đối với AI, bao nhiêu phần trăm người 18–24 tuổi có hình ảnh rất tốt về AI, và đánh giá chung của họ so với nhóm 35–49 tuổi thế nào?

**Câu trả lời**

- CŨ (EN): 12% of 18–24-year-olds had a very good image of AI. Their total positive image was 82%, compared with 68% among 35–49-year-olds.
- CŨ (VI): 12% người 18–24 tuổi có hình ảnh rất tốt về AI. Tổng tỉ lệ có hình ảnh tốt của họ là 82%, so với 68% ở nhóm 35–49 tuổi.
- MỚI: *không đổi*

**Lý do thay đổi:** Câu hỏi cũ dài 31 từ; câu mới bỏ chữ 'positive' để còn 30 từ, nên 'overall rating' kém rõ nghĩa hơn một chút. Nếu muốn giữ chữ 'positive', có thể dùng: 'In the Ifop–Impact AI survey on French perceptions of AI, what share of 18–24-year-olds rated AI very good, and how did their overall positive rating compare with 35–49-year-olds?' (28 từ). Evidence được mở rộng để chứa tiêu đề cột của bảng.

**Evidence:** CŨ có 2 quote (tr.19, tr.19); MỚI có 2 quote (tr.19, tr.19).

**Duyệt: ____**

---

# Phụ lục: 29-1 và 30-1 (đổi evidence, không nằm trong hàng đợi)

## 10. 29-1

- Tài liệu: *DataForGood - Livre Blanc IA Generative*
- Không nằm trong hàng đợi. Claude chấm v1: **FIX**. Có đổi evidence theo quyết định cho phép 4–5 quote.

**Câu hỏi**

- CŨ (EN): In the Data For Good generative AI white paper, how did BLOOM's estimated training carbon footprint change from the initial calculation to the extended full-scope estimate?
- CŨ (VI): Trong sách trắng về AI tạo sinh của Data For Good, ước tính dấu chân carbon khi huấn luyện BLOOM đã thay đổi thế nào từ phép tính ban đầu đến ước tính mở rộng toàn phạm vi?
- MỚI: *không đổi*

**Câu trả lời**

- CŨ (EN): The initial estimate was 30 tonnes CO2eq. After adding the data center's residual consumption, server manufacturing and the development phase, it rose to 123.82 tonnes CO2eq, about four times the naive estimate.
- CŨ (VI): Ước tính ban đầu là 30 tấn CO2eq. Sau khi tính thêm điện năng tiêu thụ còn lại của trung tâm dữ liệu, việc sản xuất máy chủ và giai đoạn phát triển, con số tăng lên 123,82 tấn CO2eq, khoảng gấp bốn lần ước tính đơn giản ban đầu.
- MỚI: *không đổi*

**Lý do thay đổi:** Câu hỏi và câu trả lời không đổi; chỉ đổi evidence. Bản v1 thiếu tiêu đề cột (cho biết 30 tấn là của BLOOM) và ba thành phần được cộng thêm; bản hiện tại có 4 quote, phủ đủ các ý.

**Evidence:** CŨ có 2 quote (tr.88, tr.89); MỚI có 4 quote (tr.88, tr.88, tr.89, tr.89).

**Đối chiếu từng ý của câu trả lời với quote** (đoạn trích là nguyên văn, đã kiểm bằng script)

| # | Ý (EN) | Ý (VI) | Đoạn trích trong quote (nguyên văn) | Nghĩa tiếng Việt của đoạn trích | Trang |
|---|---|---|---|---|---|
| 1 | The initial estimate was 30 tonnes CO2eq | Ước tính ban đầu (của BLOOM) là 30 tấn CO2eq | «\| Entraînement BLOOM \|» … «\| Impact carbone \| 552 tonnes CO2eq \| 30 tonnes CO2eq \|» | cột 'Huấn luyện BLOOM'; dòng 'Tác động carbon: 552 tấn (GPT-3) / 30 tấn (BLOOM)' | 88 |
| 2 | data center's residual consumption | điện năng tiêu thụ còn lại của trung tâm dữ liệu | «une estimation plus poussée a été menée, afin d'étendre le périmètre de calcul» … «quantifier la vraie consommation résiduelle» … «d'électricité du data center» | một ước tính sâu hơn được thực hiện để mở rộng phạm vi tính; định lượng điện năng tiêu thụ còn lại thực tế … của trung tâm dữ liệu (câu bị ngắt trang giữa tr. 88 và 89) | 88, 89 |
| 3 | server manufacturing | việc sản xuất máy chủ | «introduire les impacts liés à la fabrication des serveurs de calcul» | đưa vào tác động của việc sản xuất máy chủ tính toán | 89 |
| 4 | the development phase | giai đoạn phát triển | «tenir compte de la phase de développement du modèle» | tính đến giai đoạn phát triển mô hình | 89 |
| 5 | it rose to 123.82 tonnes CO2eq, about four times the naive estimate | tăng lên 123,82 tấn CO2eq, khoảng gấp bốn lần ước tính đơn giản | «L'impact total du modèle calculé est maintenant de 123,82 tonnes CO2eq, soit 4 fois plus que la première estimation plus naïve.» | tổng tác động của mô hình nay là 123,82 tấn CO2eq, gấp 4 lần ước tính đầu tiên đơn giản hơn | 89 |

**Duyệt: ____**

---

## 11. 30-1

- Tài liệu: *Livre blanc - Impact AI*
- Không nằm trong hàng đợi. Claude chấm v1: **FIX**. Có đổi evidence theo quyết định cho phép 4–5 quote.

**Câu hỏi**

- CŨ (EN): In the Ifop survey for Microsoft and Impact AI, when was the fieldwork carried out, and what share of IT and new-technology employees had a good image of AI?
- CŨ (VI): Trong khảo sát của Ifop cho Microsoft và Impact AI, khảo sát được thực hiện khi nào, và bao nhiêu phần trăm nhân viên ngành CNTT và công nghệ mới có hình ảnh tốt về AI?
- MỚI: *không đổi*

**Câu trả lời**

- CŨ (EN): The online survey was conducted from 16 to 19 October 2018. Employees in IT and new technologies had the best image of AI, with 92% holding a good image (10% very good, 82% fairly good).
- CŨ (VI): Khảo sát trực tuyến được thực hiện từ 16 đến 19/10/2018. Nhân viên ngành CNTT và công nghệ mới có hình ảnh về AI tốt nhất, với 92% có hình ảnh tốt (10% rất tốt, 82% khá tốt).
- MỚI: *không đổi*

**Lý do thay đổi:** Câu hỏi và câu trả lời không đổi; chỉ đổi evidence. Bản v1 thiếu tiêu đề cột của bảng (không biết 92/10/82 là 'tốt/rất tốt/khá tốt') và ý 'trực tuyến'; bản hiện tại có 4 quote, phủ đủ các ý.

**Evidence:** CŨ có 3 quote (tr.18, tr.19, tr.19); MỚI có 4 quote (tr.18, tr.19, tr.19, tr.19).

**Đối chiếu từng ý của câu trả lời với quote** (đoạn trích là nguyên văn, đã kiểm bằng script)

| # | Ý (EN) | Ý (VI) | Đoạn trích trong quote (nguyên văn) | Nghĩa tiếng Việt của đoạn trích | Trang |
|---|---|---|---|---|---|
| 1 | online | trực tuyến | «Questionnaire auto-administré en ligne» | bảng hỏi tự điền trực tuyến | 18 |
| 2 | from 16 to 19 October 2018 | từ 16 đến 19/10/2018 | «Enquête menée du 16 au 19 octobre 2018» | khảo sát thực hiện từ 16 đến 19/10/2018 | 18 |
| 3 | Employees in IT and new technologies had the best image of AI | Nhân viên ngành CNTT và công nghệ mới có hình ảnh về AI tốt nhất | «Ce sont les salariés de l'informatique et des nouvelles technologies qui ont la meilleure image de l'IA» | chính nhân viên ngành tin học và công nghệ mới là những người có hình ảnh về AI tốt nhất | 19 |
| 4 | 92% holding a good image | 92% có hình ảnh tốt | «TOTAL bonne image (%)» … «\| Informatique, nouvelles technologies \| 92 \|» | cột 'TỔNG hình ảnh tốt (%)'; dòng 'Tin học, công nghệ mới': 92 | 19 |
| 5 | 10% very good, 82% fairly good | 10% rất tốt, 82% khá tốt | «Une très bonne image \| Plutôt une bonne image» … «\| 92 \| 10 \| 82 \|» | cột 'Hình ảnh rất tốt / Hình ảnh khá tốt'; cùng dòng: 10 và 82 | 19 |

**Duyệt: ____**

---
