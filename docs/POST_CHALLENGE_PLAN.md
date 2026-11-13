# Kế hoạch sau challenge V2D

Viết ngày 13/11/2026. Challenge đã đóng bảng xếp hạng (4/11); phần dưới là cách biến những gì đã làm thành demo và ứng dụng.

## Ta đang có gì trong tay

| Khối | Trạng thái | Dùng lại được cho |
|---|---|---|
| Perception: video → depth (MoGe-2), mask (SAM 2.1), keypoint (MediaPipe) | Chạy được trên laptop 4050 và server 4090 | Mọi demo bên dưới |
| Tái dựng 3D người + vật từ 1 video (Track 1) | Chạy đủ 30 episode, đã có điểm Kaggle; độ chính xác còn yếu (CD-H 40.6 cm, CD-O 80.4 cm) | Demo A, C |
| Train policy robot từ chuyển động người (FlashCHORD, Track 3) | Policy mini 93.4% thành công trên ví dụ kèm repo | Demo A, B |
| Policy → ONNX fp32, chạy CPU 0.1 ms/bước | Xong, kiểm closed-loop trong sim (92.4%) | Demo B |
| Depth model int8 34 MB, 7.5 fps CPU laptop | Xong | Demo B, C |

Điểm yếu nhất hiện tại là mắt xích **video → chuyển động 3D chính xác**. Các demo dưới đây được xếp theo mức phụ thuộc vào mắt xích này.

## Ba demo đề xuất

### Demo B — Policy robot chạy trên Renesas R-Car X5H (ưu tiên 1)

- **Nội dung:** board X5H chạy policy hai bàn tay robot (ONNX 4.6 MB) điều khiển mô phỏng trên PC qua mạng; màn hình hiện robot nhặt/đặt hộp, kèm số đo latency và công suất trên board.
- **Vì sao trước:** không phụ thuộc độ chính xác Track 1; mọi thứ phía PC đã xong; đúng mục tiêu "deploy lên board không-NVIDIA".
- **Việc cần làm:**
  1. Dựng toolchain X5H (ONNX Runtime CPU trước; NPU/DRP-AI sau nếu toolchain hỗ trợ các op của policy: MatMul, Add, Mul, Relu, Tanh, ReduceMean, Sqrt, Div).
  2. Viết cầu nối sim ↔ board: PC gửi quan sát 504 chiều, board trả hành động 56 chiều (UDP hoặc ZeroMQ), đo latency vòng kín.
  3. Thử int8 cho policy; đo chênh lệch tỉ lệ thành công trong sim.
  4. Chạy depth model int8 trên board, đo fps.
- **Tiêu chí xong:** tỉ lệ thành công trong sim khi policy chạy trên board ≥ 90%; latency suy luận < 5 ms; có video demo.
- **Chặn:** chưa có board. Có thể làm trước bước 2–3 trên một máy ARM bất kỳ (Raspberry Pi, hoặc QEMU) để khi board về chỉ còn bước 1.

### Demo A — "Quay video bằng điện thoại → robot làm theo trong mô phỏng" (ưu tiên 2)

- **Nội dung:** quay một thao tác đơn giản (nhặt hộp, đặt xuống) → pipeline tái dựng → retarget sang tay robot → train policy ngắn → phát lại trong sim. Đây là câu chuyện trọn vẹn nhất để trình bày.
- **Việc cần làm:**
  1. Nâng độ chính xác tái dựng (đang làm, xem mục "Cải thiện Track 1").
  2. Viết bước chuyển kết quả tái dựng (MHR + quỹ đạo vật) sang định dạng tham chiếu mà FlashCHORD nhận; hiện ta mới chạy trên dữ liệu tham chiếu kèm repo.
  3. Chọn 2–3 thao tác dễ, vật cứng, ít che khuất; quay bằng camera tĩnh.
- **Tiêu chí xong:** ít nhất 1 thao tác tự quay đi hết pipeline, policy đạt > 70% thành công trong sim.
- **Rủi ro:** bước 2 chưa thử; tay (ngón) hiện để mặc định nên thao tác cần ngón tay khéo sẽ không ra. Nên chọn thao tác cầm nắm cả bàn tay.
- **Cập nhật 13/11:** làm theo đường góc nhìn thứ nhất + hai tay Sharpa, không qua Track 1 và không dùng MANO / SAM 3 (`phone_demo/README.md`). Bước 2 đã xong và kiểm chứng (retarget không-MANO → train 99.8 %); perception chạy trên clip iPhone mẫu của NVIDIA; đang train trên tham chiếu sinh từ video. Việc còn lại: user quay clip của mình theo hướng dẫn trong README, rồi GR00T.

### Demo C — Theo dõi người + vật 3D từ một camera (ưu tiên 3, tuỳ nhu cầu)

- **Nội dung:** dùng riêng khối perception: video một camera → vị trí 3D người và vật theo thời gian. Ứng dụng: giám sát thao tác ở trạm làm việc, đo tư thế (ergonomics), vùng an toàn người–robot.
- **Việc cần làm:** đóng gói thành một lệnh chạy; đo fps trên X5H với model int8; đánh giá sai số bằng thước đo thật (đặt vật ở khoảng cách biết trước).
- **Lưu ý:** đây là hướng sản phẩm khác với "dữ liệu cho robot"; chỉ làm nếu có người cần.

## Cải thiện Track 1 (đang làm, phục vụ Demo A)

Kaggle vẫn chấm bài nộp muộn nên dùng làm thước đo. Thứ tự:

1. Hiệu chỉnh tỉ lệ depth theo nền tĩnh (camera đứng yên nên depth nền phải không đổi giữa các frame).
2. Fit thân người theo bề mặt depth (không chỉ 15 keypoint) và cho tối ưu cỡ người.
3. Theo dõi vật: loại frame hỏng, chặn lật 180°.
4. Mesh vật kín (hiện chỉ có mặt nhìn thấy ở một frame).
5. Nếu được cấp quyền: thay phần người bằng SAM 3D Body (model của Meta, xuất thẳng định dạng MHR; cần xin quyền trên Hugging Face). Đây là đòn bẩy lớn nhất cho CD-H: các đội đầu bảng đều dùng model người chuyên dụng.

Mốc tham chiếu: đội cộng đồng cùng hướng không-gated đạt CD-H 33 cm; baseline NVIDIA 15.6 / 23.2 (CD-H / CD-O); đầu bảng 11.4 / 16.3.

## Trình tự và thời lượng ước tính

| Giai đoạn | Việc | Ước tính | Phụ thuộc |
|---|---|---|---|
| P1 | Cải thiện Track 1 mục 1–3, nộp lại Kaggle | 2–3 ngày | Server 4090 |
| P2 | Cầu nối sim ↔ board, thử trên máy ARM thay thế; int8 cho policy | 3–4 ngày | Không |
| P3 | Demo B trên X5H thật | 1 tuần sau khi có board | Board + toolchain |
| P4 | Demo A: chuyển tái dựng → FlashCHORD, 1 thao tác tự quay | 1–2 tuần | P1 |
| P5 | Track 2 (humanoid G1) hoặc Demo C | Quyết định sau P3 | — |

Ước tính là phỏng đoán của tôi, chưa có số đo cho P2–P4.

## Việc cần anh quyết định hoặc chuẩn bị

- **Giấy phép.** Đã rà 13/11: `docs/LICENSES.md`. Còn cần pháp chế đọc nguyên văn NVIDIA Open Model License (GR00T) và luật Kaggle của challenge.
- **SAM 3D Body.** Nếu muốn đi mục 5, anh cần tự gửi yêu cầu cấp quyền trên Hugging Face bằng tài khoản của anh.
- **Board X5H.** Ngày dự kiến có board, và toolchain nào (ONNX Runtime CPU, hay NPU qua công cụ của Renesas).
- **Phạm vi.** Demo B một mình đã đủ cho mục tiêu "deploy lên board"; Demo A là phần tốn công nhất.
