# Báo cáo tổng kết — dự án V2D (NVIDIA Video-to-Data Challenge)

Cập nhật: 13/11/2026. Người thực hiện: Hoang Pham.

## Mục đích

Tìm hiểu và tự chạy lại công nghệ "video → dữ liệu cho robot" của NVIDIA: từ một video thường quay người thao tác với đồ vật, tạo ra dữ liệu và kỹ năng cho robot trong mô phỏng, không cần thiết bị motion-capture hay điều khiển robot bằng tay. Sau đó thu nhỏ các model để chạy trên board không dùng GPU NVIDIA, đích là Renesas R-Car X5H.

Bảng xếp hạng của challenge đã đóng ngày 4/11/2026, nên đây là dự án học và đánh giá năng lực, không phải bài dự thi. Kaggle vẫn chấm bài nộp muộn, nên ta có điểm thật để so.

## Kết quả chính

1. **Robot học được thao tác từ chuyển động của người.** Hai bàn tay robot trong mô phỏng nhặt và đặt hộp khăn giấy theo chuyển động người ghi sẵn, thành công 93.4%.
2. **Policy đó chạy được trên CPU thường.** File 4.6 MB, 0.1 ms mỗi bước điều khiển, kết quả trong mô phỏng gần như giữ nguyên (92.4%).
3. **Tái dựng 3D người và vật từ một video, có điểm chính thức.** Chạy đủ 30 video của challenge và được Kaggle chấm. Độ chính xác còn xa nhóm đầu.
4. **Model đo độ sâu thu nhỏ cho CPU.** 34 MB, 7.5 khung hình/giây trên CPU laptop, sai khác 0.5% so với bản gốc.

Phần cứng đã dùng: một server RTX 4090 (24 GB) dùng chung với người khác, và một laptop RTX 4050. Cấu hình NVIDIA khuyến nghị là GPU 48 GB; ta thu nhỏ quy mô để chạy vừa.

## Chi tiết theo hạng mục

### Track 1 — Tái dựng 3D từ video góc nhìn thứ ba

- **Cách làm.** Chỉ dùng model tải tự do: đo độ sâu (MoGe-2), tách người và vật (SAM 2.1), điểm khớp cơ thể (MediaPipe), rồi khớp mô hình cơ thể MHR của Meta và dựng lưới + quỹ đạo vật từ độ sâu. Không dùng các model phải xin quyền (SAM 3D Body, SMPL).
- **Điểm Kaggle (cm, thấp hơn là tốt hơn).**

  | Chỉ số | Bản 1 | Bản 2 | Bản 3 | Baseline NVIDIA | Đầu bảng |
  |---|---|---|---|---|---|
  | Sai số thân người (CD-H) | 40.6 | 34.9 | 34.9 | 15.6 | 11.4 |
  | Sai số vật (CD-O) | 80.4 | 67.5 | 47.7 | 23.2 | 16.3 |
  | Độ mượt chuyển động người (ACC-H) | 0.30 | 0.22 | 0.22 | — | — |
  | Độ mượt chuyển động vật (ACC-O) | 0.60 | 0.35 | 0.31 | — | — |

- **Bản 2 cải thiện nhờ:** hiệu chỉnh tỉ lệ độ sâu theo nền tĩnh, khớp thân người theo bề mặt độ sâu, lọc các khung hình mà vị trí vật bị sai.
- **Bản 3 thêm một ràng buộc:** vật đang được thao tác phải nằm trong tầm tay, nên độ sâu của vật không được lệch độ sâu của người quá 0.3 m. Một phần mức giảm của CD-O đến từ việc kéo vật về gần người, không hẳn là nhìn chính xác hơn.
- **Đánh giá.** Pipeline chạy trọn vẹn và hợp lệ, phần thân người ngang mức đội cộng đồng đi cùng hướng (33 cm), nhưng cả hai chỉ số chính còn gấp đôi baseline. Hai thử nghiệm đã loại trừ hai nguyên nhân nghi ngờ: thân người không bị lật trước/sau ở episode nào, và tiêu cự camera không phải nguồn sai số chính.
- **Muốn tới baseline** cần model ước lượng cơ thể 3D chuyên dụng. Hai lựa chọn đều có ràng buộc giấy phép: SAM 3D Body (phải xin quyền) hoặc NLF (trọng số chỉ cho nghiên cứu phi thương mại).

### Track 3 — Robot hai bàn tay (Sharpa) học từ chuyển động người

- Train bằng FlashCHORD của NVIDIA trên ví dụ hộp khăn giấy kèm repo: 20 triệu bước, 40 phút, 5.8 GB VRAM. Công thức đầy đủ của NVIDIA là 250 triệu bước.
- Kết quả: thành công 93.4% trên 512 lần chạy, sai số vị trí vật 0.64 cm.
- Lưu ý: đây là ví dụ kèm repo, chấm bằng bộ đánh giá của repo, không phải 20 episode kiểm tra của challenge. Nửa "video → chuyển động" của Track 3 chưa làm.

### Track 2 — Robot hình người (Unitree G1 + tay Dex3)

- Mới chạy thử một lần ở quy mô nhỏ trên ví dụ nhặt/đặt hộp snack: 20 triệu bước trong 80 phút, bằng khoảng 6% công thức của NVIDIA (350 triệu bước).
- Kết quả: hệ thống chạy thông từ đầu đến cuối, nhưng policy không làm được nhiệm vụ (thành công 0%). Robot đứng vững, có cử động, không đi tới hộp.
- Muốn làm thật cần train đủ ngân sách, ước tính khoảng một ngày chạy liên tục trên 4090.

### Triển khai lên phần cứng không phải NVIDIA

| Model | Kết quả trên CPU | So với bản gốc |
|---|---|---|
| Policy robot (từ Track 3) | 4.6 MB, 0.1 ms/bước trên 1 luồng CPU | Thành công 92.4% so với 93.4% |
| Model độ sâu (MoGe-2 nhỏ, lượng tử hoá 8-bit) | 34 MB, 7.5 khung hình/giây trên CPU laptop | Sai khác độ sâu 0.5%, nhỏ hơn 3.6 lần |

Board đích R-Car X5H chưa có, nên chưa chạy gì trên board.

## Dùng được vào việc gì

1. **Demo policy robot chạy trên R-Car X5H** (gần nhất). Board chạy policy, điều khiển robot trong mô phỏng trên PC. Phần PC đã xong, chỉ chờ board.
2. **Demo "quay video → robot làm theo trong mô phỏng".** Dùng workflow chính thức của NVIDIA cho video góc nhìn thứ nhất, cộng phần train ta đã chạy được. Cần thêm công dựng môi trường và GPU trống.
3. **Theo dõi người và vật 3D từ một camera** cho giám sát thao tác hoặc an toàn người–robot. Chỉ làm nếu có nhu cầu.

Chi tiết từng hướng: `docs/POST_CHALLENGE_PLAN.md`.

## Hạn chế và rủi ro

- Độ chính xác tái dựng 3D chưa đủ để dùng trực tiếp làm dữ liệu cho robot.
- Mọi kết quả robot đều trong mô phỏng. Đưa lên robot thật cần phần cứng đúng loại và thêm bước huấn luyện model nhìn bằng camera (GR00T) mà ta chưa làm.
- Server dùng chung nên quy mô train bị giới hạn.
- Chưa rà giấy phép của dữ liệu challenge và các model cho mục đích ngoài nghiên cứu nội bộ.
- Toolchain của R-Car X5H chưa được kiểm chứng.

## Việc tiếp theo đề xuất

1. Khi có board: chạy policy và model độ sâu trên R-Car X5H, đo tốc độ và công suất.
2. Quyết định có dùng model cơ thể 3D chuyên dụng cho Track 1 hay không (liên quan giấy phép).
3. Quyết định có đầu tư train đầy đủ cho robot hình người G1 hay không.

## Tài liệu và video kèm theo

- Nhật ký chi tiết: `docs/RUNLOG.md`. Bản tiếng Anh của báo cáo này: `docs/MANAGER_REPORT.md`.
- Video hai tay robot (trái: chuyển động người, phải: policy của ta): `outputs/m2/sharpa_ref_vs_policy.mp4`.
- Video G1 thử nghiệm: `outputs/g1/g1_mini_policy.mp4`.
- Ảnh kiểm tra tái dựng 3D: `docs/figures/m3_ep09_packed_check.jpg`.
