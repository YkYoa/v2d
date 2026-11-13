# Giấy phép các thành phần — dùng được cho mục đích thương mại hay không

Rà ngày 13/11/2026. Đây là bản tra cứu kỹ thuật, không phải ý kiến pháp lý; trước khi đưa vào sản phẩm cần bộ phận pháp chế đọc nguyên văn giấy phép.

Ràng buộc của dự án (anh Hoàng chốt 13/11/2026): **được dùng GR00T; không dùng MANO, SAM 3 / SAM 3D** vì không cho thương mại hoá.

## Kết luận ngắn

- Đường "video điện thoại → dữ liệu robot" (Demo A) làm được **hoàn toàn bằng thành phần cho phép thương mại**: MoGe-2, Grounding DINO, SAM 2.1, MediaPipe Hands, FlashCHORD (Newton/Warp/MuJoCo), asset tay Sharpa, GR00T.
- Workflow e2e chính thức của NVIDIA (`docs/e2e_workflow.md` trong repo `video_to_data`) **không dùng được nguyên trạng**: bước tái dựng tay dùng HaMeR/WiLoR và bước retarget đọc tham số MANO (`MANO_LEFT.pkl`, `MANO_RIGHT.pkl`, thư viện manotorch). Ta thay hai bước này bằng `phone_demo/` (keypoint MediaPipe + IK trên model MuJoCo của tay Sharpa).
- Nhánh cải thiện điểm Track 1 đang thử NLF (trọng số chỉ cho nghiên cứu phi thương mại). Kết quả của nhánh đó **không được đưa vào Demo A hay bất cứ thứ gì định thương mại hoá**.

## Bảng tra

Cột "Nguồn": `HF` = thẻ license trên Hugging Face (đọc qua API ngày 13/11), `GH` = GitHub license API (13/11), `repo` = file LICENSE/NOTICE trong bản clone trên server, `nhớ` = theo hiểu biết của tôi, **chưa đối chiếu lại hôm nay**.

### Dùng được (thương mại)

| Thành phần | Vai trò | Giấy phép | Nguồn |
|---|---|---|---|
| MoGe-2 (`Ruicheng/moge-2-vit{s,b,l}-normal`) | depth mét + tiêu cự | MIT (trọng số); code MIT, phần DINOv2 Apache-2.0 | HF; code: nhớ |
| SAM 2.1 (`facebook/sam2.1-hiera-*`) | mask theo video | Apache-2.0 | HF, GH |
| Grounding DINO (`IDEA-Research/grounding-dino-tiny`) | tìm vật theo chữ | Apache-2.0 | HF |
| MediaPipe Hands / Pose | keypoint tay, người | Apache-2.0 | GH; file model `.task`: nhớ |
| MHR (Meta) | mô hình thân người (Track 1) | Apache-2.0 | GH |
| Repo `nvidia-isaac/video_to_data` (gồm FlashCHORD) | train policy trong sim | code Apache-2.0, tài liệu/skill CC-BY-4.0 | repo |
| Newton, Warp, MuJoCo, JAX | mô phỏng, IK | Apache-2.0 | GH (Newton, MuJoCo); Warp, JAX: nhớ |
| Asset tay Sharpa Wave (URDF/XML/mesh) | robot trong sim | Apache-2.0 (Sharpa Group) | repo |
| Asset Unitree G1 + Dex3 | robot trong sim | BSD-3-Clause | repo |
| Dữ liệu mẫu `ego_recon/tissue_box_simple` | kiểm thử | CC-BY-4.0, datacard ghi "ready for commercial or non-commercial uses" | repo |
| Dataset `nvidia/video_to_data_challenge` | video challenge | CC-BY-4.0 (phải ghi nguồn) | HF |
| GR00T N1.7 (`nvidia/GR00T-N1.7-3B`) | model VLA để fine-tune | NVIDIA Open Model License (trọng số); code Isaac-GR00T Apache-2.0 | HF, GH |
| LeRobot | định dạng dataset | Apache-2.0 | GH |
| Depth Anything V2 **Small** | depth nhẹ cho board | Apache-2.0 | HF |

### Không dùng

| Thành phần | Lý do | Nguồn |
|---|---|---|
| MANO (và mọi thứ cần nó: HaMeR, WiLoR, HaWoR, manotorch, retarget chính thức của NVIDIA) | giấy phép MPI chỉ cho nghiên cứu phi thương mại; bản thương mại phải mua qua Meshcapade | nhớ; ràng buộc dự án |
| WiLoR | trọng số CC-BY-NC-4.0 | HF |
| SAM 3, SAM 3D Body, SAM 3D Objects | "SAM License", phải xin quyền thủ công; ràng buộc dự án không cho dùng | HF; ràng buộc dự án |
| SMPL / SMPL-H (WHAM, CARI4D cần) | đăng ký MPI, phi thương mại | nhớ |
| NLF | trọng số chỉ cho nghiên cứu phi thương mại | theo `docs/BAO_CAO_TONG_KET.md`, chưa tự kiểm |
| Depth Anything V2 **Base/Large** | CC-BY-NC-4.0 | HF (bản Large) |
| FoundationPose, FoundationStereo bản NVlabs trên GitHub | NVIDIA Source Code License, phi thương mại | nhớ; GH chỉ trả "Other" |

### Chưa rõ, cần đọc nguyên văn trước khi dùng

- **NVIDIA Open Model License** (GR00T N1.7; FoundationPose bản NGC). NVIDIA mô tả là cho phép thương mại, nhưng tôi chưa đọc nguyên văn bản hiện hành. Đây là giấy phép quan trọng nhất cần pháp chế xác nhận vì GR00T nằm trên đường chính.
- **Isaac Sim / Isaac Lab** (chỉ cần nếu đi đường render + thu thập dữ liệu chính thức). Isaac Lab là BSD-3 (nhớ); Isaac Sim có điều khoản riêng. Đường FlashCHORD của ta không cần Isaac Sim.
- **Luật của challenge trên Kaggle** có thể thêm điều khoản ngoài CC-BY-4.0 của dataset. Chưa đọc.
- **Asset vật thể của challenge** (mesh 10 vật Track 3): chưa kiểm.

## Quy tắc làm việc rút ra

1. Demo A chỉ dùng thư mục `phone_demo/` + FlashCHORD. Không import gì từ nhánh NLF của Track 1.
2. Vật thể trong demo là hộp tự đo bằng thước (mesh hộp tự sinh), nên không cần model dựng mesh nào.
3. Khi thêm model mới: tra thẻ license trên Hugging Face **và** file LICENSE của repo code (hai thứ có thể khác nhau, ví dụ HaMeR code MIT nhưng phụ thuộc MANO), rồi thêm một dòng vào bảng trên.
4. Sản phẩm sinh ra từ model phi thương mại (pose, mesh, dataset) cũng coi như phi thương mại.
