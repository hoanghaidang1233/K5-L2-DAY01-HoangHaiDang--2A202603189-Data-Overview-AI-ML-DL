# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 11/6/2026**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics: 3.13.18/ 2.11.0+cpu/ 8.4.145**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`= 1 ; `score`= 0.510915 ; `taxonomy_name`= ImageNet-1K

2. Record này nói về **cả bức ảnh**.
Vì đây là bài toán gán nhãn mức độ toàn ảnh. Đầu ra của mô hình chỉ trả về xác suất lớp đại diện cho bối cảnh chung, hoàn toàn không có tọa độ không gian (bounding box/polygon) để định vị một vật thể riêng lẻ nào.

3. Danh sách tên lớp **đi kèm taxonomy/checkpoint**, không phải do model tự nghĩ ra.
Danh sách này thuộc chuẩn ImageNet-1K (1.000 lớp do con người định nghĩa sẵn) và được tích hợp cố định trong siêu dữ liệu của checkpoint `yolo11n-cls.pt`.

4. Cần ghi cả 3 thông tin vì:
- `class_id`: Định danh dạng số nguyên cho máy tính xử lý logic, tính loss và ma trận nhầm lẫn chuẩn xác, không bị lỗi gõ chuỗi.
- `class_name`: Giúp con người (annotator, reviewer) đọc hiểu trực tiếp ý nghĩa của nhãn.
- `taxonomy_name`: Xác định không gian ngữ nghĩa chuẩn (ImageNet-1K). Cùng một ID hoặc một từ có thể mang nghĩa khác nhau ở các bộ dữ liệu khác nhau (vd: ID 0 ở COCO là `person`, còn ở ImageNet là `tench`).

5. Nếu ảnh có nhiều vật/nhiều chủ đề, guideline cần nói rõ **quy tắc ưu tiên**:
- Ưu tiên chọn vật thể chiếm diện tích lớn nhất hoặc nằm ở vị trí trung tâm, rõ nét nhất (chủ thể chính).
- Bảng thứ tự ưu tiên giữa các loại vật thể khi chúng có kích thước ngang nhau.
- Gắn cờ gửi lên Reviewer/QC khi gặp trường hợp ảnh khó, không tự ý suy diễn.

6. Không được lấy `score` của model làm ground truth vì:
`score` chỉ là xác suất thể hiện mức độ tự tin nội tại của mô hình (prediction), không phải chân lý thực tế. Mô hình có thể dự đoán sai với điểm tin cậy cao (overconfident). 
Ground truth bắt buộc phải là nhãn do con người kiểm chứng độc lập theo guideline; lấy score làm ground truth sẽ dẫn đến lỗi suy luận vòng quanh  và tích lũy sai số của mô hình vào tập dữ liệu.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. **Chép một record:**`class_name`:"person"; `score`= 0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`= 113.58 ; `bbox_height`= 279.68
2. - Góc trên-trái ($x_{min}, y_{min}$): Nằm tại tọa độ $(X = 385.33, Y = 69.24)$ pixel (cách mép trái ảnh $385.33$ pixel và cách mép trên đỉnh ảnh $69.24$ pixel).
   - Góc dưới-phải ($x_{max}, y_{max}$): Nằm tại tọa độ $(X = 498.92, Y = 348.92)$ pixel (cách mép trái ảnh $498.92$ pixel và cách mép trên đỉnh ảnh $348.92$ pixel).
   - Chiều rộng (bbox_width): $113.58$ pixel (tính bằng $X_{max} - X_{min} = 498.92 - 385.33$).
   - Chiều cao (bbox_height): $279.68$ pixel (tính bằng $Y_{max} - Y_{min} = 348.92 - 69.24$).
3. **Chọn hai ngưỡng (0.20 và 0.60):**
- Tại ngưỡng **0.20**: có **17** prediction (gồm: 2 person, 5 bowl, 2 oven, 2 cup, 3 spoon, 1 potted plant, 1 dining table, 1 bottle).
- Tại ngưỡng **0.60**: có **6** prediction (gồm: 2 person, 3 bowl, 1 oven... các vật thể lớn, rõ ràng).
>-**Nhận xét:** Khi nâng ngưỡng từ 0.20 lên 0.60, số lượng prediction giảm từ 17 xuống còn 6 hộp (giảm gần 2/3). Mô hình chỉ giữ lại các đối tượng có độ tin cậy rất cao và loại bỏ các vật thể nhỏ/khó nhận diện hơn như spoon, potted plant, dining table, bottle, cup.
4. **Khi hạ ngưỡng (giữ nhiều box hơn), người Reviewer phải:**
- Lọc và xóa các hộp rác / dương tính giả (False Positives): Kiểm tra từng hộp trong số lượng lớn box được giữ lại và xóa thủ công các box nhận diện nhầm (nhầm nền, bóng hoặc hoa văn thành vật thể).
- Xử lý hộp trùng lặp / chồng chéo: Loại bỏ các box thừa cùng bao quanh một đối tượng chưa lọc hết.
- Kiểm tra và sửa lại nhãn (class_name): Các box ở ngưỡng thấp rất dễ bị phân loại sai lớp.
- Workload tăng cao: Reviewer phải tốn nhiều thời gian và công sức hơn gấp nhiều lần để duyệt qua danh sách dự đoán dày đặc.
5. Quy tắc vẽ Bounding Box cho Annotator:
Hộp bao phải được vẽ ôm sát vừa khít các điểm cực biên ngoài cùng (trên, dưới, trái, phải) của vật thể. 
Không được để thừa khoảng trống nền quá +-10 % và tuyệt đối không được cắt lẹm vào các bộ phận hoặc vật thể khác
gán nhãn.

6. Vật trong ảnh 'kitchen' bị che khuất là nồi/xoong treo ở trên bếp bị xếp chồng chéo lên nhau, bị che khuất khá nhiều
>**Trường hợp đưa vào Guideline (Quy tắc gán nhãn)**:
>- Cách xử lý: Viết rõ quy định gán nhãn cho vật bị che khuất (occluded) hoặc bị cắt rìa (truncated).
>- Ví dụ quy tắc: "Nếu vật thể bị che khuất dưới 80% và vẫn nhận diện rõ ràng được hình dạng thuộc lớp nào, hãy vẽ bounding box bao quanh toàn bộ phần ước lượng của vật thể đó. Nếu bị cắt rìa bởi mép ảnh, vẽ sát mép ảnh."
>- Lý do: Đây là những trường hợp rất phổ biến trong thực tế (occlusion & truncation) mà một bộ tài liệu Guideline chuẩn bắt buộc phải định nghĩa từ đầu để tất cả Annotators làm việc đồng nhất, tránh mỗi người tự vẽ một kiểu.

>**Trường hợp Escalation (Hỏi ý kiến Lab Coach)**:
>- Cách xử lý: Chỉ đẩy lên Escalation khi gặp trường hợp không rõ, ví dụ: phần bị che khuất quá lớn (chỉ lộ ra một góc rất nhỏ của tay cầm chiếc chảo) khiến không thể chắc chắn đó là cái chảo hay dụng cụ khác, hoặc danh mục Taxonomy hiện tại không có lớp phù hợp (ví dụ: cái giá treo có được tính là một phần của dụng cụ bếp không?).
>- Lý do: Escalation dùng để giải quyết các trường hợp nằm ngoài bộ quy chuẩn hiện tại hoặc ranh giới nhãn không rõ ràng, nhằm giúp Lab Coach cập nhật lại Guideline cho toàn đội ngũ.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.
1
1. **Chép một record:**
- `instance_id`= "kitchen-001" ; 
- `class_name`="person"; 
- `score`= 0.899318;
- số điểm polygon =`polygon_point_count`= 348 ; 
- 3 điểm đầu của =`polygon_xy`=[
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
2.
- **Đường biên thực tế (Contour/Boundary)**: Bám sát theo mép thực tế của vật thể thay vì chỉ là một khung chữ nhật thô
- **Hình học chi tiết ở mức pixel:** Thể hiện rõ các đoạn cong, góc cạnh, phần lồi, phần lõm hoặc các lỗ rỗng bên trong vật thể.
- **Loại bỏ pixel nền:** Tách rời hoàn toàn các điểm ảnh của vật thể khỏi phần nền xung quanh và các vật thể khác nằm sát cạnh.

3. **Công dụng và ý nghĩa của instance_id:**
- Công dụng: Dùng để định danh và phân biệt từng cá thể/đối tượng riêng lẻ trong cùng một bức ảnh tĩnh, cho phép tách bạch các vật thể độc lập dù chúng có cùng chung một lớp.
Ý nghĩa của câu khẳng định: 
- **Không phải class_id**: **class_id** là mã chung đại diện cho toàn bộ chủng loại/loài (ví dụ tất cả các chai đều có cùng class_id), còn instance_id là mã số riêng cho từng cái chai độc lập.
- **Không phải tracking_id**: **tracking_id** dùng trong video để theo dõi hành trình của cùng một đối tượng qua thời gian/nhiều khung hình. Còn **instance_id** chỉ có giá trị cục bộ trong phạm vi bức ảnh tĩnh hiện tại, không có ý nghĩa liên kết hay duy trì danh tính sang các bức ảnh khác.
4. **Quy tắc tách biên hai vật chạm nhau:**
- Khi hai vật thể nằm sát mép hoặc che khuất nhau, annotator phải vẽ đa giác tách rời cho từng vật thể theo 
đúng ranh giới tiếp xúc thật. Tuyệt đối không được vẽ đè/lấn pixel của vật thể này sang vật thể kia hoặc gộp hai vật thể lại thành một polygon chung."

5. Khu vực các chiếc nồi đồng và chảo treo san sát nhau trên giá treo ở chính giữa phía sau (nằm ngay phía trên lò nướng/bếp nấu)
- **Với trường hợp này chúng ta sẽ sử dụng guideline vì:**
- **Tránh tắc nghẽn:** Vật thể bị xếp chồng/che khuất xảy ra rất thường xuyên. Nếu cứ gặp là hỏi (Escalation) sẽ làm chậm tiến độ dự án.
- **Đồng bộ cách xử lý:** Giúp tất cả Annotator có chung một quy chuẩn để tự giải quyết ngay lập tức:
- **Bounding Box:** Vẽ khung bao quanh phần nhìn thấy. Nếu bị che khuất quá 80% (không rõ hình thù) ➔ Bỏ qua.
>Dùng Escalation khi:
>- Chỉ dành cho các trường hợp khó: Khi gặp vật thể biến dạng, bất thường không thể nhận diện 
>- (VD: mờ nhòe không phân biệt nổi là cái nồi hay mớ giẻ lau). Lúc này cần báo cáo để tránh gán sai nhãn (mislabel).
Polygon (Segmentation): Chỉ vẽ bám sát vào phần diện tích thực tế lộ ra ngoài của vật thể bị che.


## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn duy nhất đại diện cho toàn bộ ảnh (`class_name`, `class_id`, `taxonomy_name`) | Ảnh `traffic` có nhiều vật thể tranh chấp (xe bus, taxi, người) gây khó khăn khi chọn 1 nhãn bao trùm | Đọc kỹ guideline để chọn chủ thể chiếm ưu thế nhất (về diện tích/vị trí trung tâm); nếu không phân định được thì gắn cờ Escalation | Kiểm tra tính nhất quán, đảm bảo annotator tuân thủ đúng quy tắc ưu tiên trong guideline, không gán nhãn theo cảm tính |
| Phát hiện vật thể | Danh sách các hộp bao chữ nhật (`bbox_xyxy`) kèm nhãn `class_name` cho từng vật thể | Ở ngưỡng thấp (0.20) xuất hiện nhiều dự đoán rác/sai lớp; các vật thể bị che khuất hoặc xếp chồng (dàn nồi/chảo) | Vẽ bounding box ôm sát viền ngoài cùng của vật thể; gán đúng tên lớp; gắn cờ `occluded` nếu bị che khuất | Kiểm tra độ khít của box (không cắt lẹm, không thừa nền > 10%), loại bỏ các box rác (False Positives) hoặc box trùng lặp |
| Instance segmentation | Tập hợp các điểm đa giác khép kín (`polygon_xy`) kèm `instance_id` cho từng cá thể | Các vật thể dính sát ranh giới với nhau (nồi/chảo treo trên giá) rất dễ bị vẽ lấn ranh giới hoặc gộp chung mask | Dùng công cụ chấm đa giác bám sát mép viền thực tế ở cấp độ pixel; tách rời hoàn toàn đa giác giữa hai vật chạm nhau | Phóng to (zoom) kiểm tra đường biên mask có ôm sát từng pixel không, mask có bị lẹm sang vật bên cạnh hay lọt nền không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
>-Chỉ sử dụng các ảnh mẫu COCO công khai có sẵn với checksum cố định của bài lab, tuyệt đối không đưa ảnh cá nhân, dữ liệu khách hàng/doanh nghiệp hay thông tin định danh nhạy cảm vào báo cáo và mã nguồn công khai.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
>-Dừng ngay lập tức các thao tác xử lý, không chia sẻ hay lan truyền dữ liệu, và báo cáo ngay cho **Lab Coach / Giảng viên (Mentor)** phụ trách học phần để được hướng dẫn xử lý .
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [X] `day1_lab_outputs/classification_predictions.json`
- [X] `day1_lab_outputs/detection_predictions.json`
- [X] `day1_lab_outputs/segmentation_predictions.json`
- [X] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [X] `day1_lab_outputs/visuals/classification_top5.png`
- [X] `day1_lab_outputs/visuals/detection_predictions.png`
- [X] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
