# Report Part 7 - CPU Fallback (LightGBM)

## 1) Tổng quan
- Project ID: sturdy-cable-494205-a2
- Region/Zone: us-central1-a
- Machine type: n2-standard-8
- GPU count: 0

## 2) Mốc thời gian
- Bắt đầu terraform apply: 17:40
- Terraform apply xong: 17:46
- Chạy benchmark.py xong: 18:10
- Tổng thời gian: ~30 phút

## 3) Kết quả benchmark
- Load data time (giây): 2.0239
- Training time (giây): 0.8519
- Best iteration: 1
- AUC-ROC: 0.951649
- Accuracy: 0.998947
- F1-score: 0.727273
- Precision: 0.655738
- Recall: 0.816327
- Inference latency (1 row, ms): 0.733083
- Inference throughput (1000 rows, rows/sec): 1153093.98

## 4) Nhận xét 5-10 dòng
- Phương án CPU fallback được lựa chọn vì tài khoản mới chưa được cấp quota GPU NVIDIA T4 từ Google Cloud.
- Mô hình LightGBM đạt được chất lượng dự báo rất cao với AUC-ROC đạt mức 0.95, chứng minh hiệu quả trên bộ dữ liệu Credit Card Fraud.
- Tốc độ dự báo (inference) cực nhanh, đạt trên 1 triệu dòng mỗi giây nhờ tận dụng sức mạnh của 8 vCPU trên dòng máy n2-standard-8.
- Chi phí ước tính khoảng $0.43/giờ, thực tế trên Billing hiện tại đang ở trạng thái chờ cập nhật (Pending) nhưng rẻ hơn so với phương án chạy GPU.
- Lựa chọn hạ tầng n2-standard-8 là hoàn toàn hợp lý cho các bài toán Tabular Data quy mô này, giúp tối ưu chi phí mà vẫn đảm bảo hiệu năng vượt trội.

## 5) Danh sách minh chứng
- [x] Ảnh chụp terminal chạy python3 benchmark.py
- [x] File benchmark_result.json
- [x] Ảnh chụp GCP Billing Reports (Đang chờ cập nhật)
- [x] Thư mục terraform-gcp đã chỉnh sửa
- [x] Đã thực hiện terraform destroy sau khi nộp bài


