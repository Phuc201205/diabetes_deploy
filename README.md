# Diabetes Risk Screening

Mô-đun dự đoán nguy cơ tiểu đường sử dụng mô hình XGBoost, bộ tiền xử lý BRFSS và ngưỡng phân loại được tối ưu theo F2 trên tập validation.

## Nội dung

- `predictor.py`: chuyển đổi dữ liệu đầu vào và thực hiện dự đoán.
- `db.py`: kết nối SQL Server và lưu input/kết quả dự đoán.
- `models/`: model XGBoost, preprocessor và metadata.
- `test_model.py`: kiểm tra khả năng tải model.
- `test_prediction.py`: chạy thử một dự đoán.
- `test_db.py`, `test_save_prediction.py`: kiểm tra các thao tác với SQL Server.

## Cài đặt

```powershell
py -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

Để chạy phần database, cần cài Microsoft ODBC Driver 17 for SQL Server và cập nhật cấu hình SQL Server trong `db.py`.

## Chạy thử

```powershell
python test_model.py
python test_prediction.py
```

Các script `test_db.py` và `test_save_prediction.py` yêu cầu SQL Server/database đã được cấu hình sẵn.

## Lưu ý y khoa

Đây là công cụ sàng lọc rủi ro. Điểm dự đoán không phải là chẩn đoán và không thay thế xét nghiệm hoặc tư vấn của nhân viên y tế.

