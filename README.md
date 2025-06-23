# AutoML Pipeline with EvalML + Streamlit

Đây là một ứng dụng AutoML trực quan, cho phép người dùng không cần viết code vẫn có thể:
- Upload hoặc tạo bảng dữ liệu
- Tự động phân tích EDA bằng Sweetviz
- Tiền xử lý dữ liệu (missing, outlier, rare, one-hot, ADASYN, PCA, ...)
- Cấu hình & huấn luyện mô hình AutoML với EvalML
- Đánh giá mô hình tốt nhất và hiển thị các chỉ số
- Chạy trực tiếp với giao diện Streamlit

---

## Tính năng chính

- Upload CSV, TXT, SQL hoặc tạo bảng thủ công
- Tự động sinh báo cáo Sweetviz
- Tiền xử lý thông minh: missing, outlier, rare category, one-hot, PCA (tuỳ chọn), ADASYN (cân bằng)
- Chia tập train/test sau khi xử lý
- Huấn luyện bằng AutoML (EvalML)
- Chọn chế độ `auto` hoặc `manual`: chọn objective, model family, tuner, algorithm,...
- So sánh mô hình trực quan bằng biểu đồ và bảng
- Đánh giá mô hình tốt nhất trên tập test

---

## Cài đặt

### 1. Clone repo
```bash
git clone https://github.com/your-username/automl-pipeline.git
cd automl-pipeline

2. Tạo môi trường và cài dependencies

conda create -n automl_env python=3.10 -y
conda activate automl_env
pip install -r requirements.txt


⸻

🐘 PostgreSQL (nếu dùng kết nối SQL)

Cài bằng Docker:

docker run --name pg-automl -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres

Hoặc cài local trên Mac/Linux:

brew install postgresql
pg_ctl -D /usr/local/var/postgres start

Kết nối với host:

host: localhost
port: 5432
user: postgres
password: postgres
database: your_database


⸻

Chạy ứng dụng

streamlit run main.py

Ứng dụng sẽ mở tại http://localhost:8501

⸻

Cấu trúc dự án

automl-pipeline/
├── main.py               # Giao diện Streamlit chính
├── importer.py           # Upload, tạo bảng, đọc file
├── profilling.py         # Phân tích EDA, chọn target, tiền xử lý
├── machine_learning.py   # Huấn luyện AutoML, vẽ biểu đồ
├── X_train.csv / X_test.csv ...
├── sweetviz_report.html
├── requirements.txt
└── README.md

⸻

💡 Hướng phát triển
	•	Tích hợp MLflow để log toàn bộ mô hình + chỉ số
	•	Kết nối CI/CD bằng GitHub Actions để deploy pipeline
	•	Tự động lưu mô hình tốt nhất + tải dashboard
	•	Giao diện kéo-thả dữ liệu, không cần viết code

---