# Nghiên cứu và Triển khai Semantic Communication cho Truyền Văn bản sử dụng mô hình Transformer

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Hệ thống truyền thông ngữ nghĩa (**Semantic Communication - DeepSC**) ứng dụng kiến trúc **Transformer** cho bài toán truyền dữ liệu văn bản dạng End-to-End Joint Source-Channel Coding (JSCC). Dự án tập trung đánh giá hiệu năng hệ thống trên các kênh truyền vật lý (AWGN, Rayleigh), đo lường độ chống chịu trước các cuộc tấn công đối kháng (FGSM, C&W) và triển khai cơ chế phát hiện kết hợp phòng thủ thích ứng (Adaptive Defense).

---

## 📌 Mục lục
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cài đặt & Yêu cầu](#-cài-đặt--yêu-cầu)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Đánh giá & Tấn công đối kháng](#-đánh-giá--tấn-công-đối-kháng)
- [Tác giả](#-tác-giả)
- [Giấy phép](#-giấy-phép)

---

## 🏗 Kiến trúc hệ thống

Hệ thống bao gồm các thành phần cốt lõi:
1. **Semantic Encoder & Decoder**: Khai thác cơ chế Multi-Head Self-Attention của Transformer để trích xuất và khôi phục ý nghĩa câu văn bản thay vì mã hóa chuỗi bit truyền thống.
2. **Channel Codec**: Mã hóa ký hiệu liên tục (continuous symbols) cho kênh truyền vật lý, giả lập các môi trường nhiễu thực tế:
   - **AWGN Channel** (Additive White Gaussian Noise): Kênh nhiễu Gauss trắng cộng.
   - **Rayleigh Fading Channel**: Kênh suy hao đa đường (Multipath Fading).
3. **Adversarial Attacks & Defense System**:
   - **Tấn công (Attacks)**: Fast Gradient Sign Method (**FGSM**) và Carlini & Wagner (**C&W**) tác động vào vector ngữ nghĩa nhằm làm sai lệch bản tin nhận.
   - **Phòng thủ (Defense)**: Bộ phát hiện bất thường (**Attack Detector**) kết hợp mạng khôi phục thích ứng (**Adaptive Defense**) giúp thanh lọc nhiễu đối kháng trước khi giải mã.

## 📂 Cấu trúc thư mục

```text
semantic-communication/
├── models/              # Định nghĩa DeepSC, Semantic Encoder/Decoder, Channel Codec & Defense
├── channel/             # Giả lập kênh truyền vật lý (AWGN, Rayleigh, Traditional Pipeline)
├── attacks/             # Thuật toán tấn công đối kháng trên không gian ngữ nghĩa (FGSM, C&W)
├── training/            # Pipeline huấn luyện mô hình, lưu Vocab & tiền xử lý Dataset
├── evaluation/          # Đánh giá chỉ số (BLEU, BERTScore, Cosine Similarity) & trực quan hóa
├── demo/                # Giao diện web tương tác trực quan (Streamlit App)
├── docs/                # Báo cáo, tài liệu hướng dẫn và requirements.txt
├── data/                # Dữ liệu huấn luyện & kiểm thử (OPUS-100, Tatoeba, v.v.)
├── checkpoints/         # Lưu trọng số mô hình đã huấn luyện (.pt / .pth)
├── results/             # Kết quả thực nghiệm, biểu đồ so sánh & nhật ký chạy
├── requirements.txt     # Danh sách các thư viện phụ thuộc của dự án
├── run_experiments.py   # Script thực thi toàn bộ kịch bản thực nghiệm
├── run_benchmark.bat    # Script đo đạc hiệu năng tự động trên Windows
└── run_demo_full.bat    # Script khởi chạy nhanh giao diện Web Demo
```

---

## ⚙️ Cài đặt & Yêu cầu

### 1. Yêu cầu hệ thống
- **Hệ điều hành**: Windows 10/11 hoặc Linux
- **Python**: Phiên bản `>= 3.9` (Khuyến nghị Python 3.10 hoặc 3.11)
- **Phần cứng**: Hỗ trợ GPU NVIDIA với CUDA (hoặc CPU)

### 2. Cài đặt môi trường

Clone dự án từ GitHub:
```bash
git clone https://github.com/DoanHoaiAn123/semantic-communication.git
cd semantic-communication
```

Khởi tạo và kích hoạt môi trường ảo (khuyến nghị):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

Cài đặt các gói phụ thuộc:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Hướng dẫn sử dụng

### 1. Khởi chạy Giao diện Demo trực quan (Streamlit)
- **Cách 1 (Windows - Nhanh nhất)**: Nhấp đúp vào file `run_demo_full.bat`
- **Cách 2 (Dùng lệnh terminal)**:
  ```bash
  streamlit run demo/app.py
  ```
  Sau khi khởi chạy, truy cập đường dẫn local: `http://localhost:8501` để trải nghiệm truyền văn bản qua các mức SNR, giả lập kênh truyền và thử nghiệm tấn công/phòng thủ.

### 2. Huấn luyện mô hình (Training)
- Huấn luyện mô hình DeepSC cơ bản:
  ```bash
  python training/train.py
  ```
- Huấn luyện mô hình phòng thủ đối kháng (Adversarial Defense Training):
  ```bash
  python training/train_adversarial.py
  ```

### 3. Đánh giá hiệu năng & Benchmark
- Chạy đo kiểm benchmark tự động trên Windows:
  ```bash
  run_benchmark.bat
  ```
- Hoặc chạy lệnh trực tiếp với tham số tùy chọn:
  ```bash
  python evaluation/run_benchmark_cache.py --dataset all --samples 1000 --channel awgn
  ```

### 4. Chạy toàn bộ thí nghiệm so sánh
Thực thi toàn bộ kịch bản thực nghiệm và sinh biểu đồ so sánh với hệ thống truyền thông truyền thống:
```bash
python run_experiments.py
```

---

## 📊 Đánh giá & Tấn công đối kháng

### 1. Thang đo đánh giá hiệu năng
- **Bilingual Evaluation Understudy (BLEU)**: Đo lường độ trùng khớp n-gram giữa câu khôi phục và câu gốc.
- **BERTScore**: Đánh giá độ tương đồng ngữ nghĩa dựa trên biểu diễn ngữ cảnh sâu của BERT.
- **Cosine Semantic Similarity**: Độ tương đồng vector embedding (Sentence-Transformers).

### 2. Tấn công & Cơ chế phòng thủ
- **FGSM Attack**: Nhiễu đối kháng được tạo ra theo hướng gradient của hàm mất mát.
- **C&W Attack**: Tối ưu hóa can thiệp tối thiểu gây suy giảm nghiêm trọng độ chính xác giải mã ngữ nghĩa.
- **Defense Mechanism**: Sử dụng mô hình nhận diện bất thường kết hợp bộ lọc thích ứng giúp khôi phục bản tin ngay cả khi bị can thiệp đối kháng mạnh.

---

## 👤 Tác giả

- **Họ và tên / GitHub**: [DoanHoaiAn123](https://github.com/DoanHoaiAn123)
- **Email**: [doanhoaian65@gmail.com](mailto:doanhoaian65@gmail.com)
- **Repository**: [https://github.com/DoanHoaiAn123/semantic-communication](https://github.com/DoanHoaiAn123/semantic-communication)

---

## 📜 Giấy phép

Dự án này được cấp phép theo các điều khoản của [Giấy phép MIT](LICENSE).
