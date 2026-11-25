# Nghiên cứu và Triển khai Semantic Communication cho Truyền Văn bản sử dụng mô hình Transformer

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Hệ thống truyền thông ngữ nghĩa (**Semantic Communication - DeepSC**) ứng dụng kiến trúc **Transformer** cho bài toán truyền dữ liệu văn bản dạng End-to-End Joint Source-Channel Coding (JSCC), hỗ trợ 3 ngôn ngữ trọng tâm (**Anh – Việt – Trung**). Dự án tập trung đánh giá hiệu năng hệ thống trên các kênh truyền vật lý (**AWGN**, **Rayleigh Fading**), đo lường độ chống chịu trước các cuộc tấn công đối kháng tầng vật lý (**FGSM**, **C&W**) và triển khai cơ chế phát hiện kết hợp phòng thủ thích ứng (**AttackDetector + AdaptiveDefense**).

---

## 📌 Mục lục
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cài đặt & Yêu cầu](#-cài-đặt--yêu-cầu)
- [Hướng dẫn sử dụng & Tái lập thực nghiệm](#-hướng-dẫn-sử-dụng--tái-lập-thực-nghiệm)
- [Đánh giá & Tấn công đối kháng](#-đánh-giá--tấn-công-đối-kháng)
- [Tác giả](#-tác-giả)
- [Giấy phép](#-giấy-phép)

---

## 🏗 Kiến trúc hệ thống

Hệ thống bao gồm các thành phần cốt lõi:
1. **Semantic Encoder & Decoder**: Khai thác cơ chế Multi-Head Self-Attention của Transformer để trích xuất và khôi phục ý nghĩa câu văn bản thay vì mã hóa chuỗi bit truyền thống.
2. **Channel Codec & Kênh truyền vật lý**: Mã hóa ký hiệu liên tục (`latent_dim = 128`) kèm chuẩn hóa công suất phát, giả lập các môi trường truyền dẫn thực tế (`channel/`):
   - **AWGN Channel**: Kênh nhiễu Gauss trắng cộng.
   - **Rayleigh Fading Channel**: Kênh suy hao đa đường (Multipath Fading) kết hợp cân bằng kênh.
   - **Traditional Pipeline**: Hệ thống đối chứng truyền thống (UTF-8 / Huffman + BPSK).
3. **Adversarial Attacks & Defense System**:
   - **Tấn công (`attacks/`)**: Fast Gradient Sign Method (**FGSM**, $\varepsilon = 0.15$) và Carlini & Wagner (**C&W** chuẩn $L_2$) tác động trực tiếp vào vector biểu diễn ngữ nghĩa.
   - **Phòng thủ (`models/`)**: Bộ phát hiện bất thường 3 lớp (**AttackDetector**: Clean / FGSM / C&W) kết hợp mạng khử nhiễu thích ứng (**AdaptiveDefense**: `FGSMDenoiser` và `CWDenoiser`).

---

## 📂 Cấu trúc thư mục

```text
semantic-communication-deepsc-2026/
├── attacks/                       # Thuật toán tấn công đối kháng (fgsm.py, cw.py)
├── channel/                       # Mô phỏng kênh vật lý (awgn.py, rayleigh.py, traditional.py)
├── models/                        # Kiến trúc DeepSC, Semantic/Channel Codec, AttackDetector & AdaptiveDefense
├── training/                      # Cấu hình (config.py), tiền xử lý dataset & pipeline huấn luyện đối kháng
├── evaluation/                    # Bộ đánh giá định lượng, chạy benchmark JSON/Excel & trực quan hóa
├── demo/                          # Ứng dụng Web Demo tương tác 5 phân hệ (app.py)
├── checkpoints/                   # Bộ từ điển (vocab.json) và trọng số .pth (tải tại mục Releases)
├── data/
│   ├── raw/                       # Ngữ liệu kiểm thử/dev 3 ngôn ngữ (Tatoeba EN-VI/ZH-VI, OPUS-100 en-vi & en-zh)
│   ├── processed/                 # Thông tin metadata bộ dữ liệu đối chứng
│   ├── benchmarks/                # Kết quả khảo sát hiệu năng kênh theo dải SNR (JSON)
│   └── adversarial_benchmarks/    # Kết quả đo kiểm đối kháng FGSM & C&W kèm Audit Log từng câu (Excel/CSV)
├── results/                       # Kết quả thực nghiệm (results.csv), biểu đồ so sánh & nhật ký huấn luyện
├── docs/                          # Tài liệu hướng dẫn sử dụng & báo cáo khóa luận
├── .streamlit/                    # Cấu hình giao diện Streamlit
├── calc_tables.py                 # Script kiểm chứng công thức độ suy giảm (Δ) & tỷ lệ phục hồi (%)
├── run_experiments.py             # Script chạy thực nghiệm so sánh DeepSC vs Traditional & vẽ biểu đồ
├── run_benchmark.bat              # Menu chạy nhanh khảo sát benchmark trên Windows
├── run_demo_full.bat              # Script khởi chạy nhanh giao diện Web Demo trên Windows
├── DESIGN.md                      # Tài liệu đặc tả thiết kế hệ thống
└── requirements.txt               # Danh sách thư viện phụ thuộc của dự án
```

> **Lưu ý về Weights (`.pth`) & Tập Train lớn:** Do giới hạn dung lượng 100 MB/tệp của Git, các trọng số đã huấn luyện (`best_model.pth`, `best_robust_model.pth`) và các tập huấn luyện gốc (`*train.en/vi/zh`) được lưu tại mục **Releases** của repository. Sau khi tải về, giải nén đặt vào đúng thư mục `checkpoints/` và `data/raw/`.

---

## ⚙️ Cài đặt & Yêu cầu

### 1. Yêu cầu hệ thống
- **Hệ điều hành**: Windows 10/11 hoặc Linux
- **Python**: Phiên bản `>= 3.9` (Khuyến nghị Python 3.10 hoặc 3.11)
- **Phần cứng**: Hỗ trợ GPU NVIDIA với CUDA (hoặc chạy bằng CPU)

### 2. Cài đặt môi trường

Clone dự án từ GitHub:
```bash
git clone https://github.com/DoanHoaiAn123/semantic-communication-deepsc-2026.git
cd semantic-communication-deepsc-2026
```

Khởi tạo và kích hoạt môi trường ảo (khuyến nghị):
```powershell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

Cài đặt các gói phụ thuộc:
```bash
pip install --upgrade pip
pip install -r requirements.txt openpyxl
```

---

## 🚀 Hướng dẫn sử dụng & Tái lập thực nghiệm

### 1. Khởi chạy Giao diện Web Demo 5 phân hệ (Streamlit)
- **Cách 1 (Windows - Nhanh nhất)**: Nhấp đúp vào file `run_demo_full.bat`
- **Cách 2 (Dùng lệnh terminal)**:
  ```bash
  streamlit run demo/app.py --server.port 8501
  ```
  Truy cập `http://localhost:8501` để sử dụng 5 Tab: *So sánh 3 mô hình*, *An toàn & Phòng vệ đối kháng (FGSM/C&W)*, *Khảo sát hiệu năng kênh*, *Truyền nhận đa ngôn ngữ (7 ngôn ngữ)* và *Thông tin khóa luận*.

### 2. Chạy khảo sát hiệu năng kênh AWGN / Rayleigh theo dải SNR (Bảng 4.2)
- Chạy qua menu Windows: nhấp đúp `run_benchmark.bat`
- Hoặc chạy trực tiếp trên 2.000 mẫu OPUS-100 Test Split:
  ```bash
  python evaluation/run_benchmark_cache.py --dataset opus_test --samples 0 --channel awgn
  ```

### 3. Đánh giá tấn công đối kháng FGSM / C&W & Phòng thủ (Bảng 4.6, 4.7, 4.9, 4.10)
- **Chạy qua menu tương tác** (chọn Tập 1 `harouzie` / Tập 2 `Tatoeba` / Trọn bộ):
  ```bash
  python evaluation/run_adversarial_benchmark.py
  ```
- **Chạy trực tiếp không cần menu**:
  ```bash
  # Tập 1 (harouzie - 1.000 mẫu)
  python evaluation/run_adversarial_benchmark.py --non-interactive --mode 1 --samples 1000 --eps 0.15 --engine fast

  # Tập 2 (Tatoeba/WMT - 1.000 mẫu)
  python evaluation/run_adversarial_benchmark.py --non-interactive --mode 2 --samples 1000 --eps 0.15 --engine fast
  ```
  *(Đổi `--engine fast` thành `--engine pytorch` để chạy suy luận trực tiếp qua `checkpoints/best_robust_model.pth`).*
- **Đánh giá PyTorch thuần & kiểm chứng bảng suy giảm/phục hồi**:
  ```bash
  python evaluation/evaluate_attacks.py
  python calc_tables.py
  ```

### 4. Huấn luyện mô hình & chạy toàn bộ thí nghiệm so sánh
- Huấn luyện mô hình cơ bản và đối kháng:
  ```bash
  python training/train_pipeline.py
  python training/run_adversarial.py --checkpoint checkpoints/best_model.pth --epochs 2 --cw_steps 15
  ```
- Chạy so sánh DeepSC vs Traditional Pipeline và vẽ biểu đồ (`results/`):
  ```bash
  python run_experiments.py
  ```

---

## 📊 Đánh giá & Tấn công đối kháng

### 1. Thang đo đánh giá hiệu năng
- **Token-level Accuracy / Precision / Recall / F1 Macro**: Đánh giá độ chính xác giải mã từng token và cân bằng đa lớp từ vựng.
- **BLEU Score (`sacrebleu`)**: Đo lường độ trùng khớp n-gram giữa câu khôi phục và câu tham chiếu.
- **BERTScore & Cosine Semantic Similarity (`sentence-transformers`)**: Đánh giá mức độ bảo toàn ngữ nghĩa câu ngay cả khi từ ngữ bề mặt thay đổi.
- **BER (Bit Error Rate)**: Đo tỷ lệ lỗi bit của hệ thống đối chứng BPSK trên kênh vật lý.

### 2. Tấn công & Cơ chế phòng thủ
- **FGSM Attack**: Nhiễu đối kháng một bước theo dấu gradient của hàm mất mát trên không gian ẩn ($\varepsilon = 0.15$).
- **C&W Attack**: Tối ưu hóa chuẩn $L_2$ qua Adam optimizer nhằm tối thiểu hóa nhiễu nhưng phá hủy tối đa ngữ nghĩa (hỗ trợ tùy biến $c$ và số bước lặp).
- **Adaptive Defense**: Phối hợp `AttackDetector` (nhận diện 3 lớp Clean / FGSM / C&W) để định tuyến tín hiệu qua `FGSMDenoiser` hoặc `CWDenoiser` trước khi đưa vào `ChannelDecoder`.

---

## 👤 Tác giả

Đề tài Khóa luận Tốt nghiệp — Ngành **An Toàn Thông Tin**, Trường **Đại học Công Thương TP. Hồ Chí Minh (HUIT)** (2026):
- **Giảng viên hướng dẫn**: ThS. Nguyễn Thị Hồng Thảo
- **Sinh viên thực hiện**:
  1. **Đoàn Hoài An** (Nhóm trưởng) — `2033230001`
  2. **Dương Tuấn Lộc** — `2033230171`
  3. **Trần Võ Anh Lực** — `2033230177`
- **GitHub**: [DoanHoaiAn123](https://github.com/DoanHoaiAn123)
- **Email**: [doanhoaian65@gmail.com](mailto:doanhoaian65@gmail.com)
- **Repository**: [https://github.com/DoanHoaiAn123/semantic-communication-deepsc-2026](https://github.com/DoanHoaiAn123/semantic-communication-deepsc-2026)

---

## 📜 Giấy phép

Dự án phục vụ mục đích nghiên cứu học thuật và được phát hành theo các điều khoản của giấy phép MIT.
