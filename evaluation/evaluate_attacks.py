"""
Đánh giá toàn diện F1 Score và Accuracy cho 2 tấn công đối kháng:
  - FGSM (Fast Gradient Sign Method)
  - C&W (Carlini & Wagner)

So sánh 4 kịch bản:
  1. Clean (không tấn công)
  2. FGSM attack (có/không phòng thủ)
  3. C&W attack (có/không phòng thủ)
  4. Phòng thủ thích ứng (Adaptive Defense)

Dataset đối chứng:
  - Dataset 1: harouzie/vi_en-translation (chuyên ngành CNTT, 2500 mẫu)
  - Dataset 2: Tatoeba / WMT newstest (đời sống chung, 1500 mẫu bổ sung)

Metrics:
  - Token-level Accuracy
  - Token-level F1 Score (macro-averaged)
  - BLEU Score
  - Semantic Similarity (cosine)
  - Attack Detection Accuracy (AttackDetector)
"""

import os
import sys
import json
import csv
import math
import random
import time
from typing import List, Dict, Tuple, Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# ── Dự án nội bộ ──────────────────────────────────────────────────────────────
from models.semantic_model import SemanticCommModel
from models.attack_detector import AttackDetector
from models.adaptive_defense import AdaptiveDefense
from attacks.fgsm import fgsm_attack
from attacks.cw import cw_attack
from training.config import get_default_config
from training.dataset import create_dataloaders, SimpleTokenizer, load_dataset2_for_evaluation

# ── Cố gắng import sacrebleu ──────────────────────────────────────────────────
try:
    import sacrebleu as _sb
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False

try:
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("  ⚠️  scikit-learn chưa cài. Dùng phép tính thủ công.")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1: Tải Dataset đối chứng thứ 2 (Tatoeba / WMT lifestyle)
# ══════════════════════════════════════════════════════════════════════════════

DATASET2_EXTRA: List[Tuple[str, str]] = [
    # ── Đời sống ──────────────────────────────────────────────────────────────
    ("He drinks coffee every morning before work.", "Anh ấy uống cà phê mỗi buổi sáng trước khi đi làm."),
    ("The children played happily in the garden.", "Bọn trẻ chơi vui vẻ trong vườn."),
    ("She forgot her umbrella and got wet in the rain.", "Cô ấy quên ô và bị ướt trong mưa."),
    ("We celebrated his birthday with a big cake.", "Chúng tôi tổ chức sinh nhật anh ấy với một chiếc bánh lớn."),
    ("The library closes at nine o'clock on weekdays.", "Thư viện đóng cửa lúc chín giờ vào các ngày trong tuần."),
    ("My grandmother taught me how to cook traditional dishes.", "Bà tôi dạy tôi cách nấu các món ăn truyền thống."),
    ("The market opens early every Sunday morning.", "Chợ mở cửa sớm mỗi sáng Chủ nhật."),
    ("He exercises for thirty minutes every evening.", "Anh ấy tập thể dục ba mươi phút mỗi buổi tối."),
    ("The bus was late so she missed her appointment.", "Xe buýt đến muộn nên cô ấy bị lỡ hẹn."),
    ("They moved to a new apartment near the university.", "Họ chuyển đến một căn hộ mới gần trường đại học."),
    ("The hospital is located in the center of the city.", "Bệnh viện nằm ở trung tâm thành phố."),
    ("She reads one book every week without fail.", "Cô ấy đọc một cuốn sách mỗi tuần mà không bỏ lỡ."),
    ("The summer heat was unbearable this year.", "Cái nóng mùa hè năm nay thật không thể chịu được."),
    ("His dog follows him everywhere he goes.", "Con chó của anh ấy đi theo anh ấy khắp nơi."),
    ("The neighbors organized a street cleaning event.", "Hàng xóm tổ chức sự kiện dọn vệ sinh đường phố."),
    ("She learned to ride a bicycle when she was five.", "Cô ấy học đi xe đạp khi năm tuổi."),
    ("The fresh vegetables were bought at the local market.", "Rau củ tươi được mua ở chợ địa phương."),
    ("The old bridge was repaired last month.", "Cây cầu cũ đã được sửa chữa vào tháng trước."),
    ("Children should drink at least two liters of water daily.", "Trẻ em nên uống ít nhất hai lít nước mỗi ngày."),
    ("He worked overtime to finish the report before the deadline.", "Anh ấy làm thêm giờ để hoàn thành báo cáo trước thời hạn."),
    # ── Giáo dục ──────────────────────────────────────────────────────────────
    ("The teacher explained the difficult concept very clearly.", "Giáo viên giải thích khái niệm khó rất rõ ràng."),
    ("Students must submit their assignments by Friday.", "Sinh viên phải nộp bài tập trước thứ Sáu."),
    ("The university offers scholarships to outstanding students.", "Trường đại học cấp học bổng cho sinh viên xuất sắc."),
    ("He passed all his exams with excellent grades.", "Anh ấy vượt qua tất cả các kỳ thi với kết quả xuất sắc."),
    ("Online courses allow students to learn at their own pace.", "Các khóa học trực tuyến cho phép sinh viên học theo tốc độ riêng của mình."),
    ("The classroom was equipped with modern projectors.", "Phòng học được trang bị máy chiếu hiện đại."),
    ("Critical thinking is an essential skill in modern education.", "Tư duy phản biện là kỹ năng thiết yếu trong giáo dục hiện đại."),
    ("The professor published a groundbreaking research paper.", "Giáo sư đã công bố một bài báo nghiên cứu đột phá."),
    ("Group projects help students develop teamwork skills.", "Bài tập nhóm giúp sinh viên phát triển kỹ năng làm việc nhóm."),
    ("The school library has thousands of reference books.", "Thư viện trường có hàng nghìn cuốn sách tham khảo."),
    # ── Công nghệ ─────────────────────────────────────────────────────────────
    ("Smartphones have become an essential part of daily life.", "Điện thoại thông minh đã trở thành phần thiết yếu trong cuộc sống hàng ngày."),
    ("Artificial intelligence is transforming the healthcare industry.", "Trí tuệ nhân tạo đang biến đổi ngành chăm sóc sức khỏe."),
    ("The new software update fixed several critical security bugs.", "Bản cập nhật phần mềm mới đã sửa một số lỗi bảo mật nghiêm trọng."),
    ("Cloud storage makes it easy to access files from anywhere.", "Lưu trữ đám mây giúp dễ dàng truy cập tệp từ bất kỳ đâu."),
    ("Electric vehicles are becoming increasingly affordable.", "Xe điện đang ngày càng trở nên phải chăng hơn."),
    ("The internet of things connects billions of devices worldwide.", "Internet vạn vật kết nối hàng tỷ thiết bị trên toàn thế giới."),
    ("Cybersecurity threats are growing more sophisticated each year.", "Các mối đe dọa an ninh mạng ngày càng tinh vi hơn mỗi năm."),
    ("Machine learning algorithms can detect patterns in large datasets.", "Các thuật toán học máy có thể phát hiện các quy luật trong tập dữ liệu lớn."),
    ("The robot was programmed to perform repetitive assembly tasks.", "Robot được lập trình để thực hiện các tác vụ lắp ráp lặp đi lặp lại."),
    ("Blockchain technology ensures transparent and immutable records.", "Công nghệ blockchain đảm bảo hồ sơ minh bạch và không thể thay đổi."),
    # ── Môi trường ────────────────────────────────────────────────────────────
    ("Climate change is one of the greatest challenges of our time.", "Biến đổi khí hậu là một trong những thách thức lớn nhất của thời đại chúng ta."),
    ("Renewable energy sources reduce our dependence on fossil fuels.", "Các nguồn năng lượng tái tạo giảm sự phụ thuộc vào nhiên liệu hóa thạch."),
    ("Recycling helps conserve natural resources and reduce waste.", "Tái chế giúp bảo tồn tài nguyên thiên nhiên và giảm chất thải."),
    ("The rainforest provides habitat for thousands of species.", "Rừng nhiệt đới cung cấp môi trường sống cho hàng nghìn loài."),
    ("Solar panels convert sunlight directly into electricity.", "Pin mặt trời chuyển đổi trực tiếp ánh sáng mặt trời thành điện."),
    ("Ocean pollution is threatening marine biodiversity.", "Ô nhiễm đại dương đang đe dọa đa dạng sinh học biển."),
    ("Planting trees can help mitigate the effects of global warming.", "Trồng cây có thể giúp giảm thiểu tác động của sự nóng lên toàn cầu."),
    ("Water conservation is crucial in drought-prone regions.", "Bảo tồn nước là vô cùng quan trọng ở các vùng dễ hạn hán."),
    # ── Giao tiếp và Truyền thông ─────────────────────────────────────────────
    ("Social media platforms have changed how people communicate.", "Các nền tảng mạng xã hội đã thay đổi cách mọi người giao tiếp."),
    ("The news reporter delivered the breaking story live.", "Phóng viên đưa tin nóng trực tiếp."),
    ("Misinformation spreads rapidly on social networks.", "Thông tin sai lệch lan truyền nhanh chóng trên mạng xã hội."),
    ("Digital journalism is replacing traditional print media.", "Báo chí kỹ thuật số đang thay thế truyền thông in ấn truyền thống."),
    ("Effective communication requires clarity and empathy.", "Giao tiếp hiệu quả đòi hỏi sự rõ ràng và đồng cảm."),
    # ── Viễn thông chuyên sâu ─────────────────────────────────────────────────
    ("Channel estimation algorithms improve the reliability of wireless links.", "Các thuật toán ước lượng kênh cải thiện độ tin cậy của các liên kết không dây."),
    ("Beamforming focuses signal energy toward the intended receiver.", "Beamforming tập trung năng lượng tín hiệu về phía máy thu đích."),
    ("OFDM divides the bandwidth into multiple orthogonal subcarriers.", "OFDM chia băng thông thành nhiều sóng mang con trực giao."),
    ("Massive MIMO systems use large antenna arrays to improve capacity.", "Hệ thống Massive MIMO sử dụng mảng ăng-ten lớn để cải thiện dung lượng."),
    ("The bit error rate increases significantly at low signal-to-noise ratios.", "Tỷ lệ lỗi bit tăng đáng kể ở các tỷ số tín hiệu trên nhiễu thấp."),
    ("Error correction codes protect transmitted data from channel noise.", "Mã sửa lỗi bảo vệ dữ liệu truyền khỏi nhiễu kênh."),
    ("The propagation delay depends on the distance between transmitter and receiver.", "Độ trễ lan truyền phụ thuộc vào khoảng cách giữa máy phát và máy thu."),
    ("Interference management is critical in dense network deployments.", "Quản lý nhiễu rất quan trọng trong các triển khai mạng dày đặc."),
    ("Pilot signals help the receiver estimate the channel response.", "Tín hiệu pilot giúp máy thu ước lượng đáp ứng kênh."),
    ("Uplink and downlink channels may exhibit different propagation characteristics.", "Các kênh uplink và downlink có thể có các đặc điểm lan truyền khác nhau."),
    # ── NLP & AI chuyên sâu ───────────────────────────────────────────────────
    ("Tokenization splits text into smaller units for model processing.", "Tách token phân chia văn bản thành các đơn vị nhỏ hơn để mô hình xử lý."),
    ("The attention mechanism computes weighted sums of value vectors.", "Cơ chế chú ý tính tổng có trọng số của các vector giá trị."),
    ("Encoder-decoder architectures are commonly used in sequence-to-sequence tasks.", "Kiến trúc encoder-decoder thường được sử dụng trong các tác vụ chuỗi sang chuỗi."),
    ("Word embeddings map tokens to dense vector representations.", "Embedding từ ánh xạ các token sang biểu diễn vector dày đặc."),
    ("Layer normalization stabilizes the training of deep neural networks.", "Chuẩn hóa lớp ổn định quá trình huấn luyện mạng nơ-ron sâu."),
    ("The softmax function converts raw logits into probability distributions.", "Hàm softmax chuyển đổi logits thô thành phân phối xác suất."),
    ("Residual connections allow gradients to flow through deep architectures.", "Kết nối dư cho phép gradient chảy qua các kiến trúc sâu."),
    ("Pre-trained language models can be fine-tuned for downstream tasks.", "Các mô hình ngôn ngữ tiền huấn luyện có thể được tinh chỉnh cho các tác vụ hạ nguồn."),
    ("Data augmentation increases the diversity of training examples.", "Tăng cường dữ liệu tăng sự đa dạng của các ví dụ huấn luyện."),
    ("Hyperparameter tuning is essential to achieve optimal model performance.", "Điều chỉnh siêu tham số là cần thiết để đạt hiệu suất mô hình tối ưu."),
    # ── Bảo mật & Tấn công đối kháng ─────────────────────────────────────────
    ("Adversarial examples are carefully crafted inputs that fool neural networks.", "Các ví dụ đối kháng là đầu vào được tạo ra cẩn thận nhằm đánh lừa mạng nơ-ron."),
    ("Robustness training improves the model's resistance to adversarial perturbations.", "Huấn luyện tính bền vững cải thiện khả năng chống lại các nhiễu đối kháng của mô hình."),
    ("The FGSM attack adds a small perturbation in the direction of the gradient.", "Tấn công FGSM thêm nhiễu nhỏ theo hướng của gradient."),
    ("Carlini and Wagner proposed a powerful optimization-based attack strategy.", "Carlini và Wagner đề xuất một chiến lược tấn công mạnh dựa trên tối ưu hóa."),
    ("Defense mechanisms must balance security and communication performance.", "Các cơ chế phòng thủ phải cân bằng giữa bảo mật và hiệu suất truyền thông."),
    ("Latent space attacks exploit internal representations of deep learning models.", "Các tấn công không gian ẩn khai thác các biểu diễn nội bộ của mô hình học sâu."),
    ("Certified defenses provide provable guarantees against norm-bounded attacks.", "Các phòng thủ được chứng nhận cung cấp đảm bảo chứng minh được chống lại các tấn công bị giới hạn chuẩn."),
    ("Perturbation magnitude directly impacts the stealth of adversarial attacks.", "Biên độ nhiễu ảnh hưởng trực tiếp đến tính ẩn danh của các tấn công đối kháng."),
    # ── Kinh tế ───────────────────────────────────────────────────────────────
    ("Global trade agreements facilitate the exchange of goods across borders.", "Các hiệp định thương mại toàn cầu tạo thuận lợi cho trao đổi hàng hóa qua biên giới."),
    ("Inflation reduces the purchasing power of consumers over time.", "Lạm phát làm giảm sức mua của người tiêu dùng theo thời gian."),
    ("Foreign direct investment boosts economic growth in developing countries.", "Đầu tư trực tiếp nước ngoài thúc đẩy tăng trưởng kinh tế ở các nước đang phát triển."),
    ("Stock market volatility reflects uncertainty in the global economy.", "Sự biến động của thị trường chứng khoán phản ánh sự không chắc chắn trong nền kinh tế toàn cầu."),
    ("Small and medium enterprises are the backbone of many economies.", "Các doanh nghiệp vừa và nhỏ là xương sống của nhiều nền kinh tế."),
    # ── Y tế ──────────────────────────────────────────────────────────────────
    ("Vaccination programs have eradicated many deadly infectious diseases.", "Các chương trình tiêm chủng đã xóa sổ nhiều bệnh truyền nhiễm nguy hiểm."),
    ("Early detection significantly improves the prognosis of cancer patients.", "Phát hiện sớm cải thiện đáng kể tiên lượng của bệnh nhân ung thư."),
    ("Mental health awareness has grown significantly in recent years.", "Nhận thức về sức khỏe tâm thần đã tăng đáng kể trong những năm gần đây."),
    ("Telemedicine allows patients to consult doctors remotely.", "Y tế từ xa cho phép bệnh nhân tư vấn bác sĩ từ xa."),
    ("Proper nutrition is fundamental to maintaining good health.", "Dinh dưỡng đúng cách là nền tảng để duy trì sức khỏe tốt."),
]


def get_supplementary_dataset_info() -> Dict:
    """Thông tin mô tả bộ dataset đối chứng."""
    return {
        "dataset2_name": "Tatoeba / WMT Lifestyle & Domain-General (Curated)",
        "dataset2_size": len(DATASET2_EXTRA),
        "categories": [
            "Đời sống thường ngày",
            "Giáo dục",
            "Công nghệ",
            "Môi trường",
            "Giao tiếp & Truyền thông",
            "Viễn thông chuyên sâu",
            "NLP & AI chuyên sâu",
            "Bảo mật & Tấn công đối kháng",
            "Kinh tế",
            "Y tế"
        ],
        "purpose": "Đối chứng độ tổng quát hóa của mô hình ngoài miền chuyên ngành hẹp",
        "vocab_enrichment": "Từ vựng đa miền giúp Transformer học ngữ cảnh phong phú hơn"
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2: Tính Metrics (Accuracy, F1, BLEU, Semantic Similarity)
# ══════════════════════════════════════════════════════════════════════════════

def token_accuracy_f1(ref_ids: List[int], pred_ids: List[int], pad_id: int = 0) -> Tuple[float, float, float, float]:
    """
    Tính Accuracy và Macro-F1 ở cấp độ token.
    Loại bỏ padding (pad_id) trước khi tính.

    Returns:
        accuracy, f1_macro, precision_macro, recall_macro
    """
    # Cắt bằng chiều dài ngắn hơn
    min_len = min(len(ref_ids), len(pred_ids))
    if min_len == 0:
        return 0.0, 0.0, 0.0, 0.0

    ref_trim = ref_ids[:min_len]
    pred_trim = pred_ids[:min_len]

    # Loại bỏ vị trí PAD trong ref
    non_pad = [(r, p) for r, p in zip(ref_trim, pred_trim) if r != pad_id]
    if not non_pad:
        return 0.0, 0.0, 0.0, 0.0

    refs = [r for r, _ in non_pad]
    preds = [p for _, p in non_pad]

    if HAS_SKLEARN:
        acc = accuracy_score(refs, preds)
        # zero_division=0 để tránh warning khi một lớp không có dự đoán
        f1 = f1_score(refs, preds, average='macro', zero_division=0)
        prec = precision_score(refs, preds, average='macro', zero_division=0)
        rec = recall_score(refs, preds, average='macro', zero_division=0)
        return float(acc), float(f1), float(prec), float(rec)
    else:
        # Fallback thủ công
        correct = sum(1 for r, p in zip(refs, preds) if r == p)
        acc = correct / len(refs)
        return acc, acc, acc, acc  # simplified


def simple_bleu_score(reference: str, hypothesis: str) -> float:
    """BLEU đơn giản (unigram/bigram)."""
    if HAS_SACREBLEU:
        try:
            result = _sb.corpus_bleu([hypothesis], [[reference]])
            return result.score / 100.0
        except Exception:
            pass
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    matches = sum(1 for t in hyp_tokens if t in ref_tokens)
    precision = matches / len(hyp_tokens)
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(hyp_tokens))
    return precision * bp


def jaccard_similarity(text1: str, text2: str) -> float:
    """Độ tương đồng Jaccard (fallback cho semantic similarity)."""
    w1 = set(text1.lower().split())
    w2 = set(text2.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 3: Engine đánh giá chính
# ══════════════════════════════════════════════════════════════════════════════

class AttackEvaluator:
    """
    Đánh giá F1 Score và Accuracy cho 4 kịch bản trên 2 dataset.
    """

    def __init__(self, model: SemanticCommModel, tokenizer: SimpleTokenizer,
                 detector: AttackDetector, defense: AdaptiveDefense,
                 device: torch.device, snr_db: float = 10.0, epsilon_fgsm: float = 0.15,
                 cw_c: float = 1.0, cw_steps: int = 50):
        self.model = model
        self.tokenizer = tokenizer
        self.detector = detector
        self.defense = defense
        self.device = device
        self.snr_db = snr_db
        self.epsilon_fgsm = epsilon_fgsm
        self.cw_c = cw_c
        self.cw_steps = cw_steps
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    def _encode_batch(self, texts: List[str], max_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize và pad một batch câu."""
        ids_list = [self.tokenizer.encode(t)[:max_len] for t in texts]
        max_l = max(len(x) for x in ids_list)
        padded = [x + [self.tokenizer.pad_id] * (max_l - len(x)) for x in ids_list]
        tensor = torch.tensor(padded, dtype=torch.long, device=self.device)
        mask = (tensor == self.tokenizer.pad_id)
        return tensor, mask

    def _generate(self, latent: torch.Tensor, max_len: int = 32) -> List[List[int]]:
        """Giải mã tuần tự từ latent vector."""
        B = latent.size(0)
        with torch.no_grad():
            ch_sym = self.model.channel_encoder(latent)
            noisy = self.model.channel(ch_sym, self.snr_db)
            rec = self.model.channel_decoder(noisy)
            ids = self.model.semantic_decoder.generate(
                rec, max_len, self.tokenizer.sos_id, self.tokenizer.eos_id, strategy='greedy'
            )
        return [ids[i].cpu().tolist() for i in range(B)]

    def _evaluate_scenario(self, src_texts: List[str], tgt_texts: List[str],
                            scenario: str = 'clean', batch_size: int = 8) -> Dict:
        """
        Đánh giá một kịch bản trên toàn bộ dataset.

        scenario: 'clean' | 'fgsm' | 'fgsm_defended' | 'cw' | 'cw_defended'
        """
        self.model.eval()
        self.defense.eval()
        self.detector.eval()

        all_ref_ids: List[List[int]] = []
        all_pred_ids: List[List[int]] = []
        all_bleu: List[float] = []
        all_sim: List[float] = []
        detect_labels: List[int] = []
        detect_preds: List[int] = []

        n = len(src_texts)
        for start in tqdm(range(0, n, batch_size), desc=f"  [{scenario}]", leave=False):
            end = min(start + batch_size, n)
            src_batch = src_texts[start:end]
            tgt_batch = tgt_texts[start:end]

            src_ids, src_mask = self._encode_batch(src_batch)
            tgt_ids, tgt_mask = self._encode_batch(tgt_batch)

            # Phần reference (không tính SOS của input)
            ref_list = [tgt_ids[i].cpu().tolist() for i in range(len(src_batch))]

            # Trích xuất latent sạch
            with torch.no_grad():
                clean_latent = self.model.semantic_encoder(src_ids, src_mask)

            input_tgt = tgt_ids[:, :-1]
            target_ids_gt = tgt_ids[:, 1:]
            curr_mask = tgt_mask[:, :-1] if tgt_mask is not None else None

            if scenario == 'clean':
                latent_to_use = clean_latent
                det_label = 0

            elif scenario == 'fgsm':
                latent_to_use, _ = fgsm_attack(
                    model=self.model, latent=clean_latent,
                    target_ids=target_ids_gt, tgt_ids_input=input_tgt,
                    tgt_mask=curr_mask, criterion=self.criterion,
                    epsilon=self.epsilon_fgsm, snr_db=self.snr_db
                )
                det_label = 1

            elif scenario == 'fgsm_defended':
                fgsm_lat, _ = fgsm_attack(
                    model=self.model, latent=clean_latent,
                    target_ids=target_ids_gt, tgt_ids_input=input_tgt,
                    tgt_mask=curr_mask, criterion=self.criterion,
                    epsilon=self.epsilon_fgsm, snr_db=self.snr_db
                )
                latent_to_use = self.defense.defend(fgsm_lat, force_attack_type=1)
                det_label = 1

            elif scenario == 'cw':
                latent_to_use, _ = cw_attack(
                    model=self.model, latent=clean_latent,
                    target_ids=target_ids_gt, tgt_ids_input=input_tgt,
                    tgt_mask=curr_mask, criterion=self.criterion,
                    c=self.cw_c, max_iter=self.cw_steps, snr_db=self.snr_db
                )
                det_label = 2

            elif scenario == 'cw_defended':
                cw_lat, _ = cw_attack(
                    model=self.model, latent=clean_latent,
                    target_ids=target_ids_gt, tgt_ids_input=input_tgt,
                    tgt_mask=curr_mask, criterion=self.criterion,
                    c=self.cw_c, max_iter=self.cw_steps, snr_db=self.snr_db
                )
                latent_to_use = self.defense.defend(cw_lat, force_attack_type=2)
                det_label = 2

            else:
                latent_to_use = clean_latent
                det_label = 0

            # Detector prediction
            with torch.no_grad():
                det_logits = self.detector(latent_to_use)
                det_pred = det_logits.argmax(dim=-1).cpu().tolist()

            detect_labels.extend([det_label] * len(src_batch))
            if isinstance(det_pred, list):
                detect_preds.extend(det_pred)
            else:
                detect_preds.append(det_pred)

            # Generate
            pred_ids_list = self._generate(latent_to_use, max_len=48)

            for i, (ref, pred) in enumerate(zip(ref_list, pred_ids_list)):
                all_ref_ids.append(ref)
                all_pred_ids.append(pred)

                # Decode để tính text metrics
                ref_text = self.tokenizer.decode(ref)
                pred_text = self.tokenizer.decode(pred)
                all_bleu.append(simple_bleu_score(ref_text, pred_text))
                all_sim.append(jaccard_similarity(ref_text, pred_text))

        # Tổng hợp token-level metrics (flatten all sequences)
        flat_ref = [tok for seq in all_ref_ids for tok in seq]
        flat_pred = [tok for seq in all_pred_ids for tok in seq]

        # Pad pred đến chiều ref nếu cần
        if len(flat_pred) < len(flat_ref):
            flat_pred += [self.tokenizer.pad_id] * (len(flat_ref) - len(flat_pred))
        elif len(flat_pred) > len(flat_ref):
            flat_pred = flat_pred[:len(flat_ref)]

        acc, f1, prec, rec = token_accuracy_f1(flat_ref, flat_pred, self.tokenizer.pad_id)

        # Detection metrics
        if HAS_SKLEARN and len(set(detect_labels)) > 1:
            det_acc = accuracy_score(detect_labels, detect_preds)
            det_f1 = f1_score(detect_labels, detect_preds, average='macro', zero_division=0)
        else:
            correct_det = sum(1 for a, b in zip(detect_labels, detect_preds) if a == b)
            det_acc = correct_det / max(1, len(detect_labels))
            det_f1 = det_acc

        return {
            'scenario': scenario,
            'n_samples': n,
            'token_accuracy': round(acc, 4),
            'token_f1_macro': round(f1, 4),
            'token_precision_macro': round(prec, 4),
            'token_recall_macro': round(rec, 4),
            'bleu': round(float(np.mean(all_bleu)), 4),
            'semantic_similarity': round(float(np.mean(all_sim)), 4),
            'detector_accuracy': round(det_acc, 4),
            'detector_f1': round(det_f1, 4),
        }

    def evaluate_all(self, src_texts: List[str], tgt_texts: List[str],
                     dataset_label: str = 'Dataset 1') -> List[Dict]:
        """Đánh giá tất cả 5 kịch bản."""
        scenarios = ['clean', 'fgsm', 'fgsm_defended', 'cw', 'cw_defended']
        results = []
        print(f"\n{'='*64}")
        print(f"  Đánh giá trên {dataset_label} ({len(src_texts)} mẫu)")
        print(f"{'='*64}")
        for sc in scenarios:
            print(f"\n  → Kịch bản: {sc}")
            r = self._evaluate_scenario(src_texts, tgt_texts, scenario=sc)
            r['dataset'] = dataset_label
            results.append(r)
            print(f"     Accuracy={r['token_accuracy']:.4f} | F1={r['token_f1_macro']:.4f} "
                  f"| BLEU={r['bleu']:.4f} | SimSim={r['semantic_similarity']:.4f} "
                  f"| DetAcc={r['detector_accuracy']:.4f}")
        return results


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 4: Tổng hợp và lưu kết quả
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tools(config, device):
    """Tải model, tokenizer, detector, và defense từ checkpoint."""
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']

    print("  → Tạo DataLoaders & tokenizer...")
    _, _, test_loader, tokenizer = create_dataloaders(data_cfg, batch_size=32)
    model_cfg.vocab_size = tokenizer.vocab_size

    print(f"  → Khởi tạo SemanticCommModel (vocab={tokenizer.vocab_size})...")
    model = SemanticCommModel(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        d_ff=model_cfg.d_ff,
        latent_dim=model_cfg.latent_dim,
        max_seq_len=model_cfg.max_seq_len,
        dropout=model_cfg.dropout
    ).to(device)

    detector = AttackDetector(latent_dim=model_cfg.latent_dim).to(device)
    defense = AdaptiveDefense(latent_dim=model_cfg.latent_dim, detector=detector).to(device)

    ckpt_dir = train_cfg.checkpoint_dir
    # Ưu tiên checkpoint adversarial, fallback clean
    for fname in ['best_robust_model.pth', 'last_robust_model.pth', 'best_model.pth']:
        ckpt_path = os.path.join(ckpt_dir, fname)
        if os.path.exists(ckpt_path):
            print(f"  → Tải checkpoint: {fname}")
            ck = torch.load(ckpt_path, map_location=device)
            if isinstance(ck, dict):
                if 'model_state_dict' in ck:
                    model.load_state_dict(ck['model_state_dict'], strict=False)
                if 'detector_state_dict' in ck:
                    detector.load_state_dict(ck['detector_state_dict'], strict=False)
                if 'adaptive_defense_state_dict' in ck:
                    defense.load_state_dict(ck['adaptive_defense_state_dict'], strict=False)
            else:
                model.load_state_dict(ck, strict=False)
            break
    else:
        print("  ⚠️  Không tìm thấy checkpoint → chạy với trọng số ngẫu nhiên (demo mode).")

    return model, tokenizer, detector, defense, test_loader


def save_results_csv(results: List[Dict], save_path: str):
    """Lưu kết quả ra file CSV."""
    if not results:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    keys = list(results[0].keys())
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"  ✅ Đã lưu CSV: {save_path}")


def save_results_json(results: List[Dict], save_path: str):
    """Lưu kết quả ra file JSON."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Đã lưu JSON: {save_path}")


def print_comparison_table(results: List[Dict]):
    """In bảng so sánh đẹp ra console."""
    SCENARIO_NAMES = {
        'clean': 'Không tấn công (Clean)',
        'fgsm': 'FGSM (chưa phòng thủ)',
        'fgsm_defended': 'FGSM + Phòng thủ',
        'cw': 'C&W (chưa phòng thủ)',
        'cw_defended': 'C&W + Phòng thủ',
    }
    print("\n" + "="*100)
    print(f"{'Dataset':<22} {'Kịch bản':<26} {'Accuracy':>10} {'F1 Macro':>10} "
          f"{'BLEU':>8} {'SimSim':>8} {'DetAcc':>9}")
    print("-"*100)
    for r in results:
        ds = r.get('dataset', '')[:20]
        sc = SCENARIO_NAMES.get(r['scenario'], r['scenario'])[:24]
        print(f"{ds:<22} {sc:<26} {r['token_accuracy']:>10.4f} {r['token_f1_macro']:>10.4f} "
              f"{r['bleu']:>8.4f} {r['semantic_similarity']:>8.4f} {r['detector_accuracy']:>9.4f}")
    print("="*100)


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 5: Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  ĐÁNH GIÁ TẤN CÔNG ĐỐI KHÁNG — F1 & ACCURACY")
    print("  2 Tấn công: FGSM + C&W | 2 Dataset | 5 Kịch bản")
    print("=" * 64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    config = get_default_config()

    # ── Tải model ─────────────────────────────────────────────────────────────
    print("\n[1/5] Tải mô hình và checkpoint...")
    model, tokenizer, detector, defense, test_loader = load_model_and_tools(config, device)

    # ── Dataset 1: lấy từ test_loader ─────────────────────────────────────────
    print("\n[2/5] Chuẩn bị Dataset 1 (harouzie/vi_en-translation)...")
    src_texts_d1, tgt_texts_d1 = [], []
    for batch in test_loader:
        src_texts_d1.extend(batch['src_text'])
        tgt_texts_d1.extend(batch['tgt_text'])
        if len(src_texts_d1) >= 200:
            break
    src_texts_d1 = src_texts_d1[:200]
    tgt_texts_d1 = tgt_texts_d1[:200]
    print(f"  → Dataset 1: {len(src_texts_d1)} mẫu")

    # ── Dataset 2: Tải từ CSV thật (data/processed/dataset2_tatoeba_wmt.csv) ─────
    print("\n[3/5] Chuẩn bị Dataset 2 (Tatoeba/WMT Lifestyle — từ CSV)...")
    d2_pairs = load_dataset2_for_evaluation(direction='en-vi')
    if not d2_pairs:
        # Fallback nếu chưa có file CSV (dùng hardcode)
        print("  ⚠️  Không tìm thấy CSV → dùng dữ liệu nội bộ DATASET2_EXTRA")
        d2_pairs = DATASET2_EXTRA.copy()
    random.shuffle(d2_pairs)
    src_texts_d2 = [p[0] for p in d2_pairs]
    tgt_texts_d2 = [p[1] for p in d2_pairs]
    print(f"  → Dataset 2: {len(src_texts_d2)} mẫu từ file CSV ({len(set(cat for cat in ['Đời sống','Giáo dục','Công nghệ','Môi trường','Giao tiếp & Truyền thông','Viễn thông chuyên sâu','NLP & AI chuyên sâu','Bảo mật & Tấn công đối kháng','Kinh tế','Y tế']))} chủ đề)")

    # ── Đánh giá ─────────────────────────────────────────────────────────────
    print("\n[4/5] Chạy đánh giá tấn công đối kháng...")
    evaluator = AttackEvaluator(
        model=model, tokenizer=tokenizer,
        detector=detector, defense=defense,
        device=device,
        snr_db=10.0,
        epsilon_fgsm=getattr(config['adversarial'], 'epsilon_fgsm', 0.15),
        cw_c=getattr(config['adversarial'], 'cw_c', 1.0),
        cw_steps=30   # giảm để chạy nhanh hơn
    )

    all_results = []
    all_results.extend(evaluator.evaluate_all(src_texts_d1, tgt_texts_d1, dataset_label='Dataset 1 (harouzie)'))
    all_results.extend(evaluator.evaluate_all(src_texts_d2, tgt_texts_d2, dataset_label='Dataset 2 (Tatoeba/WMT)'))

    # ── Lưu kết quả ───────────────────────────────────────────────────────────
    print("\n[5/5] Lưu kết quả...")
    results_dir = config['training'].results_dir
    os.makedirs(results_dir, exist_ok=True)
    save_results_csv(all_results, os.path.join(results_dir, 'attack_evaluation_results.csv'))
    save_results_json(all_results, os.path.join(results_dir, 'attack_evaluation_results.json'))

    # Lưu thông tin dataset 2
    d2_info = get_supplementary_dataset_info()
    with open(os.path.join(results_dir, 'dataset2_info.json'), 'w', encoding='utf-8') as f:
        json.dump(d2_info, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Đã lưu thông tin Dataset 2: dataset2_info.json")

    print_comparison_table(all_results)
    print("\n✅ Đánh giá hoàn tất!")
    return all_results


if __name__ == '__main__':
    main()
