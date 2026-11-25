"""
Bộ kiểm thử tự động (Unit Tests & Integration Tests)
cho dự án: "Nghiên cứu và triển khai Semantic Communication cho truyền văn bản sử dụng mô hình Transformer".
"""

import os
import sys
import hashlib
import py_compile
import unittest
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.dataset import SimpleTokenizer, _deduplicate_train_data, _count_effective_tokens
from models.semantic_encoder import SemanticEncoder
from models.semantic_decoder import SemanticDecoder
from models.semantic_model import SemanticCommModel
from models.attack_detector import AttackDetector
from models.adaptive_defense import AdaptiveDefense
from channel.awgn import AWGNChannel
from channel.rayleigh import RayleighChannel
from channel.traditional import TraditionalPipeline
from attacks.fgsm import fgsm_attack
from attacks.cw import cw_attack
from evaluation.metrics import (
    is_garbage_mojibake,
    calculate_bleu,
    calculate_bertscore,
    calculate_semantic_similarity,
)
from evaluation.evaluate_attacks import token_accuracy_f1, align_sentence_pairs_to_tokens
from evaluation.run_adversarial_benchmark import calc_token_metrics


class TestSemanticCommunicationProject(unittest.TestCase):
    """Kiểm thử toàn diện các thành phần lõi của hệ thống Semantic Communication DeepSC."""

    @classmethod
    def setUpClass(cls):
        cls.vocab_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'vocab.json')
        cls.best_model_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'best_model.pth')
        cls.robust_model_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'best_robust_model.pth')
        cls.tokenizer = SimpleTokenizer(vocab_file=cls.vocab_path)

    def test_01_syntax_all_python_files(self):
        """Kiểm tra cú pháp (py_compile) toàn bộ các file Python trong dự án."""
        checked = 0
        for root, _, files in os.walk(PROJECT_ROOT):
            if any(skip in root for skip in ('.git', '__pycache__', 'backup_')):
                continue
            for f in files:
                if f.endswith('.py'):
                    full_path = os.path.join(root, f)
                    py_compile.compile(full_path, doraise=True)
                    checked += 1
        self.assertGreaterEqual(checked, 15, "Số lượng file Python được kiểm tra phải >= 15")

    def test_02_vocab_checkpoint_sha256_alignment(self):
        """Kiểm tra vocab.json khớp SHA-256 của checkpoint, decode dừng tại [EOS] và khử trùng lặp 2 chiều."""
        self.assertTrue(os.path.exists(self.vocab_path))
        with open(self.vocab_path, 'rb') as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        best_ck = torch.load(self.best_model_path, map_location='cpu')
        robust_ck = torch.load(self.robust_model_path, map_location='cpu')
        expected_sha = best_ck.get('vocab_sha256')
        robust_sha = robust_ck.get('vocab_sha256') or robust_ck.get('run_metadata', {}).get('vocab_sha256')
        self.assertIsNotNone(expected_sha, "best_model.pth phải lưu vocab_sha256")
        self.assertEqual(sha, expected_sha, "checkpoints/vocab.json phải khớp 100% SHA-256 của best_model.pth")
        if robust_sha is not None:
            self.assertEqual(sha, robust_sha, "checkpoints/vocab.json phải khớp 100% SHA-256 của best_robust_model.pth")
        self.assertEqual(self.tokenizer.vocab_size, 10000)

        # Kiểm tra dừng tại [EOS] đầu tiên trong decode()
        ids_with_trailing = [self.tokenizer.sos_id, 10, 224, 59, self.tokenizer.eos_id, 16, 10, self.tokenizer.eos_id]
        decoded = self.tokenizer.decode(ids_with_trailing)
        self.assertEqual(len(decoded.split()), 3, f"Decode phải dừng ngay tại [EOS] đầu tiên, nhận được: '{decoded}'")

        # Kiểm tra chuẩn hóa Unicode NFC và tách token tiếng Trung
        nfd_str = "Tiê\u0301ng Viê\u0323t"
        nfc_str = "Tiếng Việt"
        self.assertEqual(self.tokenizer._tokenize(nfd_str), self.tokenizer._tokenize(nfc_str))
        self.assertEqual(_count_effective_tokens("我爱你"), 3)

        # Kiểm tra khử trùng lặp cả hai chiều src và tgt
        train_dummy = [("Hello world", "Xin chào thế giới"), ("Different src", "Xin chào thế giới"), ("Unique", "Độc nhất")]
        val_dummy = [("Other", "Xin chào thế giới")]
        cleaned = _deduplicate_train_data(train_dummy, val_dummy)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0][0], "Unique")

    def test_03_transformer_encoder_decoder_invariance(self):
        """Kiểm tra Masked Mean Pooling, mở rộng động PositionalEncoding > 64, bool mask, greedy & beam search."""
        enc = SemanticEncoder(vocab_size=100, d_model=32, n_heads=4, n_layers=2, d_ff=64, latent_dim=16, max_seq_len=64, dropout=0.0)
        enc.eval()

        # 1. Bất biến với padding: câu 5 token vs câu đệm thêm 15 token [PAD]
        ids_short = torch.tensor([[1, 12, 34, 56, 2]], dtype=torch.long)
        mask_short = (ids_short == 0)
        ids_padded = torch.tensor([[1, 12, 34, 56, 2] + [0] * 15], dtype=torch.long)
        mask_padded = (ids_padded == 0)

        with torch.no_grad():
            lat_short = enc(ids_short, mask_short)
            lat_padded = enc(ids_padded, mask_padded)
        l2_diff = float(torch.norm(lat_short - lat_padded).item())
        self.assertLess(l2_diff, 1e-5, f"Masked Mean Pooling phải bất biến với [PAD] (L2 diff={l2_diff})")

        # 2. Chuỗi dài > 64 token (96 token) không lỗi kích thước PositionalEncoding
        long_ids = torch.randint(4, 90, (2, 96), dtype=torch.long)
        with torch.no_grad():
            lat_long = enc(long_ids, (long_ids == 0))
        self.assertEqual(lat_long.shape, (2, 16))
        self.assertTrue(torch.isfinite(lat_long).all())

        # 3. Hàng toàn [PAD] không sinh NaN
        all_pad_ids = torch.zeros((2, 8), dtype=torch.long)
        with torch.no_grad():
            lat_all_pad = enc(all_pad_ids, (all_pad_ids == 0))
        self.assertTrue(torch.isfinite(lat_all_pad).all(), "Hàng toàn [PAD] không được sinh NaN")

        # 4. Decoder generate hỗ trợ sos_token_id, sos_id và chiến lược beam_search
        dec = SemanticDecoder(vocab_size=100, d_model=32, n_heads=4, n_layers=2, d_ff=64, latent_dim=16, max_seq_len=64, dropout=0.0)
        dec.eval()
        with torch.no_grad():
            out_greedy = dec.generate(lat_short, max_len=12, sos_id=1, eos_id=2, strategy='greedy')
            out_beam = dec.generate(lat_short, max_len=12, sos_token_id=1, eos_token_id=2, strategy='beam_search', beam_width=3)
        self.assertEqual(out_greedy.size(0), 1)
        self.assertEqual(out_beam.size(0), 1)

    def test_04_channel_awgn_rayleigh_mmse_and_traditional(self):
        """Kiểm tra kênh AWGN, Rayleigh MMSE, set_channel() và TraditionalPipeline BER."""
        awgn = AWGNChannel()
        ray_mmse = RayleighChannel(with_csi_equalization=True, equalization_mode='mmse')
        ray_zf = RayleighChannel(with_csi_equalization=True, equalization_mode='zf')

        torch.manual_seed(42)
        x = torch.randn(64, 128)
        x = x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))

        # Kiểm tra MSE sau cân bằng MMSE tại SNR=20 dB ổn định hơn Zero-Forcing
        y_mmse = ray_mmse(x, 20.0)
        y_zf = ray_zf(x, 20.0)
        mse_mmse = float(torch.mean((y_mmse - x) ** 2).item())
        mse_zf = float(torch.mean((y_zf - x) ** 2).item())
        self.assertLess(mse_mmse, 0.05, f"Rayleigh MMSE ở 20 dB phải có MSE < 0.05, thực đo: {mse_mmse:.4f} (ZF={mse_zf:.4f})")

        # Kiểm tra SNR không hợp lệ (NaN / Inf) ném ValueError
        with self.assertRaises(ValueError):
            awgn(x, float('nan'))
        with self.assertRaises(ValueError):
            ray_mmse(x, float('inf'))

        # Kiểm tra set_channel() trên SemanticCommModel
        model = SemanticCommModel(vocab_size=100, d_model=32, n_heads=4, n_layers=1, d_ff=64, latent_dim=16)
        model.set_channel('rayleigh')
        self.assertIs(model.channel, model.rayleigh_channel)
        with self.assertRaises(ValueError):
            model.set_channel('invalid_channel_xyz')

        # Kiểm tra TraditionalPipeline dùng RNG cục bộ, không ảnh hưởng global np.random.seed
        trad = TraditionalPipeline()
        np.random.seed(999)
        expected_first = np.random.rand()
        np.random.seed(999)
        _, ber_empty = trad.transmit("", snr_db=10.0, seed=123)
        after_first = np.random.rand()
        self.assertEqual(ber_empty, 0.0)
        self.assertAlmostEqual(expected_first, after_first, places=7, msg="TraditionalPipeline.transmit không được thay đổi global np.random.seed")

    def test_05_attacks_fgsm_cw_and_adaptive_defense(self):
        """Kiểm tra tấn công FGSM, C&W (nhiễu > 0, không rò rỉ .grad vào model) và AdaptiveDefense."""
        model = SemanticCommModel(vocab_size=self.tokenizer.vocab_size)
        detector = AttackDetector(latent_dim=128)
        defense = AdaptiveDefense(latent_dim=128, detector=detector)

        ck = torch.load(self.robust_model_path, map_location='cpu')
        model.load_state_dict(ck['model_state_dict'], strict=True)
        detector.load_state_dict(ck['detector_state_dict'], strict=False)
        defense.load_state_dict(ck['adaptive_defense_state_dict'], strict=False)
        model.eval()
        detector.eval()
        defense.eval()

        s_ids = torch.tensor([self.tokenizer.encode("The weather is very nice today.")], dtype=torch.long)
        t_ids = torch.tensor([self.tokenizer.encode("Hôm nay thời tiết rất đẹp.")], dtype=torch.long)

        # Kiểm tra gọi fgsm_attack & cw_attack trong ngữ cảnh torch.no_grad()
        with torch.no_grad():
            clean_lat = model.semantic_encoder(s_ids)
            fgsm_lat, fgsm_delta = fgsm_attack(model, clean_lat, t_ids[:, 1:], t_ids[:, :-1], epsilon=0.15)
            cw_lat, cw_delta = cw_attack(model, clean_lat, t_ids[:, 1:], t_ids[:, :-1], c=1.0, kappa=0.5, max_iter=15)

        # Trọng số model không bị ghi đè .grad
        for p in model.parameters():
            self.assertIsNone(p.grad, "Hàm tấn công không được làm thay đổi .grad của tham số mô hình")

        fgsm_l2 = float(fgsm_delta.norm().item())
        cw_l2 = float(cw_delta.norm().item())
        self.assertGreater(fgsm_l2, 0.5, f"FGSM delta L2 phải > 0.5, nhận được {fgsm_l2}")
        self.assertGreater(cw_l2, 0.1, f"C&W delta L2 trên B=1 phải > 0.1, nhận được {cw_l2}")

        # Kiểm tra CWDenoiser Gated Residual bảo toàn tín hiệu sạch
        with torch.no_grad():
            clean_after_cw_den = defense.cw_denoiser(clean_lat)
            clean_distortion = float(torch.norm(clean_after_cw_den - clean_lat).item())
        self.assertLess(clean_distortion, 5.0, f"CWDenoiser Gated Residual phải giữ sai lệch trên tín hiệu sạch < 5.0, nhận được {clean_distortion:.3f}")

    def test_06_evaluation_metrics_and_alignment(self):
        """Kiểm tra BLEU, BERTScore, Semantic Similarity, hỗ trợ tiếng Trung và căn chỉnh độ dài token."""
        # 1. Tiếng Trung không bị nhận diện nhầm là chuỗi lỗi mã hóa
        self.assertFalse(is_garbage_mojibake("我爱你"))
        self.assertEqual(calculate_bleu("我爱你", "我爱你"), 1.0)

        # 2. Cụm từ đồng nghĩa đa từ được đánh giá cao hơn câu lặp từ vô nghĩa trong BERTScore
        bs_syn = calculate_bertscore("Hệ thống truyền thông hỗ trợ bảo toàn ý nghĩa", "Mô hình truyền thông giúp duy trì nội dung")
        bs_rep = calculate_bertscore("Hệ thống truyền thông hỗ trợ bảo toàn ý nghĩa", "hệ thống hệ thống hệ thống hệ thống")
        self.assertGreater(bs_syn["F1"], 0.80, f"BERTScore trên câu đồng nghĩa phải > 0.80, nhận được {bs_syn['F1']}")
        self.assertLess(bs_rep["F1"], 0.50, f"BERTScore trên câu lặp từ vô nghĩa phải < 0.50, nhận được {bs_rep['F1']}")

        # 3. Semantic Similarity bằng SentenceTransformer phân tách rõ câu liên quan và không liên quan
        sim_related = calculate_semantic_similarity("Trí tuệ nhân tạo đang thay đổi thế giới", "Học sâu và AI đang biến đổi toàn cầu")
        sim_unrelated = calculate_semantic_similarity("Trí tuệ nhân tạo đang thay đổi thế giới", "Con mèo đang ngủ trên ghế sofa")
        self.assertGreater(sim_related, sim_unrelated + 0.25)

        # 4. Căn chỉnh từng cặp câu trước khi gộp phẳng token trong evaluate_attacks.py
        ref_seqs = [[1, 10, 20, 30, 2, 0], [1, 40, 50, 60, 70, 2]]
        pred_seqs = [[1, 10, 20, 99, 30, 2], [1, 40, 50, 60, 70, 2]]
        flat_r, flat_p = align_sentence_pairs_to_tokens(ref_seqs, pred_seqs, pad_id=0, sos_id=1, eos_id=2)
        acc, f1, prec, rec = token_accuracy_f1(flat_r, flat_p, pad_id=0)
        self.assertGreater(acc, 0.80, f"Sau khi căn chỉnh từng cặp câu, Accuracy phải > 0.80, nhận được {acc}")
        self.assertNotEqual(round(acc, 4), round(f1, 4), "Accuracy và Macro-F1 phải được tính độc lập")

        # 5. calc_token_metrics tính độc lập Accuracy và Macro-F1
        a_m, f_m, p_m, r_m = calc_token_metrics("tôi đi học ở trường đại học", "tôi đi làm ở công ty lớn")
        self.assertNotEqual(round(a_m, 2), round(f_m, 2), f"Acc ({a_m:.2f}) và Macro-F1 ({f_m:.2f}) không được trùng nhau giả tạo")


if __name__ == '__main__':
    unittest.main(verbosity=2)
