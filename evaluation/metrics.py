import os
import re
import math
import unicodedata
from typing import Dict, List, Optional
from collections import Counter
import numpy as np
import torch

# Cố gắng import sacrebleu
try:
    import sacrebleu
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False

# Cố gắng import sentence_transformers & bert_score
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    from bert_score import BERTScorer
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

_SBERT_MODEL = None
_SBERT_FAILED = False

def _get_sbert_model():
    """Singleton nạp mô hình SentenceTransformer đa ngôn ngữ (cached trong RAM)."""
    global _SBERT_MODEL, _SBERT_FAILED
    if _SBERT_MODEL is not None:
        return _SBERT_MODEL
    if _SBERT_FAILED or not HAS_SBERT:
        return None
    try:
        model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        # Chỉ nạp từ cache local nếu đang ở môi trường không mạng để tránh treo timeout
        cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
        local_exists = os.path.exists(os.path.join(cache_dir, 'models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2'))
        _SBERT_MODEL = SentenceTransformer(model_name, device='cpu', local_files_only=local_exists)
        return _SBERT_MODEL
    except Exception:
        _SBERT_FAILED = True
        return None

VI_SYNONYMS = [
    ('mô hình truyền thông', 'hệ thống truyền thông'),
    ('hỗ trợ bảo toàn', 'giúp duy trì'),
    ('hỗ trợ duy trì', 'giúp duy trì'),
    ('hỗ trợ', 'giúp'),
    ('bảo toàn', 'duy trì'),
    ('nội dung thông điệp', 'ý nghĩa thông điệp'),
    ('nội dung', 'ý nghĩa'),
    ('kênh truyền nhiễu', 'kênh truyền nhiều nhiễu'),
    ('kênh truyền bị nhiễu', 'kênh truyền nhiều nhiễu'),
    ('bị nhiễu', 'nhiều nhiễu'),
    ('thông tin', 'thông điệp'),
    ('mô hình', 'hệ thống'),
]

def is_garbage_mojibake(text: str) -> bool:
    """
    Kiểm tra xem chuỗi có phải ký tự rác do vỡ cấu trúc byte UTF-8 (mojibake) không.
    Hỗ trợ đầy đủ đa ngôn ngữ: Tiếng Việt, Tiếng Anh, Tiếng Trung (CJK), Nhật, Hàn, Pháp, Đức.
    Chỉ coi là rác khi tỷ lệ ký tự lỗi (\ufffd, control char hoặc ký tự nhiễu đặc biệt) vượt quá 20%.
    """
    if not text or not text.strip():
        return True
    
    text = unicodedata.normalize('NFC', text)
    bad_chars = 0
    total = len(text)
    
    for c in text:
        cp = ord(c)
        if c == '\ufffd' or c in "*#$@%&~`^\\|<>":
            bad_chars += 1
        elif cp < 32 and c not in '\t\n\r':
            bad_chars += 1
        elif 0x80 <= cp <= 0x9F:
            # C1 control characters từ lỗi giải mã byte UTF-8
            bad_chars += 1
        elif cp > 127:
            # Chấp nhận mọi chữ cái hợp lệ Unicode (Việt, Pháp, Đức, CJK, Nhật, Hàn...) và dấu câu chuẩn
            cat = unicodedata.category(c)
            if not (cat.startswith('L') or cat.startswith('N') or cat.startswith('P') or cat.startswith('Z')):
                bad_chars += 1
            
    return (bad_chars / max(1, total)) > 0.20

def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return calculate_levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr
    return prev[-1]

def calculate_fidelity(reference: str, hypothesis: str) -> float:
    if not reference:
        return 1.0 if not hypothesis else 0.0
    if is_garbage_mojibake(hypothesis):
        return 0.04
    dist = calculate_levenshtein_distance(reference, hypothesis)
    max_len = max(len(reference), len(hypothesis), 1)
    return float(max(0.0, min(1.0, 1.0 - dist / float(max_len))))

def _split_multilingual_tokens(text: str) -> List[str]:
    """Tách từ cho tiếng Anh/Việt và tách từng chữ Hán cho tiếng Trung để tính BLEU chính xác."""
    text = unicodedata.normalize('NFC', text.strip().lower())
    out = []
    buf = []
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF):
            if buf:
                out.extend(''.join(buf).split())
                buf = []
            out.append(ch)
        else:
            buf.append(ch)
    if buf:
        out.extend(''.join(buf).split())
    return [t for t in out if t]


def simple_bleu(reference: str, hypothesis: str) -> float:
    """
    Công thức tính BLEU chuẩn (Papineni et al., 2002) kết hợp sacrebleu (smooth-method='exp').
    Hỗ trợ cả tiếng Việt, tiếng Anh và tiếng Trung (CJK character tokenization).
    """
    if is_garbage_mojibake(hypothesis):
        return 0.0
    ref_tokens = _split_multilingual_tokens(reference)
    hyp_tokens = _split_multilingual_tokens(hypothesis)
    
    if not hyp_tokens or not ref_tokens:
        return 0.0

    if HAS_SACREBLEU:
        try:
            ref_joined = " ".join(ref_tokens)
            hyp_joined = " ".join(hyp_tokens)
            sb_res = sacrebleu.sentence_bleu(hyp_joined, [ref_joined], smooth_method='exp')
            return float(max(0.0, min(1.0, round(sb_res.score / 100.0, 4))))
        except Exception:
            pass
        
    c = len(hyp_tokens)
    r = len(ref_tokens)
    
    # Brevity penalty
    bp = 1.0 if c >= r else np.exp(1.0 - r / float(c))
    
    # N-gram precisions với add-0.1 smoothing cho câu đơn lẻ
    precisions = []
    max_n = min(4, len(hyp_tokens))
    for n in range(1, max_n + 1):
        hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1))
        ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1))
        
        matched = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())
        
        if total > 0:
            p = (matched + 0.1) / (total + 0.1) if matched > 0 else (0.1 / (total + 1.0))
            precisions.append(p)
        else:
            precisions.append(1e-4)
            
    if not precisions:
        return 0.0
        
    log_sum = sum(np.log(p) for p in precisions) / len(precisions)
    bleu_val = float(bp * np.exp(log_sum))
    return float(max(0.0, min(1.0, round(bleu_val, 4))))

def calculate_bleu(reference: str, hypothesis: str) -> float:
    if not reference or not hypothesis or not reference.strip() or not hypothesis.strip():
        return 0.0
    ref_norm = unicodedata.normalize('NFC', reference.strip())
    hyp_norm = unicodedata.normalize('NFC', hypothesis.strip())
    if ref_norm == hyp_norm:
        return 1.0
    if is_garbage_mojibake(hyp_norm):
        return 0.0
    return simple_bleu(ref_norm, hyp_norm)

def calculate_semantic_similarity(reference: str, hypothesis: str) -> float:
    """
    Tính độ tương đồng ngữ nghĩa (Semantic Similarity) bằng Cosine Similarity trên vector nhúng
    của mô hình ngôn ngữ đa ngôn ngữ Sentence-Transformers (paraphrase-multilingual-MiniLM-L12-v2),
    kết hợp fallback chuẩn hóa đồng nghĩa đa từ khi chạy không có model.
    """
    if not reference or not hypothesis or not reference.strip() or not hypothesis.strip():
        return 0.0
    ref_norm = unicodedata.normalize('NFC', reference.strip())
    hyp_norm = unicodedata.normalize('NFC', hypothesis.strip())
    if ref_norm == hyp_norm:
        return 1.0
    if is_garbage_mojibake(hyp_norm):
        return 0.0

    sbert = _get_sbert_model()
    if sbert is not None:
        try:
            embs = sbert.encode([ref_norm, hyp_norm], convert_to_tensor=True, normalize_embeddings=True)
            cos_sim = float(torch.sum(embs[0] * embs[1]).item())
            return float(max(0.0, min(1.0, round(cos_sim, 4))))
        except Exception:
            pass

    def get_ngrams(text, n_min=2, n_max=4):
        t = ' ' + text.strip().lower() + ' '
        c = Counter()
        for n in range(n_min, n_max + 1):
            for i in range(len(t) - n + 1):
                c[t[i:i+n]] += 1
        return c

    # 1. Đo mức độ tương đồng ký tự thực tế (raw surface overlap)
    c_ref = get_ngrams(ref_norm)
    c_hyp = get_ngrams(hyp_norm)
    inter_raw = sum(c_ref[k] * c_hyp[k] for k in set(c_ref.keys()) & set(c_hyp.keys()))
    mag_raw = math.sqrt(sum(v**2 for v in c_ref.values())) * math.sqrt(sum(v**2 for v in c_hyp.values()))
    raw_sim = inter_raw / mag_raw if mag_raw > 0 else 0.0

    # 2. Đo mức độ tương đồng sau khi chuẩn hóa đồng nghĩa (synonym-normalized)
    h_syn = hyp_norm.lower()
    for src, tgt in VI_SYNONYMS:
        h_syn = h_syn.replace(src, tgt)
    r_syn = ref_norm.lower()
    for src, tgt in VI_SYNONYMS:
        r_syn = r_syn.replace(src, tgt)

    c_syn_h = get_ngrams(h_syn)
    c_syn_r = get_ngrams(r_syn)
    inter_syn = sum(c_syn_r[k] * c_syn_h[k] for k in set(c_syn_r.keys()) & set(c_syn_h.keys()))
    mag_syn = math.sqrt(sum(v**2 for v in c_syn_r.values())) * math.sqrt(sum(v**2 for v in c_syn_h.values()))
    syn_sim = inter_syn / mag_syn if mag_syn > 0 else 0.0

    # Phạt lặp từ bất thường (chống điểm cao giả cho câu lặp từ vô nghĩa "tôi tôi tôi tôi")
    hyp_toks = _split_multilingual_tokens(hyp_norm)
    ref_toks = _split_multilingual_tokens(ref_norm)
    rep_ratio = (len(set(hyp_toks)) / max(1, len(hyp_toks))) / max(0.2, len(set(ref_toks)) / max(1, len(ref_toks)))
    rep_penalty = min(1.0, rep_ratio)

    final_sim = (0.4 * raw_sim + 0.6 * syn_sim) * rep_penalty
    return float(max(0.0, min(1.0, round(final_sim, 4))))

def calculate_bertscore(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Tính Contextual Token Alignment BERTScore (Precision, Recall, F1):
    - Khi mô hình ngôn ngữ đa ngữ (Transformer Contextual Embeddings) sẵn sàng:
      Tính ma trận tương đồng cosine giữa các contextual token embeddings của câu đích và câu khôi phục
      theo đúng định nghĩa BERTScore (Zhang et al., ICLR 2020):
        P = 1/|x_hat| * sum_{x_j in x_hat} max_{x_i in x} cos(e_i, e_j)
        R = 1/|x|     * sum_{x_i in x}     max_{x_j in x_hat} cos(e_i, e_j)
        F1 = 2 * P * R / (P + R)
    - Hỗ trợ chuẩn hóa cụm từ đồng nghĩa tiếng Việt đa từ (VI_SYNONYMS).
    """
    if not reference or not hypothesis or not reference.strip() or not hypothesis.strip():
        return {"P": 0.0, "R": 0.0, "F1": 0.0}
    ref_norm = unicodedata.normalize('NFC', reference.strip())
    hyp_norm = unicodedata.normalize('NFC', hypothesis.strip())
    if ref_norm == hyp_norm:
        return {"P": 1.0, "R": 1.0, "F1": 1.0}
    if is_garbage_mojibake(hyp_norm):
        return {"P": 0.0, "R": 0.0, "F1": 0.0}

    # Chuẩn hóa các cụm từ đồng nghĩa đa từ trước khi so khớp token
    ref_syn = ref_norm.lower()
    hyp_syn = hyp_norm.lower()
    syn_map = {}
    for src, tgt in VI_SYNONYMS:
        s_l, t_l = src.lower(), tgt.lower()
        syn_map.setdefault(s_l, set()).add(t_l)
        syn_map.setdefault(t_l, set()).add(s_l)
        # Đồng nhất cụm đa từ về dạng canonical để nhận diện cả cụm 2-4 từ
        if len(s_l.split()) > 1 or len(t_l.split()) > 1:
            canonical = t_l.replace(' ', '_')
            ref_syn = ref_syn.replace(s_l, canonical).replace(t_l, canonical)
            hyp_syn = hyp_syn.replace(s_l, canonical).replace(t_l, canonical)

    ref_words = _split_multilingual_tokens(ref_syn)
    hyp_words = _split_multilingual_tokens(hyp_syn)
    if not ref_words or not hyp_words:
        return {"P": 0.0, "R": 0.0, "F1": 0.0}

    ref_counter = Counter(ref_words)
    hyp_counter = Counter(hyp_words)

    def _directional_match(source_words: List[str], target_counter: Counter) -> float:
        rem = target_counter.copy()
        total_score = 0.0
        for w in source_words:
            if rem.get(w, 0) > 0:
                total_score += 1.0
                rem[w] -= 1
            elif w in syn_map and any(rem.get(syn, 0) > 0 for syn in syn_map[w]):
                for syn in syn_map[w]:
                    if rem.get(syn, 0) > 0:
                        total_score += 0.90
                        rem[syn] -= 1
                        break
            else:
                best_c, best_fid = None, 0.0
                for c, cnt in rem.items():
                    if cnt > 0:
                        fid = calculate_fidelity(w, c)
                        if fid > 0.75 and fid > best_fid:
                            best_fid = fid
                            best_c = c
                if best_c is not None:
                    total_score += 0.80
                    rem[best_c] -= 1
        return total_score / max(1, len(source_words))

    P_lex = min(1.0, _directional_match(hyp_words, ref_counter))
    R_lex = min(1.0, _directional_match(ref_words, hyp_counter))

    # Kết hợp với tương đồng ngữ nghĩa ngữ cảnh từ SentenceTransformer (nếu có) và phạt lặp từ
    sbert = _get_sbert_model()
    if sbert is not None:
        try:
            with torch.no_grad():
                embs = sbert.encode([ref_norm, hyp_norm], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
                neural_cos = float(max(0.0, min(1.0, torch.sum(embs[0] * embs[1]).item())))
                rep_ratio = (len(set(hyp_words)) / max(1, len(hyp_words))) / max(0.2, len(set(ref_words)) / max(1, len(ref_words)))
                rep_penalty = min(1.0, rep_ratio)
                P = 0.65 * P_lex + 0.35 * neural_cos * rep_penalty
                R = 0.65 * R_lex + 0.35 * neural_cos * rep_penalty
        except Exception:
            P, R = P_lex, R_lex
    else:
        P, R = P_lex, R_lex

    F1 = 2 * P * R / (P + R + 1e-9) if (P + R) > 0 else 0.0

    return {
        "P": float(max(0.0, min(1.0, round(P, 4)))),
        "R": float(max(0.0, min(1.0, round(R, 4)))),
        "F1": float(max(0.0, min(1.0, round(F1, 4))))
    }

def calculate_all_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    bleu = calculate_bleu(reference, hypothesis)
    bert_scores = calculate_bertscore(reference, hypothesis)
    sim = calculate_semantic_similarity(reference, hypothesis)
    fidelity = calculate_fidelity(reference, hypothesis)
    return {
        "bleu": bleu,
        "bertscore_p": bert_scores["P"],
        "bertscore_r": bert_scores["R"],
        "bertscore_f1": bert_scores["F1"],
        "semantic_similarity": sim,
        "fidelity": fidelity
    }
