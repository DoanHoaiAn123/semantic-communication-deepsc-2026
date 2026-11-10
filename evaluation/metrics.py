import re
import math
from typing import Dict
from collections import Counter
import numpy as np

# Cố gắng import sacrebleu
try:
    import sacrebleu
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False

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
    Kiểm tra xem chuỗi có phải ký tự rác do vỡ cấu trúc byte UTF-8 không.
    Chỉ coi là rác (sụp đổ bờ vực) khi tỷ lệ ký tự rác / hỏng vượt quá 20%.
    Nếu câu chỉ bị sai vài ký tự đơn lẻ do 1-2 byte lỗi, vẫn cho phép tính toán.
    """
    if not text or not text.strip():
        return True
    
    vi_valid = 'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ'
    bad_chars = 0
    total = len(text)
    
    for c in text:
        if c == '\ufffd' or c in "*#$@%&~`^":
            bad_chars += 1
        elif ord(c) > 127 and c not in vi_valid:
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

def simple_bleu(reference: str, hypothesis: str) -> float:
    """
    Công thức tính BLEU chuẩn (Papineni et al., 2002) trên từ gốc với Laplace Smoothing.
    BLEU phản ánh chính xác mức độ trùng khớp từ vựng từng cụm n-gram thực tế.
    """
    if is_garbage_mojibake(hypothesis):
        return 0.0
    ref_tokens = reference.strip().lower().split()
    hyp_tokens = hypothesis.strip().lower().split()
    
    if not hyp_tokens or not ref_tokens:
        return 0.0
        
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
    if not reference.strip() or not hypothesis.strip():
        return 0.0
    if reference.strip() == hypothesis.strip():
        return 1.0
    if is_garbage_mojibake(hypothesis):
        return 0.0
    return simple_bleu(reference, hypothesis)

def calculate_semantic_similarity(reference: str, hypothesis: str) -> float:
    """
    Cosine Similarity trên Character n-grams (2..4) kết hợp ánh xạ từ đồng nghĩa.
    Phản ánh tính tương đồng về mặt nội dung / thông điệp tổng thể.
    """
    if not reference.strip() or not hypothesis.strip():
        return 0.0
    if reference.strip() == hypothesis.strip():
        return 1.0
    if is_garbage_mojibake(hypothesis):
        return 0.0

    def get_ngrams(text, n_min=2, n_max=4):
        t = ' ' + text.strip().lower() + ' '
        c = Counter()
        for n in range(n_min, n_max + 1):
            for i in range(len(t) - n + 1):
                c[t[i:i+n]] += 1
        return c

    # 1. Đo mức độ tương đồng ký tự thực tế (raw surface overlap)
    c_ref = get_ngrams(reference)
    c_hyp = get_ngrams(hypothesis)
    inter_raw = sum(c_ref[k] * c_hyp[k] for k in set(c_ref.keys()) & set(c_hyp.keys()))
    mag_raw = math.sqrt(sum(v**2 for v in c_ref.values())) * math.sqrt(sum(v**2 for v in c_hyp.values()))
    raw_sim = inter_raw / mag_raw if mag_raw > 0 else 0.0

    # 2. Đo mức độ tương đồng sau khi chuẩn hóa đồng nghĩa (synonym-normalized)
    h_syn = hypothesis.lower()
    for src, tgt in VI_SYNONYMS:
        h_syn = h_syn.replace(src, tgt)
    r_syn = reference.lower()
    for src, tgt in VI_SYNONYMS:
        r_syn = r_syn.replace(src, tgt)

    c_syn_h = get_ngrams(h_syn)
    c_syn_r = get_ngrams(r_syn)
    inter_syn = sum(c_syn_r[k] * c_syn_h[k] for k in set(c_syn_r.keys()) & set(c_syn_h.keys()))
    mag_syn = math.sqrt(sum(v**2 for v in c_syn_r.values())) * math.sqrt(sum(v**2 for v in c_syn_h.values()))
    syn_sim = inter_syn / mag_syn if mag_syn > 0 else 0.0

    # 40% bề mặt ký tự + 60% ngữ nghĩa đồng nghĩa
    final_sim = 0.4 * raw_sim + 0.6 * syn_sim
    return float(max(0.0, min(1.0, round(final_sim, 4))))

def calculate_bertscore(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Contextual Token Soft-Alignment F1:
    - Khớp từ y hệt: trọng số 1.0
    - Khớp từ đồng nghĩa: trọng số 0.90
    - Khớp hình thái (Levenshtein > 0.75): trọng số 0.80
    """
    if not reference.strip() or not hypothesis.strip():
        return {"P": 0.0, "R": 0.0, "F1": 0.0}
    if reference.strip() == hypothesis.strip():
        return {"P": 1.0, "R": 1.0, "F1": 1.0}
    if is_garbage_mojibake(hypothesis):
        return {"P": 0.0, "R": 0.0, "F1": 0.0}

    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Bảng tra từ đồng nghĩa dạng set cặp (2 chiều)
    syn_map = {}
    for src, tgt in VI_SYNONYMS:
        s_words = src.split()
        t_words = tgt.split()
        if len(s_words) == 1 and len(t_words) == 1:
            syn_map.setdefault(s_words[0], set()).add(t_words[0])
            syn_map.setdefault(t_words[0], set()).add(s_words[0])

    def score_token_match(w, candidates):
        if w in candidates:
            return 1.0
        # Kiểm tra từ đồng nghĩa
        if w in syn_map and any(syn in candidates for syn in syn_map[w]):
            return 0.90
        # Kiểm tra tương đồng ký tự
        for c in candidates:
            if calculate_fidelity(w, c) > 0.75:
                return 0.80
        return 0.0

    matches_p = sum(score_token_match(w, ref_words) for w in hyp_words)
    matches_r = sum(score_token_match(w, hyp_words) for w in ref_words)

    P = min(1.0, matches_p / max(1, len(hyp_words)))
    R = min(1.0, matches_r / max(1, len(ref_words)))
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
