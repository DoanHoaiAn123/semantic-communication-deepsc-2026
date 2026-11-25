import torch
from torch.utils.data import Dataset, DataLoader
import re
from typing import List, Tuple, Optional
import random
import os
import json
import unicodedata

# =====================================================================
# Dữ liệu mẫu 3 ngôn ngữ (Anh, Việt, Trung) — fallback khi không có internet
# =====================================================================
SAMPLE_DATA_EN_VI = [
    ("The weather is very nice today.", "Hôm nay thời tiết rất đẹp."),
    ("I love studying artificial intelligence.", "Tôi thích học trí tuệ nhân tạo."),
    ("Semantic communication preserves meaning.", "Truyền thông ngữ nghĩa bảo toàn ý nghĩa."),
    ("The cat is sitting on the sofa.", "Con mèo đang ngồi trên ghế sofa."),
    ("She goes to school every day.", "Cô ấy đi học mỗi ngày."),
    ("This book is very interesting.", "Cuốn sách này rất thú vị."),
    ("He works at a big company.", "Anh ấy làm việc ở một công ty lớn."),
    ("We need to finish this project.", "Chúng ta cần hoàn thành dự án này."),
    ("The students are studying hard.", "Các sinh viên đang học chăm chỉ."),
    ("Technology is changing the world.", "Công nghệ đang thay đổi thế giới."),
    ("Please open the window.", "Làm ơn mở cửa sổ."),
    ("I want to drink some water.", "Tôi muốn uống một ít nước."),
    ("The train arrives at five o'clock.", "Tàu đến lúc năm giờ."),
    ("My family lives in the city.", "Gia đình tôi sống ở thành phố."),
    ("The food at this restaurant is delicious.", "Đồ ăn ở nhà hàng này rất ngon."),
    ("He is reading a newspaper.", "Anh ấy đang đọc báo."),
    ("She can speak three languages.", "Cô ấy có thể nói ba ngôn ngữ."),
    ("The children are playing in the park.", "Bọn trẻ đang chơi trong công viên."),
    ("I need to buy some groceries.", "Tôi cần mua một ít đồ tạp hóa."),
    ("The movie was very exciting.", "Bộ phim rất hấp dẫn."),
    ("We should protect the environment.", "Chúng ta nên bảo vệ môi trường."),
    ("The doctor is very kind.", "Bác sĩ rất tốt bụng."),
    ("Learning a new language takes time.", "Học một ngôn ngữ mới cần thời gian."),
    ("The sun rises in the east.", "Mặt trời mọc ở phía đông."),
    ("He plays football every weekend.", "Anh ấy chơi bóng đá mỗi cuối tuần."),
]

SAMPLE_DATA_EN_ZH = [
    ("The weather is very nice today.", "今天天气很好。"),
    ("I love studying artificial intelligence.", "我喜欢学习人工智能。"),
    ("Semantic communication preserves meaning.", "语义通信保留了意义。"),
    ("The cat is sitting on the sofa.", "猫坐在沙发上。"),
    ("She goes to school every day.", "她每天去上学。"),
    ("This book is very interesting.", "这本书非常有趣。"),
    ("He works at a big company.", "他在一家大公司工作。"),
    ("We need to finish this project.", "我们需要完成这个项目。"),
    ("The students are studying hard.", "学生们正在努力学习。"),
    ("Technology is changing the world.", "科技正在改变世界。"),
]

SAMPLE_DATA_VI_ZH = [
    ("Hôm nay thời tiết rất đẹp.", "今天天气很好。"),
    ("Tôi thích học trí tuệ nhân tạo.", "我喜欢学习人工智能。"),
    ("Truyền thông ngữ nghĩa bảo toàn ý nghĩa.", "语义通信保留了意义。"),
    ("Con mèo đang ngồi trên ghế sofa.", "猫坐在沙发上。"),
    ("Cô ấy đi học mỗi ngày.", "她每天去上学。"),
]

# Bổ sung dữ liệu sinh tự động từ mẫu
_SUBJECTS_EN = ["The cat", "A dog", "My friend", "The teacher", "A student", "The engineer"]
_VERBS_EN = ["runs", "walks", "jumps", "sleeps", "eats", "reads"]
_OBJECTS_EN = ["in the park", "at home", "quickly", "slowly", "with joy", "a book"]

_SUBJECTS_VI = ["Con mèo", "Con chó", "Bạn tôi", "Giáo viên", "Sinh viên", "Kỹ sư"]
_VERBS_VI = ["chạy", "đi bộ", "nhảy", "ngủ", "ăn", "đọc"]
_OBJECTS_VI = ["trong công viên", "ở nhà", "nhanh chóng", "chậm rãi", "vui vẻ", "một cuốn sách"]

_SAMPLE_RNG = random.Random(42)
for _ in range(75):
    idx_s = _SAMPLE_RNG.randint(0, len(_SUBJECTS_EN) - 1)
    idx_v = _SAMPLE_RNG.randint(0, len(_VERBS_EN) - 1)
    idx_o = _SAMPLE_RNG.randint(0, len(_OBJECTS_EN) - 1)
    en = f"{_SUBJECTS_EN[idx_s]} {_VERBS_EN[idx_v]} {_OBJECTS_EN[idx_o]}."
    vi = f"{_SUBJECTS_VI[idx_s]} {_VERBS_VI[idx_v]} {_OBJECTS_VI[idx_o]}."
    SAMPLE_DATA_EN_VI.append((en, vi))


class SimpleTokenizer:
    """Tokenizer đơn giản hỗ trợ đa ngôn ngữ (Anh, Việt, Trung) cho Semantic Communication."""
    def __init__(self, max_vocab_size: int = 10000, vocab_file: Optional[str] = None):
        self.max_vocab_size = max_vocab_size
        self.pad_token = '[PAD]'
        self.sos_token = '[SOS]'
        self.eos_token = '[EOS]'
        self.unk_token = '[UNK]'
        self.lang_token_en = '[EN]'
        self.lang_token_vi = '[VI]'
        self.lang_token_zh = '[ZH]'
        
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        
        self.word2id = {
            self.pad_token: 0,
            self.sos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
            self.lang_token_en: 4,
            self.lang_token_vi: 5,
            self.lang_token_zh: 6,
        }
        self.id2word = {v: k for k, v in self.word2id.items()}
        if vocab_file and os.path.exists(vocab_file):
            self.load(vocab_file)

    def load(self, vocab_path: str):
        """Nạp từ điển từ tệp JSON."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            v_data = json.load(f)
        self.word2id = v_data['word2id']
        self.id2word = {int(k): v for k, v in v_data['id2word'].items()}

    def save(self, vocab_path: str):
        """Lưu từ điển ra tệp JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(vocab_path)), exist_ok=True)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({
                'word2id': self.word2id,
                'id2word': {str(k): v for k, v in self.id2word.items()},
                'vocab_size': self.vocab_size
            }, f, ensure_ascii=False, indent=2)
        
    def build_vocab(self, sentences: List[str], max_vocab: int = 10000):
        """Xây dựng từ điển từ danh sách các câu."""
        word_counts = {}
        for sentence in sentences:
            words = self._tokenize(sentence)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        n_special = len(self.word2id)
        for word, _ in sorted_words[:max_vocab - n_special]:
            if word not in self.word2id:
                idx = len(self.word2id)
                self.word2id[word] = idx
                self.id2word[idx] = word
                
    def _is_chinese_char(self, c: str) -> bool:
        """Kiểm tra ký tự có phải chữ Hán không."""
        cp = ord(c)
        return (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF)

    def _tokenize(self, text: str) -> List[str]:
        """Tách token tiếng Anh / tiếng Việt theo từ, tiếng Trung theo ký tự (chuẩn hóa NFC)."""
        if not text:
            return []
        text = unicodedata.normalize('NFC', str(text))
        tokens = []
        buffer = []
        
        for char in text:
            if self._is_chinese_char(char):
                if buffer:
                    segment = ''.join(buffer).strip().lower()
                    segment = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', '', segment)
                    tokens.extend(segment.split())
                    buffer = []
                tokens.append(char)
            else:
                buffer.append(char)
        
        if buffer:
            segment = ''.join(buffer).strip().lower()
            segment = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', '', segment)
            tokens.extend(segment.split())
        
        return [t for t in tokens if t]
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Chuyển đổi chuỗi thành token IDs."""
        words = self._tokenize(text)
        ids = [self.word2id.get(word, self.unk_id) for word in words]
        if add_special_tokens:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids
        
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Chuyển đổi token IDs trở lại chuỗi văn bản (dừng tại [EOS] đầu tiên)."""
        words = []
        for i in ids:
            idx = int(i)
            if skip_special_tokens and idx == self.eos_id:
                break
            if skip_special_tokens and idx in [self.pad_id, self.sos_id]:
                continue
            word = self.id2word.get(idx, self.unk_token)
            if skip_special_tokens and word in [self.lang_token_en, self.lang_token_vi, self.lang_token_zh]:
                continue
            words.append(word)
        
        result = []
        for w in words:
            if result and len(w) == 1 and self._is_chinese_char(w):
                result.append(w)
            elif result and len(result[-1]) == 1 and self._is_chinese_char(result[-1]):
                result.append(w)
            else:
                if result:
                    result.append(' ')
                result.append(w)
        
        return ''.join(result)
        
    @property
    def vocab_size(self) -> int:
        return len(self.word2id)


class SemanticCommDataset(Dataset):
    """Dataset cho quá trình huấn luyện Semantic Communication."""
    def __init__(self, data: List[Tuple[str, str]]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, tokenizer, max_len=64):
    """Hàm collate để đệm (pad) sequence đồng đều trong batch."""
    src_batch, tgt_batch = zip(*batch)
    
    src_ids_list = [tokenizer.encode(src) for src in src_batch]
    tgt_ids_list = [tokenizer.encode(tgt) for tgt in tgt_batch]
    
    src_ids_list = [ids[:max_len] for ids in src_ids_list]
    tgt_ids_list = [ids[:max_len] for ids in tgt_ids_list]
    
    max_src_len = max(len(ids) for ids in src_ids_list)
    max_tgt_len = max(len(ids) for ids in tgt_ids_list)
    
    src_padded = [ids + [tokenizer.pad_id] * (max_src_len - len(ids)) for ids in src_ids_list]
    tgt_padded = [ids + [tokenizer.pad_id] * (max_tgt_len - len(ids)) for ids in tgt_ids_list]
    
    src_tensor = torch.tensor(src_padded, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_padded, dtype=torch.long)
    
    src_mask = (src_tensor == tokenizer.pad_id)
    tgt_mask = (tgt_tensor == tokenizer.pad_id)
    
    return {
        'src_ids': src_tensor,
        'tgt_ids': tgt_tensor,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
        'src_text': src_batch,
        'tgt_text': tgt_batch
    }


def load_supplementary_dataset(csv_path: str = None, direction: str = 'en-vi') -> List[Tuple[str, str]]:
    """
    Tải bộ Dataset 2 đối chứng từ file CSV nội bộ:
      data/processed/dataset2_tatoeba_wmt.csv

    Format CSV: id, category, english, vietnamese

    Mục đích:
      - Làm giàu từ vựng ngữ cảnh cho Transformer (đa miền: đời sống, giáo dục, y tế...)
      - Đối chứng khả năng tổng quát hóa ngoài domain chuyên ngành hẹp
      - Đánh giá tấn công FGSM/C&W trên miền dữ liệu khác biệt

    Returns:
        List[Tuple[str, str]] — danh sách cặp (src, tgt)
    """
    import csv as csv_module

    if csv_path is None:
        csv_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'processed', 'dataset2_tatoeba_wmt.csv'
        )
    csv_path = os.path.normpath(csv_path)

    if not os.path.exists(csv_path):
        print(f"  ⚠️  Không tìm thấy Dataset 2 tại: {csv_path}")
        return []

    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            en = (row.get('english') or row.get('English') or '').strip()
            vi = (row.get('vietnamese') or row.get('Vietnamese') or '').strip()
            if en and vi:
                if direction == 'vi-en':
                    data.append((vi, en))
                else:
                    data.append((en, vi))

    print(f"  [Dataset 2] Đã nạp {len(data)} cặp câu từ Tatoeba/WMT Lifestyle.")
    return data


def load_dataset2_for_evaluation(direction: str = 'en-vi') -> List[Tuple[str, str]]:
    """
    Nạp toàn bộ Dataset 2 (Tatoeba/WMT) để dùng riêng trong bước evaluation/đối chứng.
    Không trộn vào tập train — chỉ dùng để test và so sánh.

    Returns:
        List[Tuple[str, str]] — toàn bộ 100 cặp câu Dataset 2
    """
    return load_supplementary_dataset(direction=direction)


def _load_harouzie_dataset(max_samples: int = 3000, direction: str = 'en-vi') -> List[Tuple[str, str]]:
    """
    Tải bộ dữ liệu Hugging Face: harouzie/vi_en-translation (Theo đề cương khóa luận).
    Features: {'English': str, 'Vietnamese': str}
    """
    from datasets import load_dataset
    print(f"  [Hugging Face] Đang nạp dataset: harouzie/vi_en-translation (tối đa {max_samples} mẫu)...")
    
    # Nạp slice từ split train
    ds = load_dataset('harouzie/vi_en-translation', split=f'train[:{max_samples}]')
    data = []
    for item in ds:
        en = (item.get('English') or '').strip()
        vi = (item.get('Vietnamese') or '').strip()
        
        # Bỏ qua câu rỗng hoặc quá dài (> 45 từ)
        if en and vi and len(en.split()) <= 45 and len(vi.split()) <= 45:
            if direction == 'vi-en':
                data.append((vi, en))
            else:
                data.append((en, vi))
                
    print(f"  [Hugging Face] Đã nạp thành công {len(data)} cặp câu song ngữ Việt - Anh.")
    return data


def _count_effective_tokens(text: str) -> int:
    """Đếm số từ (Anh/Việt) hoặc ký tự chữ Hán (Trung) sau chuẩn hóa NFC."""
    if not text:
        return 0
    text = unicodedata.normalize('NFC', text.strip())
    cjk_chars = []
    non_cjk_chars = []
    for c in text:
        if (0x4E00 <= ord(c) <= 0x9FFF) or (0x3400 <= ord(c) <= 0x4DBF):
            cjk_chars.append(c)
            non_cjk_chars.append(' ')
        else:
            non_cjk_chars.append(c)
    non_cjk_str = ''.join(non_cjk_chars)
    cleaned = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', non_cjk_str)
    word_cnt = len(cleaned.split())
    return len(cjk_chars) + word_cnt


def load_raw_opus_dataset(max_samples: int = 5000, direction: str = 'en-vi', raw_dir: str = None, split: str = 'train') -> List[Tuple[str, str]]:
    """
    Nạp dữ liệu từ file raw OPUS-100 đã tải về máy theo đúng phân chia chính thức:
      - split='train': chỉ nạp opus.en-vi-train.{en,vi}
      - split='dev':   chỉ nạp opus.en-vi-dev.{en,vi}
      - split='test':  chỉ nạp opus.en-vi-test.{en,vi}
    Tuyệt đối KHÔNG tự động gộp dev/test vào train để tránh Data Leakage.
    """
    if raw_dir is None:
        raw_dir = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'raw', 'opus-100-corpus-v1.0', 
            'opus-100-corpus', 'v1.0', 'supervised', 'en-vi'
        )
    raw_dir = os.path.normpath(raw_dir)

    valid_splits = {'train', 'dev', 'test'}
    if split not in valid_splits:
        raise ValueError(f"split phải thuộc {valid_splits}, nhận được: '{split}'")

    en_file = f'opus.en-vi-{split}.en'
    vi_file = f'opus.en-vi-{split}.vi'
    en_path = os.path.join(raw_dir, en_file)
    vi_path = os.path.join(raw_dir, vi_file)

    data = []
    seen_pairs = set()
    if not os.path.exists(en_path) or not os.path.exists(vi_path):
        return data

    print(f"  [Raw OPUS-100 | {split}] Đang nạp từ file cục bộ: {en_file}...")
    with open(en_path, 'r', encoding='utf-8', errors='ignore') as f_en, open(vi_path, 'r', encoding='utf-8', errors='ignore') as f_vi:
        for en, vi in zip(f_en, f_vi):
            en = unicodedata.normalize('NFC', en.strip())
            vi = unicodedata.normalize('NFC', vi.strip())
            if not en or not vi or en.lower() == vi.lower():
                continue
            len_en = _count_effective_tokens(en)
            len_vi = _count_effective_tokens(vi)
            if 2 <= len_en <= 50 and 2 <= len_vi <= 50:
                pair = (vi, en) if direction == 'vi-en' else (en, vi)
                pair_key = (pair[0].lower(), pair[1].lower())
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                data.append(pair)
                if max_samples and max_samples > 0 and len(data) >= max_samples:
                    break

    print(f"  [Raw OPUS-100 | {split}] Đã nạp thành công {len(data):,} cặp câu song ngữ ({direction})!")
    return data


def load_raw_tatoeba_dataset(max_samples: int = 5000, direction: str = 'en-vi', tsv_path: str = None) -> List[Tuple[str, str]]:
    """
    Nạp dữ liệu từ file raw Tatoeba TSV đã tải về máy:
      data/raw/tatoeba/Các cặp câu trong Tiếng Anh - Tiếng Việt - 2026-10-14.tsv
    """
    if tsv_path is None:
        tsv_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'raw', 'tatoeba', 
            'Các cặp câu trong Tiếng Anh - Tiếng Việt - 2026-10-14.tsv'
        )
    tsv_path = os.path.normpath(tsv_path)
    if not os.path.exists(tsv_path):
        return []

    print(f"  [Raw Tatoeba] Đang nạp từ file TSV cục bộ: {tsv_path}...")
    data = []
    seen_pairs = set()
    with open(tsv_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                en = unicodedata.normalize('NFC', parts[1].strip())
                vi = unicodedata.normalize('NFC', parts[3].strip())
                if not en or not vi or en.lower() == vi.lower():
                    continue
                len_en = _count_effective_tokens(en)
                len_vi = _count_effective_tokens(vi)
                if 2 <= len_en <= 50 and 2 <= len_vi <= 50:
                    pair = (vi, en) if direction == 'vi-en' else (en, vi)
                    pair_key = (pair[0].lower(), pair[1].lower())
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    data.append(pair)
                    if max_samples and max_samples > 0 and len(data) >= max_samples:
                        break

    print(f"  [Raw Tatoeba] Đã nạp thành công {len(data):,} cặp câu song ngữ!")
    return data


def load_all_raw_datasets(direction: str = 'en-vi', max_samples: int = 0) -> List[Tuple[str, str]]:
    """
    Nạp tập huấn luyện OPUS-100 sạch (chỉ split='train').
    KHÔNG trộn opus.en-vi-dev, opus.en-vi-test và KHÔNG trộn Tatoeba vào tập Train
    để đảm bảo tính độc lập hoàn toàn cho Validation, Test và đánh giá chéo miền (Tập 2).
    """
    print("  [Clean Train Loader] Chỉ nạp OPUS-100 split='train' (giữ dev/test và Tatoeba hoàn toàn độc lập)...")
    return load_raw_opus_dataset(max_samples=max_samples, direction=direction, split='train')


def _deduplicate_train_data(
    train_data: List[Tuple[str, str]],
    *excluded_splits: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """Loại bỏ các cặp câu trong train_data bị trùng câu nguồn (src) HOẶC câu đích (tgt) với val/test/tatoeba."""
    excluded_srcs = set()
    excluded_tgts = set()
    for split_data in excluded_splits:
        if not split_data:
            continue
        for src, tgt in split_data:
            excluded_srcs.add(unicodedata.normalize('NFC', src.strip().lower()))
            excluded_tgts.add(unicodedata.normalize('NFC', tgt.strip().lower()))

    if not excluded_srcs and not excluded_tgts:
        return train_data

    cleaned = [
        pair for pair in train_data
        if unicodedata.normalize('NFC', pair[0].strip().lower()) not in excluded_srcs
        and unicodedata.normalize('NFC', pair[1].strip().lower()) not in excluded_tgts
    ]
    removed = len(train_data) - len(cleaned)
    if removed > 0:
        print(f"  [Deduplication] Đã loại bỏ {removed:,} câu trùng lặp (cả src & tgt) giữa Train với Val/Test/Tatoeba.")
    return cleaned


def create_dataloaders(config, batch_size=64, rebuild_vocab: bool = False, save_vocab: bool = False):
    """
    Tạo DataLoader đảm bảo KHÔNG Data Leakage:
      1. Phân chia riêng biệt train_data / val_data / test_data TRƯỚC khi xây dựng từ điển:
         - Với OPUS-100 raw: dùng đúng 3 split chính thức (train / dev / test), chỉ shuffle train_data.
         - Giữ Tatoeba hoàn toàn độc lập làm Tập 2 (Cross-Domain Evaluation).
         - Lọc bỏ các câu trùng lặp giữa train_data với val_data, test_data và Tatoeba.
      2. Chỉ xây dựng Vocabulary (tokenizer.build_vocab) từ văn bản thuộc train_data.
    """
    dataset_name = getattr(config, 'dataset_name', 'harouzie/vi_en-translation') if not isinstance(config, dict) else config.get('dataset_name', 'harouzie/vi_en-translation')
    max_samples = getattr(config, 'max_samples', 3000) if not isinstance(config, dict) else config.get('max_samples', 3000)
    rebuild_vocab = rebuild_vocab or (getattr(config, 'rebuild_vocab', False) if not isinstance(config, dict) else config.get('rebuild_vocab', False))
    test_ratio = getattr(config, 'test_ratio', 0.1) if not isinstance(config, dict) else config.get('test_ratio', 0.1)
    val_ratio = getattr(config, 'val_ratio', 0.1) if not isinstance(config, dict) else config.get('val_ratio', 0.1)

    train_data: List[Tuple[str, str]] = []
    val_data: List[Tuple[str, str]] = []
    test_data: List[Tuple[str, str]] = []
    all_data: List[Tuple[str, str]] = []

    use_official_opus_splits = False

    # 1. Nạp dữ liệu theo cấu hình (KHÔNG gộp dev/test/Tatoeba vào tập train OPUS-100)
    if any(k in dataset_name for k in ('all_raw', 'toan_bo', 'opus_raw', 'opus100_raw', 'combined_raw')):
        train_data = load_raw_opus_dataset(max_samples=max_samples, direction='en-vi', split='train')
        val_data = load_raw_opus_dataset(max_samples=0, direction='en-vi', split='dev')
        test_data = load_raw_opus_dataset(max_samples=0, direction='en-vi', split='test')
        if train_data and val_data and test_data:
            use_official_opus_splits = True
        else:
            all_data = train_data

    elif 'tatoeba_raw' in dataset_name:
        all_data = load_raw_tatoeba_dataset(max_samples=max_samples, direction='en-vi')

    elif 'harouzie' in dataset_name or 'vi_en' in dataset_name:
        try:
            all_data = _load_harouzie_dataset(max_samples=max_samples or 3000, direction='en-vi')
        except Exception as e:
            print(f"  ⚠️ Không thể nạp trực tiếp harouzie/vi_en-translation ({e}).")
            print("  🔄 Tự động chuyển sang nạp Raw OPUS-100 (tách riêng train/dev/test)...")
            train_data = load_raw_opus_dataset(max_samples=max_samples, direction='en-vi', split='train')
            val_data = load_raw_opus_dataset(max_samples=0, direction='en-vi', split='dev')
            test_data = load_raw_opus_dataset(max_samples=0, direction='en-vi', split='test')
            if train_data and val_data and test_data:
                use_official_opus_splits = True
            else:
                all_data = train_data

    elif 'opus' in dataset_name:
        train_data = load_raw_opus_dataset(max_samples=max_samples, direction='en-vi', split='train')
        val_data = load_raw_opus_dataset(max_samples=0, direction='en-vi', split='dev')
        test_data = load_raw_opus_dataset(max_samples=0, direction='en-vi', split='test')
        if train_data and val_data and test_data:
            use_official_opus_splits = True
        else:
            all_data = train_data

    # 2. Fallback an toàn nếu chưa có dữ liệu
    if not use_official_opus_splits and not all_data:
        train_data = load_raw_opus_dataset(max_samples=max_samples, direction='en-vi', split='train')
        val_data = load_raw_opus_dataset(max_samples=0, direction='en-vi', split='dev')
        test_data = load_raw_opus_dataset(max_samples=0, direction='en-vi', split='test')
        if train_data and val_data and test_data:
            use_official_opus_splits = True
        else:
            all_data = train_data or load_raw_tatoeba_dataset(max_samples=max_samples, direction='en-vi')

    if not use_official_opus_splits and not all_data:
        print("  ℹ️ Sử dụng dữ liệu mẫu nội bộ phong phú (Anh - Việt).")
        all_data = list(SAMPLE_DATA_EN_VI)

    # 3. Chia tập Train / Validation / Test TRƯỚC khi xây dựng Vocabulary (cố định seed để tái lập)
    split_seed = getattr(config, 'seed', 42) if not isinstance(config, dict) else config.get('seed', 42)
    split_rng = random.Random(split_seed)
    if use_official_opus_splits:
        # Khử trùng lặp giữa dev và test
        val_data = _deduplicate_train_data(val_data, test_data)
        # Lọc trùng giữa train với dev, test và Tatoeba (Tập 2 độc lập) rồi chỉ shuffle trong train_data
        tatoeba_eval = load_raw_tatoeba_dataset(max_samples=0, direction='en-vi')
        train_data = _deduplicate_train_data(train_data, val_data, test_data, tatoeba_eval)
        split_rng.shuffle(train_data)
    else:
        if max_samples and max_samples > 0 and len(all_data) > max_samples:
            all_data = all_data[:max_samples]
        split_rng.shuffle(all_data)

        n_total = len(all_data)
        n_test = max(1, int(n_total * test_ratio))
        n_val = max(1, int(n_total * val_ratio))
        n_train = max(1, n_total - n_test - n_val)

        train_data = all_data[:n_train]
        val_data = all_data[n_train:n_train + n_val]
        test_data = all_data[n_train + n_val:]
        val_data = _deduplicate_train_data(val_data, test_data)
        train_data = _deduplicate_train_data(train_data, val_data, test_data)

    # 4. Xây dựng Tokenizer: CHỈ dùng văn bản của train_data (KHÔNG dùng val, test hay Dataset 2)
    tokenizer = SimpleTokenizer()
    vocab_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'vocab.json')
    if os.path.exists(vocab_path) and not rebuild_vocab:
        import json
        with open(vocab_path, 'r', encoding='utf-8') as f:
            v_data = json.load(f)
        tokenizer.word2id = v_data['word2id']
        tokenizer.id2word = {int(k): v for k, v in v_data['id2word'].items()}
        print(f"  [Vocab] Sử dụng từ điển có sẵn ({tokenizer.vocab_size} từ).")
    else:
        train_texts = [src for src, _ in train_data] + [tgt for _, tgt in train_data]
        max_vocab = getattr(config, 'vocab_size', 10000) if not isinstance(config, dict) else config.get('vocab_size', 10000)
        tokenizer.build_vocab(train_texts, max_vocab=max_vocab)
        print(f"  [Vocab] Xây dựng từ điển mới CHỈ từ train_data: {tokenizer.vocab_size} từ từ {len(train_texts):,} câu.")
        if save_vocab:
            import json
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'word2id': tokenizer.word2id,
                    'id2word': {str(k): v for k, v in tokenizer.id2word.items()},
                    'vocab_size': tokenizer.vocab_size
                }, f, ensure_ascii=False, indent=2)
            print(f"  [Vocab] Đã lưu từ điển mới vào: {vocab_path}")

    train_dataset = SemanticCommDataset(train_data)
    val_dataset = SemanticCommDataset(val_data)
    test_dataset = SemanticCommDataset(test_data)

    from functools import partial
    collate = partial(collate_fn, tokenizer=tokenizer, max_len=64)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    return train_loader, val_loader, test_loader, tokenizer
