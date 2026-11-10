import torch
from torch.utils.data import Dataset, DataLoader
import re
from typing import List, Tuple
import random
import os

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

for _ in range(75):
    idx_s = random.randint(0, len(_SUBJECTS_EN) - 1)
    idx_v = random.randint(0, len(_VERBS_EN) - 1)
    idx_o = random.randint(0, len(_OBJECTS_EN) - 1)
    en = f"{_SUBJECTS_EN[idx_s]} {_VERBS_EN[idx_v]} {_OBJECTS_EN[idx_o]}."
    vi = f"{_SUBJECTS_VI[idx_s]} {_VERBS_VI[idx_v]} {_OBJECTS_VI[idx_o]}."
    SAMPLE_DATA_EN_VI.append((en, vi))


class SimpleTokenizer:
    """Tokenizer đơn giản hỗ trợ đa ngôn ngữ (Anh, Việt, Trung) cho Semantic Communication."""
    def __init__(self):
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
        """Tách token tiếng Anh / tiếng Việt theo từ, tiếng Trung theo ký tự."""
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
        """Chuyển đổi token IDs trở lại chuỗi văn bản."""
        words = []
        for i in ids:
            if skip_special_tokens and i in [self.pad_id, self.sos_id, self.eos_id]:
                continue
            word = self.id2word.get(i, self.unk_token)
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


def load_raw_opus_dataset(max_samples: int = 5000, direction: str = 'en-vi', raw_dir: str = None, include_all_splits: bool = False) -> List[Tuple[str, str]]:
    """
    Nạp dữ liệu từ file raw OPUS-100 đã tải về máy:
      data/raw/opus-100-corpus-v1.0/opus-100-corpus/v1.0/supervised/en-vi/opus.en-vi-train.{en,vi}
      Nếu include_all_splits=True, nạp thêm cả dev và test.
    """
    if raw_dir is None:
        raw_dir = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'raw', 'opus-100-corpus-v1.0', 
            'opus-100-corpus', 'v1.0', 'supervised', 'en-vi'
        )
    raw_dir = os.path.normpath(raw_dir)
    
    splits = [('opus.en-vi-train.en', 'opus.en-vi-train.vi')]
    if include_all_splits or (max_samples is not None and max_samples <= 0):
        splits.append(('opus.en-vi-dev.en', 'opus.en-vi-dev.vi'))
        splits.append(('opus.en-vi-test.en', 'opus.en-vi-test.vi'))

    data = []
    for en_file, vi_file in splits:
        en_path = os.path.join(raw_dir, en_file)
        vi_path = os.path.join(raw_dir, vi_file)
        if not os.path.exists(en_path) or not os.path.exists(vi_path):
            continue

        print(f"  [Raw OPUS-100] Đang nạp từ file cục bộ: {en_file}...")
        with open(en_path, 'r', encoding='utf-8', errors='ignore') as f_en, open(vi_path, 'r', encoding='utf-8', errors='ignore') as f_vi:
            for en, vi in zip(f_en, f_vi):
                en, vi = en.strip(), vi.strip()
                if 2 <= len(en.split()) <= 50 and 2 <= len(vi.split()) <= 50:
                    if direction == 'vi-en':
                        data.append((vi, en))
                    else:
                        data.append((en, vi))
                    if max_samples and max_samples > 0 and len(data) >= max_samples:
                        break
        if max_samples and max_samples > 0 and len(data) >= max_samples:
            break

    print(f"  [Raw OPUS-100] Đã nạp thành công {len(data):,} cặp câu song ngữ (en-vi)!")
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
    with open(tsv_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                en = parts[1].strip()
                vi = parts[3].strip()
                if 2 <= len(en.split()) <= 50 and 2 <= len(vi.split()) <= 50:
                    if direction == 'vi-en':
                        data.append((vi, en))
                    else:
                        data.append((en, vi))
                    if max_samples and max_samples > 0 and len(data) >= max_samples:
                        break

    print(f"  [Raw Tatoeba] Đã nạp thành công {len(data):,} cặp câu song ngữ!")
    return data


def load_all_raw_datasets(direction: str = 'en-vi') -> List[Tuple[str, str]]:
    """
    Nạp TOÀN BỘ dữ liệu có trong thư mục data/raw không chừa lại bất cứ gì:
      1. OPUS-100: train + dev + test
      2. Tatoeba: English - Vietnamese + Chinese - Vietnamese
    """
    print("  🔥 Đang nạp TOÀN BỘ dataset từ thư mục data/raw (không chừa lại bất kỳ file nào)...")
    data_opus = load_raw_opus_dataset(max_samples=0, direction=direction, include_all_splits=True)
    data_tatoeba_en = load_raw_tatoeba_dataset(max_samples=0, direction=direction)
    
    # Nạp thêm Tatoeba Trung - Việt nếu có
    tat_zh_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'raw', 'tatoeba',
        'Các cặp câu trong Tiếng Trung - Tiếng Việt - 2026-10-14.tsv'
    )
    tat_zh_path = os.path.normpath(tat_zh_path)
    data_tatoeba_zh = []
    if os.path.exists(tat_zh_path):
        with open(tat_zh_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    zh = parts[1].strip()
                    vi = parts[3].strip()
                    if zh and vi:
                        data_tatoeba_zh.append((zh, vi))
        print(f"  [Raw Tatoeba ZH-VI] Đã nạp thêm {len(data_tatoeba_zh):,} cặp câu Trung - Việt.")

    total_data = data_opus + data_tatoeba_en + data_tatoeba_zh
    print(f"  🌟 TỔNG CỘNG ĐÃ NẠP TOÀN BỘ: {len(total_data):,} cặp câu!")
    return total_data


def create_dataloaders(config, batch_size=64, rebuild_vocab: bool = False, save_vocab: bool = False):
    """
    Tạo DataLoader hỗ trợ cả:
      1. Raw Dataset tải về máy: data/raw/opus-100-corpus-v1.0 và data/raw/tatoeba
      2. Hugging Face: harouzie/vi_en-translation (Chuẩn đề cương)
      3. Hugging Face: opus100 (Đa ngữ)
      4. Dataset 2 Tatoeba/WMT Lifestyle (Làm giàu vocab — tùy chọn)
      5. Fallback nội bộ an toàn (Offline)
    """
    all_data = []
    dataset_name = getattr(config, 'dataset_name', 'harouzie/vi_en-translation') if not isinstance(config, dict) else config.get('dataset_name', 'harouzie/vi_en-translation')
    max_samples = getattr(config, 'max_samples', 3000) if not isinstance(config, dict) else config.get('max_samples', 3000)
    use_supplementary = getattr(config, 'use_supplementary_dataset', True) if not isinstance(config, dict) else config.get('use_supplementary_dataset', True)
    rebuild_vocab = rebuild_vocab or (getattr(config, 'rebuild_vocab', False) if not isinstance(config, dict) else config.get('rebuild_vocab', False))

    # Ưu tiên 0: Yêu cầu trực tiếp raw dataset
    if 'all_raw' in dataset_name or 'toan_bo' in dataset_name:
        all_data = load_all_raw_datasets(direction='en-vi')
    elif 'opus_raw' in dataset_name or 'opus100_raw' in dataset_name:
        all_data = load_raw_opus_dataset(max_samples=max_samples, direction='en-vi')
    elif 'tatoeba_raw' in dataset_name:
        all_data = load_raw_tatoeba_dataset(max_samples=max_samples, direction='en-vi')
    elif 'combined_raw' in dataset_name:
        half = max_samples // 2 if (max_samples and max_samples > 0) else 0
        d_opus = load_raw_opus_dataset(max_samples=half, direction='en-vi')
        d_tat = load_raw_tatoeba_dataset(max_samples=half, direction='en-vi')
        all_data = d_opus + d_tat
        print(f"  [Combined Raw] Đã kết hợp {len(all_data):,} mẫu (OPUS-100: {len(d_opus):,} + Tatoeba: {len(d_tat):,}).")

    # 1. Trường hợp dùng harouzie/vi_en-translation (Chuẩn đề cương)
    elif 'harouzie' in dataset_name or 'vi_en' in dataset_name:
        try:
            all_data = _load_harouzie_dataset(max_samples=max_samples or 3000, direction='en-vi')
        except Exception as e:
            print(f"  ⚠️ Không thể nạp trực tiếp harouzie/vi_en-translation ({e}).")
            print("  🔄 Tự động chuyển sang nạp Raw OPUS-100 đã có trong thư mục data/raw...")
            all_data = load_raw_opus_dataset(max_samples=max_samples, direction='en-vi')
            if not all_data:
                all_data = load_raw_tatoeba_dataset(max_samples=max_samples, direction='en-vi')

    # 2. Trường hợp dùng opus100 từ huggingface
    elif 'opus' in dataset_name:
        # Thử nạp raw offline trước
        all_data = load_raw_opus_dataset(max_samples=max_samples, direction='en-vi')
        if not all_data:
            max_per_pair = getattr(config, 'max_samples_per_pair', 500) if not isinstance(config, dict) else config.get('max_samples_per_pair', 500)
            lang_pairs = getattr(config, 'language_pairs', []) if not isinstance(config, dict) else config.get('language_pairs', [])
            for pair_info in lang_pairs:
                src_lang = pair_info['src']
                tgt_lang = pair_info['tgt']
                subset = pair_info.get('subset')
                if subset is not None:
                    try:
                        data = _load_opus100_pair(dataset_name, subset, src_lang, tgt_lang, max_per_pair)
                        all_data.extend(data)
                    except Exception:
                        pass

    # 3. Fallback nếu không có dữ liệu: thử nạp từ raw folder trước khi dùng SAMPLE_DATA
    if not all_data:
        all_data = load_raw_opus_dataset(max_samples=max_samples, direction='en-vi')
        if not all_data:
            all_data = load_raw_tatoeba_dataset(max_samples=max_samples, direction='en-vi')

    if not all_data:
        print("  ℹ️ Sử dụng dữ liệu mẫu nội bộ phong phú (Anh - Việt).")
        all_data = SAMPLE_DATA_EN_VI

    if max_samples and max_samples > 0 and len(all_data) > max_samples:
        all_data = all_data[:max_samples]

    random.shuffle(all_data)

    # ── Dataset 2: Làm giàu từ vựng cho tokenizer ─────────────────────────────
    # Chỉ dùng văn bản để build_vocab phong phú hơn (đa miền).
    # KHÔNG trộn vào tập train/val/test của Dataset 1 → giữ nguyên split chuẩn.
    supplementary_texts: List[str] = []
    if use_supplementary:
        d2_data = load_supplementary_dataset()
        if d2_data:
            supplementary_texts = [src for src, _ in d2_data] + [tgt for _, tgt in d2_data]
            print(f"  [Dataset 2] Bổ sung {len(supplementary_texts)} câu vào vocab (không trộn vào train).")

    # Xây dựng tokenizer từ file vocab đã lưu (khớp với checkpoint) hoặc tạo mới
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
        # Dùng cả văn bản Dataset 1 + Dataset 2 để build vocab phong phú hơn
        all_texts = [src for src, _ in all_data] + [tgt for _, tgt in all_data]
        all_texts += supplementary_texts  # Làm giàu từ vựng từ Dataset 2
        max_vocab = getattr(config, 'vocab_size', 10000) if not isinstance(config, dict) else config.get('vocab_size', 10000)
        tokenizer.build_vocab(all_texts, max_vocab=max_vocab)
        print(f"  [Vocab] Xây dựng từ điển mới: {tokenizer.vocab_size} từ từ {len(all_texts)} câu.")
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

    test_ratio = getattr(config, 'test_ratio', 0.1) if not isinstance(config, dict) else config.get('test_ratio', 0.1)
    val_ratio = getattr(config, 'val_ratio', 0.1) if not isinstance(config, dict) else config.get('val_ratio', 0.1)

    n_total = len(all_data)
    n_test = max(1, int(n_total * test_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_train = max(1, n_total - n_test - n_val)

    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train+n_val]
    test_data = all_data[n_train+n_val:]

    train_dataset = SemanticCommDataset(train_data)
    val_dataset = SemanticCommDataset(val_data)
    test_dataset = SemanticCommDataset(test_data)

    from functools import partial
    collate = partial(collate_fn, tokenizer=tokenizer, max_len=64)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    return train_loader, val_loader, test_loader, tokenizer
