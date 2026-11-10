import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import json
import torch
from training.config import get_default_config
from training.dataset import create_dataloaders
from models.semantic_model import SemanticCommModel

cfg = get_default_config()
cfg['data'].dataset_name = 'harouzie/vi_en-translation'
cfg['data'].max_samples = 1500

print("Nạp dữ liệu 1500 mẫu và trích xuất từ điển chuẩn...")
_, _, _, tokenizer = create_dataloaders(cfg['data'], batch_size=2)
print(f"Vocab size thực tế: {tokenizer.vocab_size}")

ckpt_dir = r"c:\Users\doanh\OneDrive\Desktop\Tốt Nghiệp 2026\semantic-communication\checkpoints"
vocab_path = os.path.join(ckpt_dir, "vocab.json")

# Lưu từ điển
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump({
        "word2id": tokenizer.word2id,
        "id2word": {str(k): v for k, v in tokenizer.id2word.items()},
        "vocab_size": tokenizer.vocab_size
    }, f, ensure_ascii=False, indent=2)

print(f"Đã lưu từ điển chuẩn tại: {vocab_path}")

# Kiểm tra tương thích với best_model.pth
ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
ckpt = torch.load(ckpt_path, map_location="cpu")
model = SemanticCommModel(vocab_size=tokenizer.vocab_size)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
print("✅ Khớp 100% trọng số với checkpoint best_model.pth!")
