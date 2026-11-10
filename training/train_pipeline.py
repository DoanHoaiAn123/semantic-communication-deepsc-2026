"""
HUẤN LUYỆN TOÀN DIỆN MÔ HÌNH SEMANTIC COMMUNICATION (DeepSC)
Hỗ trợ trực tiếp các bộ Dataset Raw tải về:
  1. OPUS-100 (Anh - Việt, 1,000,000 cặp câu): data/raw/opus-100-corpus-v1.0
  2. Tatoeba (Anh - Việt, 23,172 cặp câu):   data/raw/tatoeba
  3. Combined (Kết hợp cả hai nguồn)

Bao gồm 2 giai đoạn:
  Giai đoạn 1: Huấn luyện chuẩn (Standard Training) qua kênh AWGN/Rayleigh.
  Giai đoạn 2 (Tùy chọn): Huấn luyện đối kháng thích ứng (Adversarial Training - FGSM & C&W).
"""

import os
import sys
import time
import shutil
import argparse
from datetime import datetime

# Đảm bảo UTF-8 và đường dẫn sys.path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from training.config import get_default_config
from training.dataset import create_dataloaders
from models.semantic_model import SemanticCommModel
from training.train import Trainer
from training.train_adversarial import AdversarialTrainer


def backup_existing_checkpoints(ckpt_dir: str):
    """Tạo bản sao lưu an toàn cho các checkpoint cũ trước khi huấn luyện mới."""
    if not os.path.exists(ckpt_dir):
        return
    files_to_backup = ['best_model.pth', 'last_model.pth', 'best_robust_model.pth', 'last_robust_model.pth', 'vocab.json']
    existing = [f for f in files_to_backup if os.path.exists(os.path.join(ckpt_dir, f))]
    if not existing:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(ckpt_dir, f"backup_{timestamp}")
    os.makedirs(backup_path, exist_ok=True)
    for f in existing:
        shutil.copy2(os.path.join(ckpt_dir, f), os.path.join(backup_path, f))
    print(f"  📦 Đã sao lưu an toàn các checkpoint cũ vào: {backup_path}")


def evaluate_test_loss(model, test_loader, pad_id):
    """Evaluate on the same device as the model, including on Colab GPUs."""
    device = next(model.parameters()).device
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            src_ids = batch['src_ids'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)
            src_mask = batch['src_mask']
            tgt_mask = batch['tgt_mask']
            if src_mask is not None:
                src_mask = src_mask.to(device)
            if tgt_mask is not None:
                tgt_mask = tgt_mask.to(device)
            out = model(src_ids, tgt_ids[:, :-1], snr_db=10.0, src_mask=src_mask,
                        tgt_mask=tgt_mask[:, :-1] if tgt_mask is not None else None)
            loss = criterion(out.reshape(-1, out.size(-1)), tgt_ids[:, 1:].reshape(-1))
            test_loss += loss.item()
    return test_loss / len(test_loader)


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Semantic Communication DeepSC với Dataset Raw")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "opus", "tatoeba", "combined"],
                        help="Nguồn dataset: all (toàn bộ dữ liệu loader hỗ trợ), opus (OPUS-100), tatoeba (Tatoeba TSV), combined (cả hai)")
    parser.add_argument("--samples", type=int, default=0,
                        help="Số lượng cặp câu (0 = nạp 100%% dữ liệu loader hỗ trợ, hoặc số cụ thể e.g. 5000, 10000...)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Số epoch huấn luyện chuẩn (mặc định: 3)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Kích thước batch (mặc định: 64, tối ưu tốc độ CPU)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Tốc độ học Learning Rate (mặc định: 2e-4)")
    parser.add_argument("--rebuild_vocab", action="store_true",
                        help="Xây dựng lại từ điển mới từ dataset và lưu vào checkpoints/vocab.json")
    parser.add_argument("--vocab_size", type=int, default=10000,
                        help="Kích thước từ điển tối đa nếu tạo mới (mặc định: 10000)")
    parser.add_argument("--save_every_steps", type=int, default=250,
                        help="Tần suất lưu checkpoint trung gian (mặc định: mỗi 250 batch ~ 60-90 giây)")
    parser.add_argument("--resume", action="store_true",
                        help="Huấn luyện tiếp tục từ checkpoint có sẵn (checkpoints/best_model.pth)")
    parser.add_argument("--train_adv", action="store_true",
                        help="Chạy tiếp huấn luyện phòng vệ đối kháng (FGSM + C&W)")
    parser.add_argument("--adv_epochs", type=int, default=2,
                        help="Số epoch huấn luyện đối kháng (mặc định: 2)")
    parser.add_argument("--no_backup", action="store_true",
                        help="Bỏ qua sao lưu checkpoint cũ")

    args = parser.parse_args()

    # Tối ưu số luồng CPU
    num_threads = min(16, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)

    print("=" * 72)
    print("  🚀 CHƯƠNG TRÌNH HUẤN LUYỆN SEMANTIC COMMUNICATION (DeepSC)")
    print(f"  📅 Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    device_label = (f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available()
                    else f"CPU ({num_threads} threads)")
    print(f"  💻 Thiết bị: {device_label} | PyTorch {torch.__version__}")
    print(f"  🎯 Chế độ: Huấn luyện {'100% TOÀN BỘ dữ liệu' if args.samples <= 0 else f'{args.samples:,} mẫu'}")
    print("=" * 72)

    # 1. Cấu hình
    config = get_default_config()
    data_cfg = config['data']
    train_cfg = config['training']
    model_cfg = config['model']

    ds_map = {
        'all': 'all_raw',
        'opus': 'opus_raw',
        'tatoeba': 'tatoeba_raw',
        'combined': 'combined_raw'
    }
    data_cfg.dataset_name = ds_map[args.dataset]
    data_cfg.max_samples = args.samples
    data_cfg.vocab_size = args.vocab_size

    train_cfg.batch_size = args.batch_size
    train_cfg.epochs = args.epochs
    train_cfg.lr = args.lr
    train_cfg.save_every_steps = args.save_every_steps

    ckpt_dir = train_cfg.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    if not args.no_backup:
        backup_existing_checkpoints(ckpt_dir)

    # 2. Nạp dữ liệu
    print(f"\n📂 [BƯỚC 1/4] NẠP DỮ LIỆU TỪ THƯ MỤC RAW: {args.dataset.upper()} ({args.samples:,} mẫu)...")
    t0_data = time.time()
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
        data_cfg,
        batch_size=train_cfg.batch_size,
        rebuild_vocab=args.rebuild_vocab,
        save_vocab=args.rebuild_vocab
    )
    t1_data = time.time()

    print(f"  ✓ Thời gian nạp dữ liệu: {t1_data - t0_data:.2f}s")
    print(f"  ✓ Kích thước từ vựng (Vocab size): {tokenizer.vocab_size} từ")
    print(f"  ✓ Phân chia tập dữ liệu:")
    print(f"      - Huấn luyện (Train): {len(train_loader.dataset):,} mẫu ({len(train_loader)} batches)")
    print(f"      - Xác thực (Val):     {len(val_loader.dataset):,} mẫu ({len(val_loader)} batches)")
    print(f"      - Kiểm thử (Test):    {len(test_loader.dataset):,} mẫu ({len(test_loader)} batches)")

    # Hiển thị vài mẫu thử
    print(f"\n📝 Xem trước 2 mẫu dữ liệu:")
    sample_batch = next(iter(train_loader))
    for i in range(min(2, len(sample_batch['src_text']))):
        print(f"  [Mẫu {i+1}]")
        print(f"    • English:    {sample_batch['src_text'][i]}")
        print(f"    • Tiếng Việt: {sample_batch['tgt_text'][i]}")

    # 3. Khởi tạo mô hình
    print(f"\n🧠 [BƯỚC 2/4] KHỞI TẠO MÔ HÌNH TRANSFORMER JSCC...")
    model_cfg.vocab_size = tokenizer.vocab_size

    model = SemanticCommModel(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        d_ff=model_cfg.d_ff,
        latent_dim=model_cfg.latent_dim,
        max_seq_len=model_cfg.max_seq_len,
        dropout=model_cfg.dropout
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Kiến trúc: d_model={model_cfg.d_model}, n_heads={model_cfg.n_heads}, n_layers={model_cfg.n_layers}")
    print(f"  ✓ Kênh truyền vô tuyến: AWGN / Rayleigh Fading (SNR: {train_cfg.snr_min} dB → {train_cfg.snr_max} dB)")
    print(f"  ✓ Tổng tham số mô hình: {total_params:,} ({total_params * 4 / (1024**2):.1f} MB)")

    if args.resume:
        for r_name in ['last_model.pth', 'best_model.pth']:
            r_path = os.path.join(ckpt_dir, r_name)
            if os.path.exists(r_path):
                print(f"  🔄 Đang nạp tiếp tục trọng số từ checkpoint: {r_name}")
                ck = torch.load(r_path, map_location='cpu')
                sd = ck['model_state_dict'] if isinstance(ck, dict) and 'model_state_dict' in ck else ck
                model.load_state_dict(sd, strict=False)
                print(f"  ✓ Đã nạp thành công trọng số trước đó.")
                break

    # 4. Huấn luyện chuẩn
    print(f"\n🏋️ [BƯỚC 3/4] BẮT ĐẦU HUẤN LUYỆN CHUẨN (STANDARD TRAINING) — {train_cfg.epochs} EPOCHS...")
    trainer = Trainer(model, train_loader, val_loader, tokenizer, config)
    t0_train = time.time()
    history = trainer.train(num_epochs=train_cfg.epochs)
    t1_train = time.time()

    print(f"\n  ✅ Huấn luyện chuẩn hoàn tất trong: {(t1_train - t0_train)/60:.2f} phút!")
    print(f"  ✓ Train Loss ban đầu: {history['train_loss'][0]:.4f} → Train Loss cuối: {history['train_loss'][-1]:.4f}")
    print(f"  ✓ Val Loss ban đầu:   {history['val_loss'][0]:.4f} → Val Loss cuối:   {history['val_loss'][-1]:.4f}")
    print(f"  💾 Đã lưu trọng số tốt nhất vào: {os.path.join(ckpt_dir, 'best_model.pth')}")

    # Đánh giá trên tập test độc lập
    print(f"\n🔍 Đánh giá trên tập kiểm thử (Test set)...")
    test_loss = evaluate_test_loss(model, test_loader, tokenizer.pad_id)
    print(f"  ✓ Test Loss (SNR=10dB): {test_loss:.4f}")

    # 5. Huấn luyện phòng vệ đối kháng (tùy chọn)
    if args.train_adv:
        print(f"\n🛡️ [BƯỚC 4/4] HUẤN LUYỆN PHÒNG VỆ ĐỐI KHÁNG (ADVERSARIAL TRAINING: FGSM & C&W)...")
        print(f"  ✓ Cấu hình: {args.adv_epochs} epochs | ε(FGSM)={config['adversarial'].epsilon_fgsm} | C&W steps={config['adversarial'].cw_max_iter}")
        adv_trainer = AdversarialTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            config=config
        )
        t0_adv = time.time()
        adv_history = adv_trainer.train(num_epochs=args.adv_epochs)
        t1_adv = time.time()
        print(f"\n  ✅ Huấn luyện đối kháng hoàn tất trong: {(t1_adv - t0_adv)/60:.2f} phút!")
        print(f"  💾 Đã lưu trọng số phòng vệ vào: {os.path.join(ckpt_dir, 'best_robust_model.pth')}")
    else:
        print(f"\n💡 [BƯỚC 4/4] Bỏ qua huấn luyện đối kháng (Chạy với cờ --train_adv nếu muốn kích hoạt).")

    print("\n" + "=" * 72)
    print("  🎉 TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN ĐÃ HOÀN TẤT THÀNH CÔNG RỰC RỠ!")
    print("=" * 72)
    print("\n📌 Để chạy Web Demo và kiểm tra tương tác thực tế:")
    print("   streamlit run demo/app.py\n")


if __name__ == "__main__":
    main()
