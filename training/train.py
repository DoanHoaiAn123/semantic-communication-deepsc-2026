import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import math
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

class Trainer:
    """Lớp quản lý quá trình huấn luyện mô hình Semantic Communication."""
    def __init__(self, model, train_loader, val_loader, tokenizer, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config.get('training') if isinstance(config, dict) else config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Hàm mất mát với label smoothing để chống overconfidence
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=0.1
        )
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        # Warmup cosine scheduler
        total_steps = len(self.train_loader) * self.config.epochs
        self.scheduler = self._get_scheduler(total_steps)
        
        # Tạo thư mục checkpoint
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
    def _get_scheduler(self, total_steps):
        """Tạo lịch trình học tốc độ warmup và cosine decay."""
        warmup_steps = self.config.warmup_steps
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
        return LambdaLR(self.optimizer, lr_lambda)
        
    def train_epoch(self, epoch_idx=1):
        """Huấn luyện một epoch."""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch_idx}')
        try:
            for batch_idx, batch in enumerate(progress_bar):
                src_ids = batch['src_ids'].to(self.device)
                tgt_ids = batch['tgt_ids'].to(self.device)
                src_mask = batch['src_mask'].to(self.device)
                tgt_mask = batch['tgt_mask'].to(self.device)
                
                # Khởi tạo mức nhiễu SNR ngẫu nhiên nếu huấn luyện nhận biết nhiễu
                if self.config.noise_aware:
                    snr_db = random.uniform(self.config.snr_min, self.config.snr_max)
                else:
                    snr_db = 20.0 # Mức SNR cao coi như kênh lý tưởng
                    
                self.optimizer.zero_grad()
                
                # Forward pass: tgt_ids[:, :-1] dùng làm đầu vào cho decoder
                # tgt_ids[:, 1:] là nhãn cần dự đoán
                input_tgt_ids = tgt_ids[:, :-1]
                target_ids = tgt_ids[:, 1:]
                
                logits = self.model(
                    src_ids=src_ids,
                    tgt_ids=input_tgt_ids,
                    snr_db=snr_db,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask[:, :-1] if tgt_mask is not None else None
                )
                
                # Tính loss
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                pct = ((batch_idx + 1) / len(self.train_loader)) * 100
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'snr': f'{snr_db:.1f}dB', 'prog': f'{pct:.1f}%'})

                # Tự động lưu checkpoint định kỳ (mặc định mỗi 250 batch ~ 60-90 giây)
                save_interval = getattr(self.config, 'save_every_steps', 250)
                if save_interval and (batch_idx + 1) % save_interval == 0:
                    mid_path = os.path.join(self.config.checkpoint_dir, 'last_model.pth')
                    self.save_checkpoint(mid_path, epoch=epoch_idx, step=batch_idx + 1)

            return total_loss / len(self.train_loader)
        except KeyboardInterrupt:
            pct = ((batch_idx + 1) / len(self.train_loader)) * 100
            print(f"\n\n  ⚠️ BẮT ĐƯỢC TÍN HIỆU DỪNG (Ctrl+C) TẠI BATCH {batch_idx + 1}/{len(self.train_loader)} ({pct:.1f}%)")
            mid_path = os.path.join(self.config.checkpoint_dir, 'last_model.pth')
            self.save_checkpoint(mid_path, epoch=epoch_idx, step=batch_idx + 1)
            print(f"  💾 ĐÃ LƯU TIẾN TRÌNH CHÍNH XÁC VÀO: {mid_path}")
            raise

    def validate(self):
        """Đánh giá trên tập validation."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                src_ids = batch['src_ids'].to(self.device)
                tgt_ids = batch['tgt_ids'].to(self.device)
                src_mask = batch['src_mask'].to(self.device)
                tgt_mask = batch['tgt_mask'].to(self.device)
                
                snr_db = random.choice([-5, 0, 5, 10, 15, 20])
                
                input_tgt_ids = tgt_ids[:, :-1]
                target_ids = tgt_ids[:, 1:]
                
                logits = self.model(
                    src_ids=src_ids,
                    tgt_ids=input_tgt_ids,
                    snr_db=snr_db,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask[:, :-1] if tgt_mask is not None else None
                )
                
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
        
    def train(self, num_epochs=None):
        """Thực hiện toàn bộ quá trình huấn luyện có hỗ trợ dừng khẩn cấp và tiếp tục."""
        epochs = num_epochs or self.config.epochs
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        print(f"Bắt đầu huấn luyện trên thiết bị: {self.device}")
        
        try:
            for epoch in range(1, epochs + 1):
                print(f"\nEpoch {epoch}/{epochs}")
                
                train_loss = self.train_epoch(epoch_idx=epoch)
                val_loss = self.validate()
                
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
                # Lưu checkpoint tốt nhất
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
                    self.save_checkpoint(checkpoint_path, epoch=epoch)
                    print(f"⭐ Đã lưu mô hình tốt nhất (best_model.pth) tại epoch {epoch}")
                    
                # Cập nhật checkpoint cuối epoch
                self.save_checkpoint(os.path.join(self.config.checkpoint_dir, 'last_model.pth'), epoch=epoch)
        except KeyboardInterrupt:
            print("\n" + "=" * 65)
            print("  🛑 ĐÃ TẠM DỪNG HUẤN LUYỆN THEO YÊU CẦU CỦA BẠN (PAUSE)!")
            print("  💾 Toàn bộ trọng số đã được lưu an toàn tại: checkpoints/last_model.pth")
            print("  👉 Ngày mai hoặc bất kỳ lúc nào, bạn chỉ cần chạy thêm cờ: --resume")
            print("     để tiếp tục huấn luyện mà KHÔNG mất đi kết quả đã học!")
            print("=" * 65 + "\n")
            return history

        # Lưu checkpoint hoàn tất
        self.save_checkpoint(os.path.join(self.config.checkpoint_dir, 'last_model.pth'), epoch=epochs)
        return history
        
    def save_checkpoint(self, path, epoch=None, step=None):
        """Lưu trạng thái mô hình (hỗ trợ pause/resume)."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': getattr(self.model, 'config', None),
            'epoch': epoch,
            'step': step,
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path):
        """Tải trạng thái mô hình."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Đã tải checkpoint từ {path}")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from training.config import get_default_config
    from training.dataset import create_dataloaders
    from models.semantic_model import SemanticCommModel
    
    print("=" * 60)
    print("  SEMANTIC COMMUNICATION — HUẤN LUYỆN MÔ HÌNH")
    print("=" * 60)
    
    # 1. Tải cấu hình
    config = get_default_config()
    model_cfg = config['model']
    train_cfg = config['training']
    
    print(f"\n📋 Cấu hình mô hình:")
    print(f"   d_model={model_cfg.d_model}, n_heads={model_cfg.n_heads}, "
          f"n_layers={model_cfg.n_layers}, latent_dim={model_cfg.latent_dim}")
    print(f"   d_ff={model_cfg.d_ff}, max_seq_len={model_cfg.max_seq_len}, "
          f"dropout={model_cfg.dropout}")
    print(f"\n📋 Cấu hình huấn luyện:")
    print(f"   epochs={train_cfg.epochs}, batch_size={train_cfg.batch_size}, "
          f"lr={train_cfg.lr}")
    print(f"   noise_aware={train_cfg.noise_aware}, "
          f"SNR range=[{train_cfg.snr_min}, {train_cfg.snr_max}]")
    
    # 2. Tạo DataLoader
    print("\n📦 Đang tạo DataLoader...")
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(config['data'], batch_size=train_cfg.batch_size)
    model_cfg.vocab_size = tokenizer.vocab_size
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}")
    
    # 3. Khởi tạo mô hình
    print("\n🧠 Đang khởi tạo mô hình SemanticCommModel...")
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
    print(f"   Tổng tham số: {total_params:,}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # 4. Tạo Trainer và bắt đầu huấn luyện
    print("\n🏋️ Bắt đầu huấn luyện...")
    print("-" * 60)
    
    trainer = Trainer(model, train_loader, val_loader, tokenizer, config)
    history = trainer.train(num_epochs=train_cfg.epochs)
    
    print("\n" + "=" * 60)
    print("  ✅ HUẤN LUYỆN HOÀN TẤT!")
    print("=" * 60)
    print(f"\n📊 Kết quả:")
    if history['train_loss']:
        print(f"   Train Loss cuối: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"   Val Loss cuối:   {history['val_loss'][-1]:.4f}")
    if history.get('best_val_loss', float('inf')) < float('inf'):
        print(f"   Best Val Loss:   {history['best_val_loss']:.4f}")
    print(f"\n💾 Checkpoint đã lưu tại: {train_cfg.checkpoint_dir}/")
    print(f"   - best_model.pth  (mô hình tốt nhất)")
    print(f"   - last_model.pth  (mô hình cuối cùng)")
