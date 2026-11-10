"""
Huấn luyện phòng vệ đối kháng (Adversarial Training) cho Semantic Communication (DeepSC).
Tích hợp:
  - Tấn công đối kháng FGSM (Fast Gradient Sign Method)
  - Tấn công đối kháng C&W (Carlini & Wagner)
  - Attack Detector (phân loại loại tấn công)
  - Adaptive Defense (khử nhiễu thích ứng)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import math
import random
import json
from datetime import datetime, timezone
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from attacks import fgsm_attack, cw_attack
from models import AttackDetector, AdaptiveDefense

class AdversarialTrainer:
    """Lớp huấn luyện phòng vệ đối kháng cho mô hình Semantic Communication."""

    def __init__(self, model, train_loader, val_loader, tokenizer, config):
        """
        Args:
            model:         SemanticCommModel
            train_loader:  DataLoader tập train
            val_loader:    DataLoader tập val
            tokenizer:     SimpleTokenizer
            config:        Dict cấu hình tổng thể
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        
        self.config_train = config.get('training') if isinstance(config, dict) else getattr(config, 'training', {})
        self.config_adv = config.get('adversarial', {}) if isinstance(config, dict) else getattr(config, 'adversarial', {})
        self.config_model = config.get('model', {}) if isinstance(config, dict) else getattr(config, 'model', {})

        self.cw_c = getattr(self.config_adv, 'cw_c', 0.1) if not isinstance(self.config_adv, dict) else self.config_adv.get('cw_c', 0.1)
        self.cw_max_iter = getattr(self.config_adv, 'cw_max_iter', 20) if not isinstance(self.config_adv, dict) else self.config_adv.get('cw_max_iter', 20)
        self.alpha_clean = getattr(self.config_adv, 'alpha_clean', 1.0) if not isinstance(self.config_adv, dict) else self.config_adv.get('alpha_clean', 1.0)
        self.alpha_fgsm = getattr(self.config_adv, 'alpha_fgsm', 0.5) if not isinstance(self.config_adv, dict) else self.config_adv.get('alpha_fgsm', 0.5)
        self.alpha_cw = getattr(self.config_adv, 'alpha_cw', 0.5) if not isinstance(self.config_adv, dict) else self.config_adv.get('alpha_cw', 0.5)
        self.alpha_detector = getattr(self.config_adv, 'alpha_detector', 1.0) if not isinstance(self.config_adv, dict) else self.config_adv.get('alpha_detector', 1.0)
        self.epsilon = getattr(self.config_adv, 'epsilon_fgsm', 0.15) if not isinstance(self.config_adv, dict) else self.config_adv.get('epsilon_fgsm', 0.15)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        latent_dim = getattr(self.config_model, 'latent_dim', 128) if not isinstance(self.config_model, dict) else self.config_model.get('latent_dim', 128)
        
        self.detector = AttackDetector(latent_dim=latent_dim).to(self.device)
        self.adaptive_defense = AdaptiveDefense(latent_dim=latent_dim, detector=self.detector).to(self.device)

        # Hàm mất mát
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=0.1
        )
        self.detector_criterion = nn.CrossEntropyLoss()

        # self.adaptive_defense đã chứa self.detector, khử trùng lặp parameters
        all_params = list(self.model.parameters()) + list(self.adaptive_defense.parameters())
        unique_params = list(dict.fromkeys(all_params))
        lr = getattr(self.config_train, 'lr', 1e-4) if not isinstance(self.config_train, dict) else self.config_train.get('lr', 1e-4)
        weight_decay = getattr(self.config_train, 'weight_decay', 1e-4) if not isinstance(self.config_train, dict) else self.config_train.get('weight_decay', 1e-4)
        
        self.optimizer = AdamW(unique_params, lr=lr, weight_decay=weight_decay)

        # Scheduler
        epochs = getattr(self.config_train, 'epochs', 10) if not isinstance(self.config_train, dict) else self.config_train.get('epochs', 10)
        total_steps = len(self.train_loader) * max(1, epochs)
        self.scheduler = self._get_scheduler(total_steps)

        self.checkpoint_dir = getattr(self.config_train, 'checkpoint_dir', 'checkpoints') if not isinstance(self.config_train, dict) else self.config_train.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_every_steps = self._training_value('save_every_steps', 250)
        self.progress_path = self._training_value('progress_path', None)
        self.current_epoch = 0
        self.epoch_step = 0
        self.global_step = 0
        self.processed_samples = 0
        self.total_epochs = epochs
        self.last_losses = {}
        self.run_metadata = self._training_value('run_metadata', {})

    def _training_value(self, name, default):
        if isinstance(self.config_train, dict):
            return self.config_train.get(name, default)
        return getattr(self.config_train, name, default)

    def _write_progress(self, status):
        if not self.progress_path:
            return
        progress = {
            'status': status,
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'epoch': self.current_epoch,
            'epochs': self.total_epochs,
            'step': self.epoch_step,
            'steps_per_epoch': len(self.train_loader),
            'train_samples': len(self.train_loader.dataset),
            'processed_samples': self.processed_samples,
            'global_step': self.global_step,
            'losses': self.last_losses,
            'run_metadata': self.run_metadata,
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.progress_path)), exist_ok=True)
        temporary_path = str(self.progress_path) + '.tmp'
        with open(temporary_path, 'w', encoding='utf-8') as handle:
            json.dump(progress, handle, ensure_ascii=False, indent=2)
        os.replace(temporary_path, self.progress_path)

    def _get_scheduler(self, total_steps):
        warmup_steps = getattr(self.config_train, 'warmup_steps', 100) if not isinstance(self.config_train, dict) else self.config_train.get('warmup_steps', 100)
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self):
        """Huấn luyện 1 epoch đa tấn công."""
        self.model.train()
        self.detector.train()
        self.adaptive_defense.train()
        
        total_loss = 0.0
        total_loss_clean = 0.0
        total_loss_fgsm = 0.0
        total_loss_cw = 0.0
        total_loss_detector = 0.0

        progress_bar = tqdm(self.train_loader, desc='Adversarial Training')
        for batch in progress_bar:
            src_ids = batch['src_ids'].to(self.device)
            tgt_ids = batch['tgt_ids'].to(self.device)
            src_mask = batch['src_mask'].to(self.device)
            tgt_mask = batch['tgt_mask'].to(self.device)

            noise_aware = getattr(self.config_train, 'noise_aware', True) if not isinstance(self.config_train, dict) else self.config_train.get('noise_aware', True)
            if noise_aware:
                snr_min = getattr(self.config_train, 'snr_min', -5.0) if not isinstance(self.config_train, dict) else self.config_train.get('snr_min', -5.0)
                snr_max = getattr(self.config_train, 'snr_max', 20.0) if not isinstance(self.config_train, dict) else self.config_train.get('snr_max', 20.0)
                snr_db = random.uniform(snr_min, snr_max)
            else:
                snr_db = 20.0

            input_tgt_ids = tgt_ids[:, :-1]
            target_ids = tgt_ids[:, 1:]
            current_tgt_mask = tgt_mask[:, :-1] if tgt_mask is not None else None

            self.optimizer.zero_grad()

            # 1. Trích xuất latent vector gốc
            clean_latent = self.model.semantic_encoder(src_ids, src_mask)
            
            # Clean forward & loss
            clean_defended = self.adaptive_defense.defend(clean_latent, force_attack_type=0)
            ch_sym = self.model.channel_encoder(clean_defended)
            noisy_sym = self.model.channel(ch_sym, snr_db)
            rec_latent = self.model.channel_decoder(noisy_sym)
            logits_clean = self.model.semantic_decoder(rec_latent, input_tgt_ids, current_tgt_mask)
            loss_clean = self.criterion(logits_clean.reshape(-1, logits_clean.size(-1)), target_ids.reshape(-1))

            # 2. Sinh nhiễu đối kháng FGSM
            fgsm_latent, _ = fgsm_attack(
                model=self.model,
                latent=clean_latent,
                target_ids=target_ids,
                tgt_ids_input=input_tgt_ids,
                tgt_mask=current_tgt_mask,
                criterion=self.criterion,
                epsilon=self.epsilon,
                snr_db=10.0
            )
            
            # 3. Sinh nhiễu đối kháng C&W (số bước nhỏ trong training để tăng tốc)
            cw_latent, _ = cw_attack(
                model=self.model,
                latent=clean_latent,
                target_ids=target_ids,
                tgt_ids_input=input_tgt_ids,
                tgt_mask=current_tgt_mask,
                criterion=self.criterion,
                c=self.cw_c,
                max_iter=min(15, self.cw_max_iter),
                snr_db=10.0
            )

            # Keep perturbations fixed while training the encoder as well as
            # the decoder and defense on the adversarial objective.
            fgsm_latent = clean_latent + (fgsm_latent - clean_latent).detach()
            cw_latent = clean_latent + (cw_latent - clean_latent).detach()
            
            # 4. Huấn luyện Detector phân loại (0: Clean, 1: FGSM, 2: C&W)
            combined_latent = torch.cat([clean_latent.detach(), fgsm_latent.detach(), cw_latent.detach()], dim=0)
            labels = torch.cat([
                torch.zeros(clean_latent.size(0), dtype=torch.long, device=self.device),
                torch.ones(fgsm_latent.size(0), dtype=torch.long, device=self.device),
                torch.full((cw_latent.size(0),), 2, dtype=torch.long, device=self.device)
            ])
            
            detector_logits = self.detector(combined_latent)
            loss_detector = self.detector_criterion(detector_logits, labels)
            
            # 5. Khử nhiễu và tính loss cho FGSM (force_attack_type=1)
            defended_fgsm = self.adaptive_defense.defend(fgsm_latent, force_attack_type=1)
            adv_ch_fgsm = self.model.channel_encoder(defended_fgsm)
            adv_noisy_fgsm = self.model.channel(adv_ch_fgsm, snr_db)
            adv_rec_fgsm = self.model.channel_decoder(adv_noisy_fgsm)
            logits_fgsm = self.model.semantic_decoder(adv_rec_fgsm, input_tgt_ids, current_tgt_mask)
            loss_fgsm = self.criterion(logits_fgsm.reshape(-1, logits_fgsm.size(-1)), target_ids.reshape(-1))
            
            # 6. Khử nhiễu và tính loss cho C&W (force_attack_type=2)
            defended_cw = self.adaptive_defense.defend(cw_latent, force_attack_type=2)
            adv_ch_cw = self.model.channel_encoder(defended_cw)
            adv_noisy_cw = self.model.channel(adv_ch_cw, snr_db)
            adv_rec_cw = self.model.channel_decoder(adv_noisy_cw)
            logits_cw = self.model.semantic_decoder(adv_rec_cw, input_tgt_ids, current_tgt_mask)
            loss_cw = self.criterion(logits_cw.reshape(-1, logits_cw.size(-1)), target_ids.reshape(-1))

            # 7. Tổng hợp hàm mất mát
            loss_total = (self.alpha_clean * loss_clean + 
                          self.alpha_fgsm * loss_fgsm + 
                          self.alpha_cw * loss_cw + 
                          self.alpha_detector * loss_detector)
            
            # Attack generation backpropagates through the model to construct
            # perturbations. Only the training objective may update its weights.
            self.optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            
            grad_clip = getattr(self.config_train, 'grad_clip', 1.0) if not isinstance(self.config_train, dict) else self.config_train.get('grad_clip', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.detector.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.adaptive_defense.parameters(), grad_clip)
            
            self.optimizer.step()
            self.scheduler.step()
            self.epoch_step += 1
            self.global_step += 1
            self.processed_samples += src_ids.size(0)
            self.last_losses = {
                'total': loss_total.item(),
                'clean': loss_clean.item(),
                'fgsm': loss_fgsm.item(),
                'cw': loss_cw.item(),
                'detector': loss_detector.item(),
            }

            if self.save_every_steps and self.global_step % self.save_every_steps == 0:
                self.save_checkpoint(os.path.join(self.checkpoint_dir, 'last_robust_model.pth'))
            if self.epoch_step == 1 or self.global_step % 25 == 0:
                self._write_progress('training')

            total_loss += loss_total.item()
            total_loss_clean += loss_clean.item()
            total_loss_fgsm += loss_fgsm.item()
            total_loss_cw += loss_cw.item()
            total_loss_detector += loss_detector.item()
            
            progress_bar.set_postfix({
                'tot': f'{loss_total.item():.3f}',
                'cln': f'{loss_clean.item():.3f}',
                'fgsm': f'{loss_fgsm.item():.3f}',
                'cw': f'{loss_cw.item():.3f}',
                'det': f'{loss_detector.item():.3f}'
            })

        n = len(self.train_loader)
        return {
            'total': total_loss / n,
            'clean': total_loss_clean / n,
            'fgsm': total_loss_fgsm / n,
            'cw': total_loss_cw / n,
            'detector': total_loss_detector / n
        }

    def validate(self):
        """Đánh giá trên tập Validation."""
        self.model.eval()
        self.detector.eval()
        self.adaptive_defense.eval()
        
        total_loss_clean = 0.0
        total_loss_fgsm = 0.0
        total_loss_cw = 0.0
        total_loss_detector = 0.0

        for batch in tqdm(self.val_loader, desc='Validation'):
            src_ids = batch['src_ids'].to(self.device)
            tgt_ids = batch['tgt_ids'].to(self.device)
            src_mask = batch['src_mask'].to(self.device)
            tgt_mask = batch['tgt_mask'].to(self.device)

            snr_db = random.choice([-5, 0, 5, 10, 15, 20])
            input_tgt_ids = tgt_ids[:, :-1]
            target_ids = tgt_ids[:, 1:]
            current_tgt_mask = tgt_mask[:, :-1] if tgt_mask is not None else None

            with torch.no_grad():
                latent = self.model.semantic_encoder(src_ids, src_mask)
                clean_defended = self.adaptive_defense.defend(latent, force_attack_type=0)
                ch_sym = self.model.channel_encoder(clean_defended)
                noisy = self.model.channel(ch_sym, snr_db)
                rec = self.model.channel_decoder(noisy)
                logits = self.model.semantic_decoder(rec, input_tgt_ids, current_tgt_mask)
                loss_clean = self.criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
                total_loss_clean += loss_clean.item()

            fgsm_latent, _ = fgsm_attack(
                model=self.model,
                latent=latent,
                target_ids=target_ids,
                tgt_ids_input=input_tgt_ids,
                tgt_mask=current_tgt_mask,
                criterion=self.criterion,
                epsilon=self.epsilon,
                snr_db=10.0
            )
            cw_latent, _ = cw_attack(
                model=self.model,
                latent=latent,
                target_ids=target_ids,
                tgt_ids_input=input_tgt_ids,
                tgt_mask=current_tgt_mask,
                criterion=self.criterion,
                c=self.cw_c,
                max_iter=5,
                snr_db=10.0
            )

            with torch.no_grad():
                combined_latent = torch.cat([latent.detach(), fgsm_latent.detach(), cw_latent.detach()], dim=0)
                labels = torch.cat([
                    torch.zeros(latent.size(0), dtype=torch.long, device=self.device),
                    torch.ones(fgsm_latent.size(0), dtype=torch.long, device=self.device),
                    torch.full((cw_latent.size(0),), 2, dtype=torch.long, device=self.device)
                ])
                detector_logits = self.detector(combined_latent)
                loss_detector = self.detector_criterion(detector_logits, labels)
                total_loss_detector += loss_detector.item()

                defended_fgsm = self.adaptive_defense.defend(fgsm_latent, force_attack_type=1)
                adv_ch_fgsm = self.model.channel_encoder(defended_fgsm)
                adv_noisy_fgsm = self.model.channel(adv_ch_fgsm, snr_db)
                adv_rec_fgsm = self.model.channel_decoder(adv_noisy_fgsm)
                adv_logits_fgsm = self.model.semantic_decoder(adv_rec_fgsm, input_tgt_ids, current_tgt_mask)
                loss_fgsm = self.criterion(adv_logits_fgsm.reshape(-1, adv_logits_fgsm.size(-1)), target_ids.reshape(-1))
                total_loss_fgsm += loss_fgsm.item()
                
                defended_cw = self.adaptive_defense.defend(cw_latent, force_attack_type=2)
                adv_ch_cw = self.model.channel_encoder(defended_cw)
                adv_noisy_cw = self.model.channel(adv_ch_cw, snr_db)
                adv_rec_cw = self.model.channel_decoder(adv_noisy_cw)
                adv_logits_cw = self.model.semantic_decoder(adv_rec_cw, input_tgt_ids, current_tgt_mask)
                loss_cw = self.criterion(adv_logits_cw.reshape(-1, adv_logits_cw.size(-1)), target_ids.reshape(-1))
                total_loss_cw += loss_cw.item()

        n = len(self.val_loader)
        return {
            'clean': total_loss_clean / n,
            'fgsm': total_loss_fgsm / n,
            'cw': total_loss_cw / n,
            'detector': total_loss_detector / n
        }

    def train(self, num_epochs=None):
        """Thực hiện toàn bộ quá trình Adversarial Training."""
        epochs = num_epochs if num_epochs is not None else self._training_value('epochs', 10)
        if epochs < 1:
            raise ValueError('num_epochs must be at least 1')
        self.total_epochs = epochs
        # Rebuild the schedule for the requested adversarial duration, which may
        # differ from the preceding standard training configuration.
        self.scheduler = self._get_scheduler(len(self.train_loader) * epochs)
        best_val_loss = float('inf')
        history = {
            'train_total': [], 'train_clean': [], 'train_fgsm': [], 'train_cw': [], 'train_detector': [],
            'val_clean': [], 'val_fgsm': [], 'val_cw': [], 'val_detector': []
        }

        print(f"{'='*60}")
        print(f"  ADVERSARIAL TRAINING — FGSM (eps={self.epsilon}) & C&W (c={self.cw_c})")
        print(f"  Device: {self.device}")
        print(f"{'='*60}")

        try:
            for epoch in range(1, epochs + 1):
                self.current_epoch = epoch
                self.epoch_step = 0
                self._write_progress('training')
                print(f"\nEpoch {epoch}/{epochs}")

                train_losses = self.train_epoch()
                self._write_progress('validating')
                val_losses = self.validate()

                history['train_total'].append(train_losses['total'])
                history['train_clean'].append(train_losses['clean'])
                history['train_fgsm'].append(train_losses['fgsm'])
                history['train_cw'].append(train_losses['cw'])
                history['train_detector'].append(train_losses['detector'])

                history['val_clean'].append(val_losses['clean'])
                history['val_fgsm'].append(val_losses['fgsm'])
                history['val_cw'].append(val_losses['cw'])
                history['val_detector'].append(val_losses['detector'])

                print(f"  Train -> Tot: {train_losses['total']:.3f} | Cln: {train_losses['clean']:.3f} | FGSM: {train_losses['fgsm']:.3f} | CW: {train_losses['cw']:.3f} | Det: {train_losses['detector']:.3f}")
                print(f"  Val   -> Cln: {val_losses['clean']:.3f} | FGSM: {val_losses['fgsm']:.3f} | CW: {val_losses['cw']:.3f} | Det: {val_losses['detector']:.3f}")

                val_adv_loss = val_losses['fgsm'] + val_losses['cw']
                if val_adv_loss < best_val_loss:
                    best_val_loss = val_adv_loss
                    path = os.path.join(self.checkpoint_dir, 'best_robust_model.pth')
                    self.save_checkpoint(path)
                    print(f"  [OK] Da luu model robust tot nhat tai epoch {epoch}")

                self.save_checkpoint(os.path.join(self.checkpoint_dir, 'last_robust_model.pth'))
                self._write_progress('epoch_complete')
        except KeyboardInterrupt:
            self.save_checkpoint(os.path.join(self.checkpoint_dir, 'last_robust_model.pth'))
            self._write_progress('interrupted')
            raise

        self._write_progress('completed')
        print(f"\n{'='*60}")
        print(f"  [HOAN TAT] ADVERSARIAL TRAINING DA SAN SANG!")
        print(f"{'='*60}")
        return history

    def save_checkpoint(self, path, epoch=None, step=None):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'detector_state_dict': self.detector.state_dict(),
            'adaptive_defense_state_dict': self.adaptive_defense.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch if epoch is None else epoch,
            'step': self.epoch_step if step is None else step,
            'epochs': self.total_epochs,
            'global_step': self.global_step,
            'completed_samples': self.processed_samples,
            'processed_samples': self.processed_samples,
            'train_samples': len(self.train_loader.dataset),
            'steps_per_epoch': len(self.train_loader),
            'losses': self.last_losses,
            'run_metadata': self.run_metadata,
        }
        temporary_path = str(path) + '.tmp'
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'detector_state_dict' in checkpoint:
            self.detector.load_state_dict(checkpoint['detector_state_dict'])
        if 'adaptive_defense_state_dict' in checkpoint:
            self.adaptive_defense.load_state_dict(checkpoint['adaptive_defense_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Da tai checkpoint tu {path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from training.config import get_default_config
    from training.dataset import create_dataloaders
    from models.semantic_model import SemanticCommModel

    config = get_default_config()
    model_cfg = config['model']
    train_cfg = config['training']

    print("Dang chuan bi du lieu va khoi tao Trainer...")
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
        config['data'], batch_size=train_cfg.batch_size
    )
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

    trainer = AdversarialTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=config
    )
    print("Khoi tao AdversarialTrainer thanh cong 100%!")
