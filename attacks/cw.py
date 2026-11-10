import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple

def cw_attack(model: nn.Module, 
              latent: torch.Tensor, 
              target_ids: Optional[torch.Tensor] = None, 
              tgt_ids_input: Optional[torch.Tensor] = None, 
              tgt_mask: Optional[torch.Tensor] = None, 
              criterion: Optional[nn.Module] = None,
              c: float = 1.0, 
              kappa: float = 0.0, 
              max_iter: int = 100, 
              num_steps: Optional[int] = None,
              lr: float = 0.01, 
              snr_db: float = 10.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tấn công đối kháng C&W (Carlini & Wagner) chuẩn L2 trên không gian ẩn (latent space).
    
    Tối ưu hóa:
        minimize ‖δ‖₂² - c × Loss(latent + δ)
    
    Args:
        model: Mô hình SemanticCommModel
        latent: Latent vector gốc [B, latent_dim]
        target_ids: Ground truth token IDs [B, seq_len]
        tgt_ids_input: Decoder input [B, seq_len]
        tgt_mask: Attention padding mask
        criterion: Hàm loss (mặc định CrossEntropyLoss)
        c: Hằng số đánh đổi (trade-off constant)
        kappa: Margin tin cậy (confidence margin)
        max_iter / num_steps: Số bước tối ưu hóa Adam
        lr: Tốc độ học của biến nhiễu
        snr_db: Mức SNR kênh truyền khi tối ưu
    
    Returns:
        adv_latent: Latent đã bị nhiễu tối ưu [B, latent_dim]
        perturbation: Vector nhiễu delta [B, latent_dim]
    """
    steps = num_steps if num_steps is not None else max_iter
    if steps <= 0:
        return latent.clone().detach(), torch.zeros_like(latent)

    was_training = model.training
    model.eval()

    device = latent.device
    latent_orig = latent.clone().detach()

    # Biến tối ưu w
    w = torch.zeros_like(latent_orig, requires_grad=True, device=device)
    optimizer = optim.Adam([w], lr=lr)
    scale = latent_orig.abs().max().item() + 1e-4

    best_adv_latent = latent_orig.clone()
    best_perturbation = torch.zeros_like(latent_orig)
    best_loss = float('inf')

    if criterion is None:
        criterion = nn.CrossEntropyLoss(ignore_index=0)

    for _ in range(steps):
        optimizer.zero_grad()

        # Dùng tanh để giới hạn biên độ nhiễu hợp lý
        delta = torch.tanh(w) * scale
        adv_latent = latent_orig + delta

        tx_sig = model.channel_encoder(adv_latent)
        rx_sig = model.channel(tx_sig, snr_db)
        rx_latent = model.channel_decoder(rx_sig)

        l2_loss = torch.sum(delta ** 2)

        if target_ids is not None and tgt_ids_input is not None:
            logits = model.semantic_decoder(rx_latent, tgt_ids_input, tgt_mask)
            vocab_size = logits.size(-1)
            ce_loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
            total_loss = l2_loss - c * ce_loss
        else:
            dist = torch.mean((rx_latent - latent_orig) ** 2)
            total_loss = l2_loss - c * dist

        total_loss.backward()
        optimizer.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_adv_latent = adv_latent.detach().clone()
            best_perturbation = delta.detach().clone()

    if was_training:
        model.train()

    return best_adv_latent, best_perturbation
