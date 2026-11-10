import torch
import torch.nn as nn
from typing import Optional, Tuple

def fgsm_attack(model: nn.Module, 
                latent: torch.Tensor, 
                target_ids: Optional[torch.Tensor] = None, 
                tgt_ids_input: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None, 
                criterion: Optional[nn.Module] = None, 
                epsilon: float = 0.15, 
                snr_db: float = 10.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tấn công đối kháng FGSM (Fast Gradient Sign Method) trên không gian ẩn (latent space).
    
    Công thức:
        latent_adv = latent + ε × sign(∇_latent L)
    
    Args:
        model: Mô hình SemanticCommModel
        latent: Latent vector gốc [B, latent_dim]
        target_ids: Ground truth token IDs [B, seq_len] (tuỳ chọn)
        tgt_ids_input: Decoder input token IDs [B, seq_len] (tuỳ chọn)
        tgt_mask: Attention padding mask (tuỳ chọn)
        criterion: Hàm loss (mặc định nn.CrossEntropyLoss)
        epsilon: Cường độ nhiễu ε
        snr_db: SNR của kênh truyền khi tấn công
    
    Returns:
        adv_latent: Latent vector đã bị tấn công [B, latent_dim]
        perturbation: Vector nhiễu delta [B, latent_dim]
    """
    if epsilon <= 0.0:
        return latent.clone().detach(), torch.zeros_like(latent)

    was_training = model.training
    model.eval()

    latent_adv = latent.clone().detach().requires_grad_(True)

    # 1. Mã hóa kênh và truyền qua kênh AWGN
    tx_sig = model.channel_encoder(latent_adv)
    rx_sig = model.channel(tx_sig, snr_db)
    rx_latent = model.channel_decoder(rx_sig)

    # 2. Tính hàm mất mát cần tối đa hóa (maximize loss)
    if target_ids is not None and tgt_ids_input is not None:
        if criterion is None:
            criterion = nn.CrossEntropyLoss(ignore_index=0)
        logits = model.semantic_decoder(rx_latent, tgt_ids_input, tgt_mask)
        vocab_size = logits.size(-1)
        loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
    else:
        # Nếu không có nhãn đích, tối đa hóa khoảng cách sai lệch latent (untargeted)
        loss = torch.mean((rx_latent - latent) ** 2)

    # 3. Tính gradient theo latent_adv
    model.zero_grad()
    loss.backward()

    # 4. Sinh vector nhiễu theo dấu gradient
    if latent_adv.grad is not None:
        grad_sign = latent_adv.grad.sign()
    else:
        grad_sign = torch.sign(torch.randn_like(latent_adv))

    perturbation = epsilon * grad_sign
    adv_latent = latent + perturbation

    if was_training:
        model.train()

    return adv_latent.detach(), perturbation.detach()
