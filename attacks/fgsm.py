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

    try:
        with torch.enable_grad():
            latent_adv = latent.clone().detach().requires_grad_(True)

            # 1. Mã hóa kênh và truyền qua kênh truyền
            tx_sig = model.channel_encoder(latent_adv)
            rx_sig = model.channel(tx_sig, snr_db)
            rx_latent = model.channel_decoder(rx_sig)

            # 2. Tính hàm mất mát cần tối đa hóa (maximize loss)
            if target_ids is not None and tgt_ids_input is not None:
                if criterion is None:
                    criterion = nn.CrossEntropyLoss(ignore_index=0)
                # Tự động căn chỉnh nếu người gọi truyền tgt_tensor chưa dịch lệch [B, L]
                if tgt_ids_input.shape == target_ids.shape and torch.equal(tgt_ids_input, target_ids) and tgt_ids_input.size(1) > 1:
                    inp_ids = tgt_ids_input[:, :-1]
                    lbl_ids = target_ids[:, 1:]
                    mask_in = tgt_mask[:, :-1] if (tgt_mask is not None and tgt_mask.size(1) == tgt_ids_input.size(1)) else tgt_mask
                else:
                    inp_ids = tgt_ids_input
                    lbl_ids = target_ids
                    mask_in = tgt_mask
                logits = model.semantic_decoder(rx_latent, inp_ids, mask_in)
                vocab_size = logits.size(-1)
                loss = criterion(logits.reshape(-1, vocab_size), lbl_ids.reshape(-1))
            else:
                # Nếu không có nhãn đích, tối đa hóa năng lượng sai lệch tại đầu thu
                with torch.no_grad():
                    clean_rx = model.channel_decoder(model.channel_encoder(latent.detach()))
                loss = torch.mean((rx_latent - clean_rx) ** 2) + 0.1 * torch.mean(rx_latent ** 2)

            # 3. Tính gradient chỉ theo latent_adv (không làm bẩn .grad của trọng số model)
            grad = torch.autograd.grad(loss, [latent_adv], retain_graph=False, create_graph=False)[0]
            grad_sign = grad.sign()

            perturbation = epsilon * grad_sign
            adv_latent = latent.detach() + perturbation
    finally:
        if was_training:
            model.train()

    return adv_latent.detach(), perturbation.detach()
