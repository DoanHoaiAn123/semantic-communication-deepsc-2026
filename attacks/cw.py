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
    B = latent_orig.size(0)

    try:
        with torch.enable_grad():
            # Biến tối ưu w (khởi tạo nhẹ để phá điểm yên ngựa tại 0)
            w = torch.zeros_like(latent_orig, requires_grad=True, device=device)
            optimizer = optim.Adam([w], lr=lr)
            # Tính scale theo từng mẫu [B, 1] thay vì gộp cả batch
            scale = latent_orig.abs().amax(dim=-1, keepdim=True) + 1e-4

            # Cố định vector nhiễu kênh truyền trong suốt vòng lặp tối ưu C&W để gradient ổn định
            with torch.no_grad():
                tx_clean = model.channel_encoder(latent_orig)
                if isinstance(snr_db, (int, float)) and snr_db > 100:
                    fixed_channel_noise = torch.zeros_like(tx_clean)
                else:
                    snr_val = float(snr_db) if not isinstance(snr_db, torch.Tensor) else float(snr_db.mean().item())
                    snr_lin = 10.0 ** (snr_val / 10.0)
                    p_sig = torch.mean(tx_clean ** 2, dim=-1, keepdim=True)
                    p_n = p_sig / (snr_lin + 1e-10)
                    fixed_channel_noise = torch.randn_like(tx_clean) * torch.sqrt(p_n + 1e-12)

            # Tự động căn chỉnh nếu người gọi truyền tgt_tensor chưa dịch lệch [B, L]
            if target_ids is not None and tgt_ids_input is not None:
                if tgt_ids_input.shape == target_ids.shape and torch.equal(tgt_ids_input, target_ids) and tgt_ids_input.size(1) > 1:
                    inp_ids = tgt_ids_input[:, :-1]
                    lbl_ids = target_ids[:, 1:]
                    mask_in = tgt_mask[:, :-1] if (tgt_mask is not None and tgt_mask.size(1) == tgt_ids_input.size(1)) else tgt_mask
                else:
                    inp_ids = tgt_ids_input
                    lbl_ids = target_ids
                    mask_in = tgt_mask
            else:
                inp_ids, lbl_ids, mask_in = None, None, None

            ce_none = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

            def _compute_sample_attack_loss(rx_lat: torch.Tensor) -> torch.Tensor:
                """Tính loss ngữ nghĩa trên từng mẫu [B]."""
                if inp_ids is not None and lbl_ids is not None:
                    logits = model.semantic_decoder(rx_lat, inp_ids, mask_in)
                    V = logits.size(-1)
                    raw_ce = ce_none(logits.reshape(-1, V), lbl_ids.reshape(-1)).view(B, -1)
                    valid = (lbl_ids != 0).to(raw_ce.dtype)
                    sample_ce = (raw_ce * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1.0)
                    return sample_ce
                else:
                    with torch.no_grad():
                        clean_rx = model.channel_decoder(tx_clean + fixed_channel_noise)
                    return torch.mean((rx_lat - clean_rx) ** 2, dim=-1)

            with torch.no_grad():
                clean_rx0 = model.channel_decoder(tx_clean + fixed_channel_noise)
                baseline_attack_loss = _compute_sample_attack_loss(clean_rx0)

            best_adv_latent = latent_orig.clone()
            best_perturbation = torch.zeros_like(latent_orig)
            best_obj = torch.full((B,), float('inf'), device=device)

            # Target margin cho C&W: mục tiêu đạt mức suy giảm ngữ nghĩa >= baseline + (1.0 + kappa)
            target_margin = baseline_attack_loss + 1.0 + max(0.0, float(kappa))

            for _ in range(steps):
                optimizer.zero_grad()

                delta = torch.tanh(w) * scale
                adv_latent = latent_orig + delta

                tx_sig = model.channel_encoder(adv_latent)
                rx_sig = tx_sig + fixed_channel_noise
                rx_latent = model.channel_decoder(rx_sig)

                # Chuẩn L2 trung bình theo chiều đặc trưng cho từng mẫu [B] (không nhân B)
                l2_per_sample = torch.mean(delta ** 2, dim=-1)
                atk_loss_per_sample = _compute_sample_attack_loss(rx_latent)

                # Hàm mục tiêu C&W chuẩn với margin kappa:
                # Tối thiểu hóa L2 đồng thời đẩy atk_loss vượt ngưỡng target_margin
                hinge_penalty = torch.clamp(target_margin - atk_loss_per_sample, min=0.0)
                obj_per_sample = l2_per_sample + c * (hinge_penalty - 0.25 * atk_loss_per_sample)
                total_loss = obj_per_sample.mean()

                grad_w = torch.autograd.grad(total_loss, [w], retain_graph=False, create_graph=False)[0]
                w.grad = grad_w
                optimizer.step()

                # Đánh giá nghiệm mới sau bước cập nhật w
                with torch.no_grad():
                    new_delta = torch.tanh(w) * scale
                    new_adv = latent_orig + new_delta
                    new_tx = model.channel_encoder(new_adv)
                    new_rx_lat = model.channel_decoder(new_tx + fixed_channel_noise)
                    new_l2 = torch.mean(new_delta ** 2, dim=-1)
                    new_atk = _compute_sample_attack_loss(new_rx_lat)
                    new_hinge = torch.clamp(target_margin - new_atk, min=0.0)
                    new_obj = new_l2 + c * (new_hinge - 0.25 * new_atk)

                    # Chỉ cập nhật cho những mẫu có nhiễu > 0 và cải thiện hàm mục tiêu
                    improved = (new_l2 > 1e-10) & (new_obj < best_obj)
                    if improved.any():
                        best_obj = torch.where(improved, new_obj, best_obj)
                        mask_b = improved.unsqueeze(-1)
                        best_adv_latent = torch.where(mask_b, new_adv, best_adv_latent)
                        best_perturbation = torch.where(mask_b, new_delta, best_perturbation)
    finally:
        if was_training:
            model.train()

    return best_adv_latent.detach(), best_perturbation.detach()
