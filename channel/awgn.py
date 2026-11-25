import math
import torch
import torch.nn as nn
from typing import Union

class AWGNChannel(nn.Module):
    """
    Kênh truyền nhiễu trắng cộng (AWGN - Additive White Gaussian Noise).
    Mô phỏng tác động của nhiễu lên tín hiệu truyền qua kênh truyền.
    """
    def __init__(self):
        super(AWGNChannel, self).__init__()

    def forward(self, x: torch.Tensor, snr_db: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Thêm nhiễu AWGN vào tín hiệu.

        Args:
            x (torch.Tensor): Tín hiệu đầu vào [B, ...].
            snr_db (float hoặc torch.Tensor): Tỷ số tín hiệu trên nhiễu (dB).

        Returns:
            torch.Tensor: Tín hiệu sau khi đi qua kênh truyền (đã cộng nhiễu).
        """
        if isinstance(snr_db, torch.Tensor):
            if not torch.isfinite(snr_db).all():
                raise ValueError(f"snr_db chứa giá trị NaN hoặc Inf: {snr_db}")
            if (snr_db > 100).all():
                return x
            snr_t = snr_db.to(dtype=x.dtype, device=x.device)
            while snr_t.ndim < x.ndim:
                snr_t = snr_t.unsqueeze(-1)
            snr_linear = 10.0 ** (torch.clamp(snr_t, max=100.0) / 10.0)
        else:
            snr_val = float(snr_db)
            if not math.isfinite(snr_val):
                raise ValueError(f"snr_db phải là số hữu hạn, nhận được: {snr_db}")
            if snr_val > 100.0:
                return x
            snr_linear = 10.0 ** (snr_val / 10.0)

        # Tính công suất tín hiệu P_signal trên từng mẫu độc lập trong batch
        reduce_dims = tuple(range(1, x.ndim)) if x.ndim > 1 else (0,)
        p_signal = torch.mean(x ** 2, dim=reduce_dims, keepdim=True)
        
        # Tính công suất nhiễu P_noise = P_signal / SNR
        p_noise = p_signal / (snr_linear + 1e-10)
        
        # Sinh nhiễu Gaussian n ~ N(0, P_noise) (cộng 1e-12 để bảo vệ đạo hàm tại 0)
        noise = torch.randn_like(x) * torch.sqrt(p_noise + 1e-12)
        
        # Trả về tín hiệu đã cộng nhiễu y = x + n
        return x + noise
