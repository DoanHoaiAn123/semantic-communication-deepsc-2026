import math
import torch
import torch.nn as nn
from typing import Union

class RayleighChannel(nn.Module):
    """
    Kênh truyền fading Rayleigh kết hợp nhiễu trắng cộng Gauss (AWGN).
    Mô phỏng hiện tượng lan truyền đa đường (Multipath Fading) trong truyền thông không dây.
    
    Mô hình:
        y = h * x + n
    Trong đó:
        - h là hệ số fading tuân theo phân bố Rayleigh với E[h^2] = 1
        - n là nhiễu trắng AWGN ~ N(0, P_noise)
        - Máy thu thực hiện cân bằng kênh MMSE (Minimum Mean Square Error) với CSI hoàn hảo:
          x_hat = (h * y) / (h^2 + P_noise / (P_signal + 1e-10))
          giúp triệt tiêu hiện tượng nổ phương sai nhiễu khi fading sâu (h -> 0).
    """
    def __init__(self, with_csi_equalization: bool = True, equalization_mode: str = 'mmse'):
        super(RayleighChannel, self).__init__()
        self.with_csi_equalization = with_csi_equalization
        self.equalization_mode = equalization_mode

    def forward(self, x: torch.Tensor, snr_db: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Truyền tín hiệu qua kênh Rayleigh Fading + AWGN.

        Args:
            x (torch.Tensor): Tín hiệu đầu vào [B, ...].
            snr_db (float hoặc torch.Tensor): Mức SNR tính bằng dB.

        Returns:
            torch.Tensor: Tín hiệu nhận được sau fading và nhiễu (đã cân bằng kênh).
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

        # 1. Sinh hệ số fading Rayleigh h với E[|h|^2] = 1
        h_r = torch.randn_like(x) * (0.5 ** 0.5)
        h_i = torch.randn_like(x) * (0.5 ** 0.5)
        h = torch.sqrt(h_r ** 2 + h_i ** 2 + 1e-10)

        # 2. Tính công suất tín hiệu theo từng mẫu và công suất nhiễu theo SNR
        reduce_dims = tuple(range(1, x.ndim)) if x.ndim > 1 else (0,)
        p_signal = torch.mean(x ** 2, dim=reduce_dims, keepdim=True)
        p_noise = p_signal / (snr_linear + 1e-10)
        
        noise = torch.randn_like(x) * torch.sqrt(p_noise + 1e-12)

        # 3. Tín hiệu qua kênh fading
        y = h * x + noise

        # 4. Cân bằng kênh (Equalization)
        if self.with_csi_equalization:
            if self.equalization_mode == 'zf':
                return y / (h + 1e-3)
            # MMSE equalization: loại bỏ hiện tượng khuếch đại nhiễu 13x-56x tại h -> 0
            noise_to_sig = p_noise / (p_signal + 1e-10)
            y_recovered = (h * y) / (h ** 2 + noise_to_sig + 1e-8)
            return y_recovered
        else:
            return y
