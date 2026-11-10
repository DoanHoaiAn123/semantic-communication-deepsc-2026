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
        - h là hệ số fading tuân theo phân bố Rayleigh: h = sqrt(h_real^2 + h_imag^2) / sqrt(2)
        - n là nhiễu trắng AWGN ~ N(0, P_noise)
        - Máy thu thực hiện cân bằng kênh Zero-Forcing với giả định CSI hoàn hảo:
          y_eq = y / (h + 1e-8) = x + n / (h + 1e-8)
    """
    def __init__(self, with_csi_equalization: bool = True):
        super(RayleighChannel, self).__init__()
        self.with_csi_equalization = with_csi_equalization

    def forward(self, x: torch.Tensor, snr_db: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Truyền tín hiệu qua kênh Rayleigh Fading + AWGN.

        Args:
            x (torch.Tensor): Tín hiệu đầu vào [B, ...].
            snr_db (float hoặc torch.Tensor): Mức SNR tính bằng dB.

        Returns:
            torch.Tensor: Tín hiệu nhận được sau fading và nhiễu (đã cân bằng kênh).
        """
        if isinstance(snr_db, (float, int)) and snr_db > 100:
            return x

        # 1. Sinh hệ số fading Rayleigh h với E[|h|^2] = 1
        # Hai thành phần vuông pha độc lập I và Q tuân theo N(0, 0.5)
        h_r = torch.randn_like(x) * (0.5 ** 0.5)
        h_i = torch.randn_like(x) * (0.5 ** 0.5)
        h = torch.sqrt(h_r ** 2 + h_i ** 2 + 1e-10)

        # 2. Tính công suất tín hiệu và công suất nhiễu theo SNR
        p_signal = torch.mean(x ** 2)
        snr_linear = 10.0 ** (snr_db / 10.0)
        p_noise = p_signal / (snr_linear + 1e-10)
        
        noise = torch.randn_like(x) * torch.sqrt(p_noise)

        # 3. Tín hiệu qua kênh fading
        y = h * x + noise

        # 4. Cân bằng kênh (Equalization)
        if self.with_csi_equalization:
            # Zero-Forcing equalization
            y_recovered = y / (h + 1e-6)
            return y_recovered
        else:
            return y
