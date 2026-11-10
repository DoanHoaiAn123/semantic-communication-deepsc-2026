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
            x (torch.Tensor): Tín hiệu đầu vào.
            snr_db (float hoặc torch.Tensor): Tỷ số tín hiệu trên nhiễu (dB).

        Returns:
            torch.Tensor: Tín hiệu sau khi đi qua kênh truyền (đã cộng nhiễu).
        """
        # Nếu SNR rất cao (ví dụ: > 100dB), coi như không có nhiễu
        if isinstance(snr_db, (float, int)) and snr_db > 100:
            return x

        # Tính công suất tín hiệu P_signal = mean(x^2)
        p_signal = torch.mean(x ** 2)
        
        # Chuyển SNR từ dB sang tỷ lệ tuyến tính
        if isinstance(snr_db, torch.Tensor):
            snr_linear = 10.0 ** (snr_db / 10.0)
        else:
            snr_linear = 10.0 ** (snr_db / 10.0)
            
        # Tính công suất nhiễu P_noise = P_signal / SNR
        # Tránh chia cho 0 hoặc giá trị quá nhỏ
        p_noise = p_signal / (snr_linear + 1e-10)
        
        # Sinh nhiễu Gaussian n ~ N(0, P_noise)
        noise = torch.randn_like(x) * torch.sqrt(p_noise)
        
        # Trả về tín hiệu đã cộng nhiễu y = x + n
        return x + noise
