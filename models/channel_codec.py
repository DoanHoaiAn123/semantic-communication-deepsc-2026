import torch
import torch.nn as nn

class ChannelEncoder(nn.Module):
    """
    Bộ mã hóa kênh (Channel Encoder)
    Ánh xạ vector ngữ nghĩa sang các symbol kênh và chuẩn hóa công suất.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.linear = nn.Linear(latent_dim, latent_dim)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Chuẩn hóa công suất năng lượng trước khi truyền qua kênh.
        Args:
            latent: Vector ngữ nghĩa [B, latent_dim]
        Returns:
            channel_symbols: Symbols đã được chuẩn hóa công suất
        """
        x = self.linear(latent)
        # Power normalization
        # P = mean(x^2), symbols = x / sqrt(P)
        power = torch.mean(x ** 2, dim=-1, keepdim=True)
        channel_symbols = x / torch.sqrt(power + 1e-8)
        return channel_symbols

class ChannelDecoder(nn.Module):
    """
    Bộ giải mã kênh (Channel Decoder)
    Khôi phục vector ngữ nghĩa từ các symbol kênh bị nhiễu.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, noisy_symbols: torch.Tensor) -> torch.Tensor:
        """
        Khôi phục lại latent vector từ tín hiệu thu được.
        Args:
            noisy_symbols: Tín hiệu bị nhiễu [B, latent_dim]
        Returns:
            recovered_latent: Latent vector đã khôi phục
        """
        recovered_latent = self.net(noisy_symbols)
        return recovered_latent
