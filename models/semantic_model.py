import torch
import torch.nn as nn
from typing import Optional

from .semantic_encoder import SemanticEncoder
from .semantic_decoder import SemanticDecoder
from .channel_codec import ChannelEncoder, ChannelDecoder
from channel.awgn import AWGNChannel
from channel.rayleigh import RayleighChannel

class SemanticCommModel(nn.Module):
    """
    Mô hình truyền thông ngữ nghĩa End-to-End (DeepSC).
    Bao gồm các khối:
      1. Semantic Encoder (Transformer)
      2. Channel Encoder (Dense + Power Normalization)
      3. Physical Channel (AWGN / Rayleigh Fading)
      4. Channel Decoder (MLP)
      5. Semantic Decoder (Transformer Autoregressive)
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 256, 
                 n_heads: int = 4, 
                 n_layers: int = 2, 
                 d_ff: int = 512, 
                 latent_dim: int = 128, 
                 max_seq_len: int = 64, 
                 dropout: float = 0.1):
        super().__init__()
        
        # 1. Semantic Encoder
        self.semantic_encoder = SemanticEncoder(
            vocab_size, d_model, n_heads, n_layers, d_ff, latent_dim, max_seq_len, dropout
        )
        
        # 2. Channel Encoder
        self.channel_encoder = ChannelEncoder(latent_dim)
        
        # 3. Kênh vật lý mô phỏng
        self.awgn_channel = AWGNChannel()
        self.rayleigh_channel = RayleighChannel()
        self.channel = self.awgn_channel  # Kênh mặc định
        
        # 4. Channel Decoder
        self.channel_decoder = ChannelDecoder(latent_dim)
        
        # 5. Semantic Decoder
        self.semantic_decoder = SemanticDecoder(
            vocab_size, d_model, n_heads, n_layers, d_ff, latent_dim, max_seq_len, dropout
        )

    def set_channel(self, channel_type: str = 'awgn'):
        """Chuyển đổi giữa kênh AWGN và kênh Rayleigh."""
        ct = channel_type.strip().lower()
        if 'rayleigh' in ct:
            self.channel = self.rayleigh_channel
        elif 'awgn' in ct:
            self.channel = self.awgn_channel
        else:
            raise ValueError(f"Loại kênh '{channel_type}' không hợp lệ. Chỉ hỗ trợ 'awgn' hoặc 'rayleigh'.")

    def _resolve_channel(self, channel_type: Optional[str] = None):
        if channel_type is None:
            return self.channel
        ct = channel_type.strip().lower()
        if 'rayleigh' in ct:
            return self.rayleigh_channel
        elif 'awgn' in ct:
            return self.awgn_channel
        else:
            raise ValueError(f"Loại kênh '{channel_type}' không hợp lệ. Chỉ hỗ trợ 'awgn' hoặc 'rayleigh'.")

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor, snr_db: float, 
                src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None,
                channel_type: Optional[str] = None) -> torch.Tensor:
        """
        Lan truyền tiến dùng cho quá trình huấn luyện với Teacher Forcing.
        """
        # Bước 1: Mã hóa ngữ nghĩa
        latent = self.semantic_encoder(src_ids, src_mask)
        
        # Bước 2: Mã hóa kênh (chuẩn hóa năng lượng)
        channel_symbols = self.channel_encoder(latent)
        
        # Bước 3: Truyền qua kênh nhiễu
        chan = self._resolve_channel(channel_type)
        noisy_symbols = chan(channel_symbols, snr_db)
        
        # Bước 4: Giải mã kênh
        recovered_latent = self.channel_decoder(noisy_symbols)
        
        # Bước 5: Giải mã ngữ nghĩa
        logits = self.semantic_decoder(recovered_latent, tgt_ids, tgt_mask)
        
        return logits

    def transmit(self, src_ids: torch.Tensor, snr_db: float, max_len: int, 
                 sos_token_id: int, eos_token_id: int, src_mask: Optional[torch.Tensor] = None,
                 channel_type: Optional[str] = None, strategy: str = 'greedy') -> torch.Tensor:
        """
        Mô phỏng quá trình truyền nhận suy luận (Inference / Evaluation).
        """
        B, S = src_ids.size()
        # Kiểm tra chuỗi rỗng (chỉ chứa [PAD] hoặc chỉ có [SOS], [EOS] mà không có từ nội dung)
        content_mask = (src_ids != 0) & (src_ids != sos_token_id) & (src_ids != eos_token_id)
        if S == 0 or not content_mask.any():
            return torch.full((B, 2), eos_token_id, dtype=torch.long, device=src_ids.device)

        # Bước 1: Mã hóa ngữ nghĩa
        latent = self.semantic_encoder(src_ids, src_mask)
        
        # Bước 2: Mã hóa kênh
        channel_symbols = self.channel_encoder(latent)
        
        # Bước 3: Truyền qua kênh nhiễu (AWGN / Rayleigh)
        chan = self._resolve_channel(channel_type)
        noisy_symbols = chan(channel_symbols, snr_db)
        
        # Bước 4: Giải mã kênh
        recovered_latent = self.channel_decoder(noisy_symbols)
        
        # Bước 5: Giải mã ngữ nghĩa tuần tự
        token_ids = self.semantic_decoder.generate(
            recovered_latent, max_len, sos_token_id, eos_token_id, strategy=strategy
        )
        
        # Nếu một mẫu cụ thể trong batch không có token nội dung, gán chuỗi rỗng [SOS, EOS]
        empty_rows = ~content_mask.any(dim=1)
        if empty_rows.any():
            token_ids[empty_rows] = eos_token_id
            token_ids[empty_rows, 0] = sos_token_id
            
        return token_ids
