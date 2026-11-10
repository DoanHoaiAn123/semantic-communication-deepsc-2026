import math
import torch
import torch.nn as nn
from typing import Optional

class PositionalEncoding(nn.Module):
    """
    Thêm thông tin vị trí vào các token embedding (Sinusoidal Positional Encoding)
    """
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hàm forward.
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        Returns:
            Tensor đã cộng positional encoding.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    """
    Một lớp Transformer Encoder bao gồm: Self-Attention -> Add&Norm -> Feed Forward -> Add&Norm
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-Attention
        src2 = self.self_attn(src, src, src, key_padding_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed Forward
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class SemanticEncoder(nn.Module):
    """
    Bộ mã hóa ngữ nghĩa (Semantic Encoder) - Ánh xạ đầu vào sang không gian latent.
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
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Stack các layer Transformer
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Lớp chiếu từ d_model sang latent_dim
        self.projection = nn.Linear(d_model, latent_dim)
        
        # Khởi tạo trọng số
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, src_ids: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Biến đổi chuỗi đầu vào thành vector ngữ nghĩa.
        Args:
            src_ids: Tensor [batch_size, seq_len]
            src_mask: Tensor mask tùy chọn
        Returns:
            latent: Tensor [batch_size, latent_dim]
        """
        # Lấy embedding: [B, S] -> [B, S, d_model]
        x = self.embedding(src_ids) * math.sqrt(self.embedding.embedding_dim)
        
        # Cộng Positional Encoding
        x = self.pos_encoder(x)
        
        # Truyền qua các lớp Transformer Encoder
        for layer in self.layers:
            x = layer(x, src_mask)
            
        # [B, S, d_model] -> [B, d_model] (lấy trung bình theo chiều chuỗi - mean pooling)
        x = x.mean(dim=1)
        
        # Chiếu về không gian latent: [B, latent_dim]
        latent = self.projection(x)
        return latent
