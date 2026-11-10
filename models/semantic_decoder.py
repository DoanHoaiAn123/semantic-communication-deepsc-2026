import math
import torch
import torch.nn as nn
from typing import Optional
from .semantic_encoder import PositionalEncoding

class TransformerDecoderLayer(nn.Module):
    """
    Một lớp Transformer Decoder bao gồm: Masked Self-Attention -> Cross-Attention -> Feed Forward
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Masked Self-Attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-Attention (dùng memory từ encoder/latent)
        tgt2 = self.cross_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed Forward
        tgt2 = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class SemanticDecoder(nn.Module):
    """
    Bộ giải mã ngữ nghĩa (Semantic Decoder)
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
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Mở rộng latent thành context sequence cho cross-attention. 
        # Độ dài chuỗi context là 4 (chỉ định tuỳ ý).
        self.context_len = 4
        self.latent_proj = nn.Linear(latent_dim, d_model * self.context_len) 
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.out_proj = nn.Linear(d_model, vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.latent_proj.weight)
        nn.init.zeros_(self.latent_proj.bias)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, latent: torch.Tensor, tgt_ids: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Giải mã latent vector thành logits.
        Args:
            latent: Vector ngữ nghĩa từ kênh truyền [B, latent_dim]
            tgt_ids: Sequence đích [B, seq_len]
            tgt_mask: Attention padding mask (boolean tensor)
        """
        B, seq_len = tgt_ids.size()
        
        # Biến đổi latent thành chuỗi nhớ (memory)
        # [B, latent_dim] -> [B, d_model * context_len] -> [B, context_len, d_model]
        memory = self.latent_proj(latent).view(B, self.context_len, self.d_model)
        
        # Causal mask để ngăn mô hình nhìn vào tương lai
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(tgt_ids.device)
            
        x = self.embedding(tgt_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_mask)
            
        logits = self.out_proj(x)
        return logits

    def _greedy_decode(self, latent: torch.Tensor, max_len: int, sos_token_id: int, eos_token_id: int) -> torch.Tensor:
        """
        Giải mã chuỗi tuần tự theo cơ chế tham lam (greedy)
        """
        device = latent.device
        B = latent.size(0)
        
        memory = self.latent_proj(latent).view(B, self.context_len, self.d_model)
        
        tgt_ids = torch.full((B, 1), sos_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            seq_len = tgt_ids.size(1)
            tgt_mask = self._generate_square_subsequent_mask(seq_len).to(device)
            
            x = self.embedding(tgt_ids) * math.sqrt(self.d_model)
            x = self.pos_encoder(x)
            
            for layer in self.layers:
                x = layer(x, memory, tgt_mask)
                
            logits = self.out_proj(x[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
            
        return tgt_ids

    def _beam_search_decode(self, latent: torch.Tensor, max_len: int, sos_token_id: int, eos_token_id: int) -> torch.Tensor:
        """
        Khung làm việc cho Beam Search (hiện dùng lại greedy decode).
        """
        return self._greedy_decode(latent, max_len, sos_token_id, eos_token_id)

    def generate(self, latent: torch.Tensor, max_len: int, sos_token_id: int, eos_token_id: int, strategy: str = 'greedy') -> torch.Tensor:
        """
        Sinh chuỗi dự đoán trong quá trình suy luận (Inference).
        """
        if strategy == 'greedy':
            return self._greedy_decode(latent, max_len, sos_token_id, eos_token_id)
        elif strategy == 'beam_search':
            return self._beam_search_decode(latent, max_len, sos_token_id, eos_token_id)
        else:
            raise ValueError(f"Chiến lược sinh {strategy} không được hỗ trợ.")
