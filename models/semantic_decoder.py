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

    def _generate_square_subsequent_mask(self, sz: int, device=None) -> torch.Tensor:
        """Tạo causal mask kiểu bool (True tại vị trí tương lai bị che)."""
        return torch.triu(torch.ones((sz, sz), dtype=torch.bool, device=device), diagonal=1)

    def forward(self, latent: torch.Tensor, tgt_ids: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Giải mã latent vector thành logits.
        Args:
            latent: Vector ngữ nghĩa từ kênh truyền [B, latent_dim]
            tgt_ids: Sequence đích [B, seq_len]
            tgt_mask: Attention padding mask (boolean tensor)
        """
        B, seq_len = tgt_ids.size()
        if seq_len == 0:
            return torch.zeros((B, 0, self.out_proj.out_features), device=tgt_ids.device)
        
        # Biến đổi latent thành chuỗi nhớ (memory)
        # [B, latent_dim] -> [B, d_model * context_len] -> [B, context_len, d_model]
        memory = self.latent_proj(latent).view(B, self.context_len, self.d_model)
        
        # Causal mask kiểu bool để đồng nhất với tgt_key_padding_mask
        causal_mask = self._generate_square_subsequent_mask(seq_len, device=tgt_ids.device)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(dtype=torch.bool, device=tgt_ids.device)
            all_pad = tgt_mask.all(dim=1)
            if all_pad.any():
                tgt_mask = tgt_mask.clone()
                tgt_mask[all_pad, 0] = False
            
        x = self.embedding(tgt_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_mask)
            
        logits = self.out_proj(x)
        return logits

    def _greedy_decode(self, latent: torch.Tensor, max_len: int, sos_token_id: int, eos_token_id: int) -> torch.Tensor:
        """
        Giải mã chuỗi tuần tự theo cơ chế tham lam (greedy) có dừng sớm tại [EOS].
        """
        device = latent.device
        B = latent.size(0)
        
        memory = self.latent_proj(latent).view(B, self.context_len, self.d_model)
        
        tgt_ids = torch.full((B, 1), sos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for _ in range(max(1, max_len - 1)):
            seq_len = tgt_ids.size(1)
            tgt_mask = self._generate_square_subsequent_mask(seq_len, device=device)
            
            x = self.embedding(tgt_ids) * math.sqrt(self.d_model)
            x = self.pos_encoder(x)
            
            for layer in self.layers:
                x = layer(x, memory, tgt_mask)
                
            logits = self.out_proj(x[:, -1, :])
            next_token = torch.argmax(logits, dim=-1)
            
            # Những mẫu đã sinh [EOS] ở bước trước sẽ điền [PAD] (0) hoặc giữ [EOS]
            next_token = torch.where(finished, torch.zeros_like(next_token), next_token)
            tgt_ids = torch.cat([tgt_ids, next_token.unsqueeze(1)], dim=1)
            
            finished = finished | (next_token == eos_token_id)
            if finished.all():
                break
            
        return tgt_ids

    def _beam_search_decode(self, latent: torch.Tensor, max_len: int, sos_token_id: int, eos_token_id: int, beam_width: int = 3) -> torch.Tensor:
        """
        Giải mã Beam Search thực sự với kích thước chùm tia beam_width.
        """
        if beam_width <= 1:
            return self._greedy_decode(latent, max_len, sos_token_id, eos_token_id)

        device = latent.device
        B = latent.size(0)
        memory_all = self.latent_proj(latent).view(B, self.context_len, self.d_model)

        results = []
        max_steps = max(1, max_len - 1)
        for b in range(B):
            mem_b = memory_all[b:b+1]  # [1, context_len, d_model]
            beams = [(0.0, torch.tensor([[sos_token_id]], dtype=torch.long, device=device), False)]

            for _ in range(max_steps):
                candidates = []
                all_done = True
                for score, seq, is_done in beams:
                    if is_done:
                        candidates.append((score, seq, True))
                        continue
                    all_done = False
                    seq_len = seq.size(1)
                    tgt_mask = self._generate_square_subsequent_mask(seq_len, device=device)
                    x = self.embedding(seq) * math.sqrt(self.d_model)
                    x = self.pos_encoder(x)
                    for layer in self.layers:
                        x = layer(x, mem_b, tgt_mask)
                    log_probs = torch.log_softmax(self.out_proj(x[:, -1, :]), dim=-1).squeeze(0)
                    top_lp, top_idx = torch.topk(log_probs, beam_width)
                    for k in range(beam_width):
                        tok = top_idx[k].item()
                        lp = top_lp[k].item()
                        new_seq = torch.cat([seq, torch.tensor([[tok]], dtype=torch.long, device=device)], dim=1)
                        candidates.append((score + lp, new_seq, tok == eos_token_id))
                if all_done:
                    break
                # Chuẩn hóa điểm theo độ dài để không thiên vị câu quá ngắn
                candidates.sort(key=lambda item: item[0] / (item[1].size(1) ** 0.7), reverse=True)
                beams = candidates[:beam_width]

            best_seq = beams[0][1].squeeze(0)
            results.append(best_seq)

        max_out_len = max(r.size(0) for r in results)
        padded = torch.zeros((B, max_out_len), dtype=torch.long, device=device)
        for b, r in enumerate(results):
            padded[b, :r.size(0)] = r
        return padded

    def generate(self, latent: torch.Tensor, max_len: int = 32,
                 sos_token_id: Optional[int] = None, eos_token_id: Optional[int] = None,
                 strategy: str = 'greedy', beam_width: int = 3,
                 sos_id: Optional[int] = None, eos_id: Optional[int] = None) -> torch.Tensor:
        """
        Sinh chuỗi dự đoán trong quá trình suy luận (Inference).
        Hỗ trợ cả tên tham số sos_token_id/eos_token_id và alias sos_id/eos_id.
        """
        sos = sos_token_id if sos_token_id is not None else (sos_id if sos_id is not None else 1)
        eos = eos_token_id if eos_token_id is not None else (eos_id if eos_id is not None else 2)
        if strategy == 'greedy':
            return self._greedy_decode(latent, max_len, sos, eos)
        elif strategy == 'beam_search':
            return self._beam_search_decode(latent, max_len, sos, eos, beam_width=beam_width)
        else:
            raise ValueError(f"Chiến lược sinh {strategy} không được hỗ trợ.")
