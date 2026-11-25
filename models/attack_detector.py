import torch
import torch.nn as nn

class AttackDetector(nn.Module):
    """
    Bộ phát hiện loại tấn công dựa trên phân tích latent vector.
    
    Phân tích đặc trưng thống kê của latent vector để nhận diện:
      0 = Clean (không bị tấn công)
      1 = FGSM (Fast Gradient Sign Method)
      2 = C&W (Carlini & Wagner)
    
    Kiến trúc: MLP đa tầng với LayerNorm (an toàn cho cả batch size = 1), ReLU, Dropout.
    """
    
    ATTACK_NAMES = {0: 'Clean', 1: 'FGSM', 2: 'C&W'}
    
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.latent_dim = latent_dim
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
        # Nhánh phân tích đặc trưng thống kê bậc cao (L2 norm, Linf, std, kurtosis, frac_eps...)
        # giúp bảo toàn thông tin biên độ và hình thái đạo hàm sign(grad)*eps vốn bị LayerNorm triệt tiêu
        self.stat_dim = 10
        self.stat_head = nn.Sequential(
            nn.Linear(self.stat_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, num_classes)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _extract_stats(self, latent: torch.Tensor) -> torch.Tensor:
        """Trích xuất 10 đặc trưng thống kê biên độ và phổ lưới 0.15 của latent vector [B, 10]."""
        eps = 1e-6
        mean = latent.mean(dim=-1, keepdim=True)
        centered = latent - mean
        std = latent.std(dim=-1, keepdim=True).clamp(min=eps)
        l2 = torch.norm(latent, p=2, dim=-1, keepdim=True) / 30.0
        l1 = torch.norm(latent, p=1, dim=-1, keepdim=True) / 300.0
        linf = latent.abs().amax(dim=-1, keepdim=True) / 10.0
        z = centered / std
        kurt = (z ** 4).mean(dim=-1, keepdim=True) / 5.0
        # Đặc trưng phổ chu kỳ 0.15 và 0.05 (đặc trưng cho nhiễu cộng đồng nhất dạng ±epsilon của FGSM)
        cos_015 = torch.cos(latent * (2.0 * 3.14159265 / 0.15)).mean(dim=-1, keepdim=True)
        sin_015 = torch.sin(latent * (2.0 * 3.14159265 / 0.15)).mean(dim=-1, keepdim=True)
        diff1 = (latent[:, 1:] - latent[:, :-1]).abs().mean(dim=-1, keepdim=True)
        high_tail = (z.abs() > 2.0).to(latent.dtype).mean(dim=-1, keepdim=True)
        return torch.cat([mean, std / 5.0, l2, l1, linf, kurt, cos_015, sin_015, diff1, high_tail], dim=-1)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Phân loại loại tấn công.
        Args:
            latent: [B, latent_dim]
        Returns:
            logits: [B, num_classes] — logits cho 3 lớp
        """
        base_logits = self.classifier(latent)
        stat_features = self._extract_stats(latent)
        stat_logits = self.stat_head(stat_features)
        return base_logits + stat_logits
    
    def predict(self, latent: torch.Tensor):
        """Dự đoán loại tấn công với confidence.
        Returns:
            attack_type: [B] — index loại tấn công (0/1/2)
            confidence: [B] — xác suất cao nhất
            attack_names: list[str] — tên loại tấn công
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            logits = self.forward(latent)
            probs = torch.softmax(logits, dim=-1)
            confidence, attack_type = probs.max(dim=-1)
            attack_names = [self.ATTACK_NAMES[t.item()] for t in attack_type]
        if was_training:
            self.train()
        return attack_type, confidence, attack_names
