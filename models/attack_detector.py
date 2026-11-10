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
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Phân loại loại tấn công.
        Args:
            latent: [B, latent_dim]
        Returns:
            logits: [B, num_classes] — logits cho 3 lớp
        """
        return self.classifier(latent)
    
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
