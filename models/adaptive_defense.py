import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from .attack_detector import AttackDetector

class FGSMDenoiser(nn.Module):
    """
    Bộ khử nhiễu chuyên dụng cho tấn công FGSM.
    
    FGSM tạo nhiễu dạng sign(gradient) × ε → nhiễu có biên độ đồng đều.
    Chiến lược: Gradient Smoothing — dùng MLP học cách lọc nhiễu dạng sign.
    """
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # Residual connection: output = latent + learned_correction
        return latent + self.net(latent)


class CWDenoiser(nn.Module):
    """
    Bộ khử nhiễu chuyên dụng cho tấn công C&W.
    
    C&W tạo nhiễu L2 tối ưu, nhỏ nhưng tinh vi.
    Chiến lược: AutoEncoder projection — chiếu latent về manifold sạch qua bottleneck.
    """
    def __init__(self, latent_dim: int = 128, bottleneck_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        compressed = self.encoder(latent)
        reconstructed = self.decoder(compressed)
        # Residual blend: giữ lại 70% manifold sạch và 30% tín hiệu thu
        return 0.7 * reconstructed + 0.3 * latent


class AdaptiveDefense(nn.Module):
    """
    Hệ thống phòng thủ thích ứng — Tự nhận diện loại tấn công
    và áp dụng chiến lược phòng thủ phù hợp.
    
    Pipeline:
      1. AttackDetector phân loại latent → Clean (0) / FGSM (1) / C&W (2)
      2. Chọn denoiser tương ứng:
         - Clean (0): bypass (không can thiệp)
         - FGSM (1): FGSMDenoiser (gradient smoothing)
         - C&W (2): CWDenoiser (autoencoder projection)
      3. Trả về latent đã được phòng thủ
    """
    
    def __init__(self, latent_dim: int = 128, detector: Optional[AttackDetector] = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.detector = detector if detector is not None else AttackDetector(latent_dim=latent_dim)
        self.fgsm_denoiser = FGSMDenoiser(latent_dim=latent_dim)
        self.cw_denoiser = CWDenoiser(latent_dim=latent_dim)
    
    def defend(self, latent: torch.Tensor, force_attack_type: Optional[int] = None) -> torch.Tensor:
        """
        Phương thức tiện ích: Chỉ trả về Tensor latent đã được phòng vệ.
        """
        defended_latent, _ = self.forward(latent, force_attack_type=force_attack_type)
        return defended_latent

    def forward(self, latent: torch.Tensor, force_attack_type: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Phòng thủ thích ứng.
        
        Args:
            latent: [B, latent_dim] — latent vector
            force_attack_type: int or None — nếu chỉ định, bỏ qua detector:
                0 = Clean (bypass), 1 = FGSM denoiser, 2 = C&W denoiser
        
        Returns:
            defended_latent: [B, latent_dim] — latent đã phòng thủ
            detection_info: dict — thông tin nhận diện
        """
        # Bước 1: Phát hiện loại tấn công
        if force_attack_type is not None:
            attack_type = torch.full((latent.size(0),), force_attack_type, 
                                     dtype=torch.long, device=latent.device)
            confidence = torch.ones(latent.size(0), device=latent.device)
            attack_names = [AttackDetector.ATTACK_NAMES.get(force_attack_type, 'Unknown')] * latent.size(0)
        else:
            attack_type, confidence, attack_names = self.detector.predict(latent)
        
        # Bước 2: Áp dụng denoiser phù hợp theo từng sample trong batch
        defended_latent = latent.clone()
        
        fgsm_mask = (attack_type == 1)
        cw_mask = (attack_type == 2)
        
        if fgsm_mask.any():
            defended_latent[fgsm_mask] = self.fgsm_denoiser(latent[fgsm_mask])
        
        if cw_mask.any():
            defended_latent[cw_mask] = self.cw_denoiser(latent[cw_mask])
        
        detection_info = {
            'attack_type': attack_type,
            'confidence': confidence,
            'attack_names': attack_names
        }
        
        return defended_latent, detection_info
    
    def get_detection_summary(self, latent: torch.Tensor) -> Dict[str, Any]:
        """
        Chỉ phát hiện loại tấn công (không phòng thủ).
        Dùng cho hiển thị UI/logging.
        """
        attack_type, confidence, attack_names = self.detector.predict(latent)
        return {
            'attack_type': attack_type,
            'confidence': confidence,
            'attack_names': attack_names,
            'summary': f"{attack_names[0]} (confidence: {confidence[0]:.1%})"
        }
