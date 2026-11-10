from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Cấu hình mô hình Semantic Communication."""
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    latent_dim: int = 128
    max_seq_len: int = 64
    dropout: float = 0.1
    vocab_size: int = 10000

@dataclass
class TrainingConfig:
    """Cấu hình cho quá trình huấn luyện."""
    lr: float = 1e-4
    batch_size: int = 64
    epochs: int = 50
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    grad_clip: float = 1.0
    snr_min: float = -5.0
    snr_max: float = 20.0
    noise_aware: bool = True
    checkpoint_dir: str = 'checkpoints'
    results_dir: str = 'results'

@dataclass
class DataConfig:
    """Cấu hình cho dữ liệu — hỗ trợ Hugging Face harouzie/vi_en-translation và opus100.
    
    Dataset 1 (Chính): harouzie/vi_en-translation — chuyên ngành CNTT/Viễn thông
        → Tăng từ 2500 lên 5000 mẫu để cải thiện từ vựng và coverage.
    
    Dataset 2 (Đối chứng): Tatoeba/WMT Lifestyle — đời sống đa miền
        → Bổ sung 100 mẫu từ 10 chủ đề đa dạng, làm giàu ngữ cảnh cho Transformer.
        → Từ vựng đa miền giúp mô hình tổng quát hóa tốt hơn ngoài domain hẹp.
    """
    dataset_name: str = 'harouzie/vi_en-translation'   # Chuẩn theo đề cương chính khóa luận
    max_samples: int = 5000                             # Tăng từ 2500 → 5000 để phong phú từ vựng
    # Danh sách các cặp ngôn ngữ cần tải (nếu dùng opus100)
    language_pairs: list = field(default_factory=lambda: [
        {'src': 'en', 'tgt': 'vi', 'subset': 'en-vi'},   # Anh → Việt
        {'src': 'en', 'tgt': 'zh', 'subset': 'en-zh'},   # Anh → Trung
        {'src': 'vi', 'tgt': 'zh', 'subset': None},       # Việt → Trung (tạo gián tiếp qua pivot)
    ])
    max_samples_per_pair: int = 500                   # Số mẫu tối đa mỗi cặp ngôn ngữ (khi dùng opus100)
    test_ratio: float = 0.1
    val_ratio: float = 0.1
    # Dataset 2: Tatoeba/WMT Lifestyle — bộ đối chứng bổ sung (làm giàu ngữ cảnh)
    use_supplementary_dataset: bool = True            # Bật/tắt dataset đối chứng
    supplementary_dataset_path: str = 'data/processed/dataset2_tatoeba_wmt_info.json'

@dataclass
class AdversarialConfig:
    """Cấu hình cho huấn luyện đối kháng đa tấn công (FGSM + C&W)."""
    # FGSM parameters
    epsilon_fgsm: float = 0.15         # Cường độ tấn công FGSM
    # C&W parameters  
    cw_c: float = 1.0                  # Hằng số confidence của C&W
    cw_kappa: float = 0.0              # Confidence margin
    cw_max_iter: int = 100             # Số bước tối ưu Adam cho C&W
    cw_lr: float = 0.01               # Learning rate cho C&W optimizer
    # Loss weights
    alpha_clean: float = 0.4           # Trọng số cho clean loss
    alpha_fgsm: float = 0.25           # Trọng số cho FGSM adversarial loss
    alpha_cw: float = 0.25             # Trọng số cho C&W adversarial loss
    alpha_detector: float = 0.1        # Trọng số cho detector classification loss
    # Detector điều chỉnh
    detector_hidden_dim: int = 256     # Kích thước ẩn của AttackDetector
    detector_lr: float = 1e-3          # Learning rate riêng cho detector

def get_default_config():
    """Lấy cấu hình mặc định dạng dict."""
    return {
        'model': ModelConfig(),
        'training': TrainingConfig(),
        'data': DataConfig(),
        'adversarial': AdversarialConfig()
    }
