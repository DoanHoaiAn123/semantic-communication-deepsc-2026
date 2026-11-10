import numpy as np
from typing import Tuple

class TraditionalPipeline:
    """
    Hệ thống truyền thông truyền thống:
    Text -> Bits (UTF-8) -> BPSK Modulation -> Wireless Channel (AWGN / Rayleigh) -> BPSK Demodulation -> Bits -> Text
    
    Được sử dụng làm hệ quy chiếu (baseline) để so sánh với hệ thống DeepSC.
    """
    def __init__(self):
        pass

    def text_to_bits(self, text: str) -> np.ndarray:
        """Chuyển đổi văn bản thành mảng bit (UTF-8)."""
        bytes_data = text.encode('utf-8')
        bits_str = ''.join([format(b, '08b') for b in bytes_data])
        bits = np.array([int(b) for b in bits_str], dtype=np.uint8)
        return bits

    def bits_to_text(self, bits: np.ndarray) -> str:
        """Giải mã mảng bit trở lại thành văn bản (UTF-8)."""
        n_bits = len(bits)
        if n_bits % 8 != 0:
            padding = 8 - (n_bits % 8)
            bits = np.pad(bits, (0, padding), 'constant', constant_values=0)
            
        bytes_array = np.packbits(bits)
        try:
            return bytes_array.tobytes().decode('utf-8', errors='replace')
        except Exception:
            return ""

    def bpsk_modulate(self, bits: np.ndarray) -> np.ndarray:
        """Điều chế BPSK: bit 0 -> -1.0, bit 1 -> +1.0."""
        symbols = 2.0 * bits - 1.0
        return symbols.astype(np.float32)

    def bpsk_demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """Giải điều chế BPSK với ngưỡng quyết định 0."""
        bits = (symbols > 0).astype(np.uint8)
        return bits

    def awgn_channel(self, symbols: np.ndarray, snr_db: float) -> np.ndarray:
        """Kênh truyền AWGN (Additive White Gaussian Noise)."""
        if snr_db > 100:
            return symbols
            
        p_signal = np.mean(symbols ** 2)
        snr_linear = 10.0 ** (snr_db / 10.0)
        p_noise = p_signal / (2.0 * snr_linear + 1e-10)
        noise = np.random.normal(0, np.sqrt(p_noise), size=symbols.shape)
        return symbols + noise

    def rayleigh_channel(self, symbols: np.ndarray, snr_db: float) -> np.ndarray:
        """Kênh truyền Rayleigh Fading + AWGN (có cân bằng Zero-Forcing)."""
        if snr_db > 100:
            return symbols
            
        # Hệ số fading Rayleigh
        h_r = np.random.normal(0, np.sqrt(0.5), size=symbols.shape)
        h_i = np.random.normal(0, np.sqrt(0.5), size=symbols.shape)
        h = np.sqrt(h_r ** 2 + h_i ** 2 + 1e-10)
        
        # Nhiễu AWGN
        p_signal = np.mean(symbols ** 2)
        snr_linear = 10.0 ** (snr_db / 10.0)
        p_noise = p_signal / (2.0 * snr_linear + 1e-10)
        noise = np.random.normal(0, np.sqrt(p_noise), size=symbols.shape)
        
        # Nhận và cân bằng
        y = h * symbols + noise
        y_eq = y / (h + 1e-6)
        return y_eq

    def calculate_ber(self, original_bits: np.ndarray, received_bits: np.ndarray) -> float:
        """Tính toán tỷ lệ lỗi bit (Bit Error Rate - BER)."""
        min_len = min(len(original_bits), len(received_bits))
        if min_len == 0:
            return 1.0
        orig = original_bits[:min_len]
        recv = received_bits[:min_len]
        errors = np.sum(orig != recv)
        return float(errors) / min_len

    def transmit(self, text: str, snr_db: float, channel_type: str = 'awgn', seed: int = None) -> Tuple[str, float]:
        """
        Thực hiện toàn bộ quá trình truyền văn bản qua kênh truyền thống.
        
        Args:
            text (str): Văn bản gốc cần truyền.
            snr_db (float): SNR của kênh truyền (dB).
            channel_type (str): 'awgn' hoặc 'rayleigh'.
            seed (int, optional): Seed ngẫu nhiên để tái lập kết quả nhất quán.
            
        Returns:
            Tuple[str, float]: (Văn bản nhận được, Tỷ lệ lỗi bit BER)
        """
        if not text:
            return "", 0.0
            
        if seed is not None:
            np.random.seed(abs(int(seed)) % (2**32 - 1))
            
        tx_bits = self.text_to_bits(text)
        tx_symbols = self.bpsk_modulate(tx_bits)
        
        if 'rayleigh' in channel_type.lower():
            rx_symbols = self.rayleigh_channel(tx_symbols, snr_db)
        else:
            rx_symbols = self.awgn_channel(tx_symbols, snr_db)
            
        rx_bits = self.bpsk_demodulate(rx_symbols)
        ber = self.calculate_ber(tx_bits, rx_bits)
        rx_text = self.bits_to_text(rx_bits)
        
        return rx_text, ber
