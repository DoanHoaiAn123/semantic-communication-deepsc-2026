import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import pandas as pd
from tqdm import tqdm

from models.semantic_model import SemanticCommModel
from channel.traditional import TraditionalPipeline
from evaluation.metrics import calculate_bleu, calculate_bertscore, calculate_semantic_similarity
from evaluation.visualization import plot_all_results
from training.config import get_default_config
from training.dataset import create_dataloaders
from training.train import Trainer

def run_experiment_1(model, test_loader, tokenizer, device, config):
    """Experiment 1: Truyền thông ở điều kiện không nhiễu (SNR cao = 20dB)."""
    print("\n--- Chay Experiment 1: Dieu kien khong nhieu (SNR=20dB) ---")
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Exp 1"):
            src_ids = batch['src_ids'].to(device)
            src_text = batch['src_text']
            tgt_text = batch['tgt_text']
            
            output_ids = model.transmit(
                src_ids=src_ids,
                snr_db=20.0,
                max_len=32,
                sos_token_id=tokenizer.sos_id,
                eos_token_id=tokenizer.eos_id,
                src_mask=batch['src_mask'].to(device)
            )
            
            for i in range(len(src_text)):
                pred_text = tokenizer.decode(output_ids[i].cpu().tolist())
                results.append({
                    'snr': 20.0,
                    'src_text': src_text[i],
                    'tgt_text': tgt_text[i],
                    'pred_text': pred_text,
                    'method': 'Semantic'
                })
                
    return results

def run_experiment_2(model, test_loader, tokenizer, device, config, snr_levels):
    """Experiment 2: Đánh giá hiệu năng dưới kênh AWGN với nhiều mức SNR."""
    print(f"\n--- Chay Experiment 2: Dieu kien AWGN (SNR: {snr_levels}) ---")
    model.eval()
    results = []
    
    with torch.no_grad():
        for snr in snr_levels:
            print(f"Danh gia tai SNR = {snr}dB")
            for batch in tqdm(test_loader, desc=f"SNR={snr}dB"):
                src_ids = batch['src_ids'].to(device)
                src_text = batch['src_text']
                tgt_text = batch['tgt_text']
                
                output_ids = model.transmit(
                    src_ids=src_ids,
                    snr_db=snr,
                    max_len=32,
                    sos_token_id=tokenizer.sos_id,
                    eos_token_id=tokenizer.eos_id,
                    src_mask=batch['src_mask'].to(device)
                )
                
                for i in range(len(src_text)):
                    pred_text = tokenizer.decode(output_ids[i].cpu().tolist())
                    results.append({
                        'snr': float(snr),
                        'src_text': src_text[i],
                        'tgt_text': tgt_text[i],
                        'pred_text': pred_text,
                        'method': 'Semantic'
                    })
    return results

def run_experiment_3(traditional_pipeline, test_loader, snr_levels):
    """Experiment 3: So sánh với hệ thống truyền thông truyền thống (BPSK + Huffman)."""
    print("\n--- Chay Experiment 3: Baseline Truyen thong ---")
    results = []
    
    for snr in snr_levels:
        print(f"Truyen thong tai SNR = {snr}dB")
        for batch in tqdm(test_loader, desc=f"Trad SNR={snr}dB"):
            src_text = batch['src_text']
            tgt_text = batch['tgt_text']
            
            for i in range(len(src_text)):
                received_text, ber = traditional_pipeline.transmit(src_text[i], snr_db=snr)
                results.append({
                    'snr': float(snr),
                    'src_text': src_text[i],
                    'tgt_text': tgt_text[i],
                    'pred_text': received_text,
                    'method': 'Traditional',
                    'ber_traditional': ber
                })
    return results

def main():
    print("Khoi chay he thong thuc nghiem Semantic Communication...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_default_config()
    
    os.makedirs(config['training'].results_dir, exist_ok=True)
    os.makedirs(config['training'].checkpoint_dir, exist_ok=True)
    
    # 1. Tạo Dataloader
    print("Tao DataLoader...")
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(config['data'], batch_size=config['training'].batch_size)
    config['model'].vocab_size = tokenizer.vocab_size
    
    # 2. Khởi tạo Models
    print("Khoi tao mo hinh Semantic...")
    model_cfg = config['model']
    model = SemanticCommModel(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        d_ff=model_cfg.d_ff,
        latent_dim=model_cfg.latent_dim,
        max_seq_len=model_cfg.max_seq_len,
        dropout=model_cfg.dropout
    ).to(device)
    
    # Tìm kiếm checkpoint ưu tiên best_robust_model.pth hoặc best_model.pth
    ckpt_candidates = [
        os.path.join(config['training'].checkpoint_dir, 'best_robust_model.pth'),
        os.path.join(config['training'].checkpoint_dir, 'best_model.pth')
    ]
    ckpt_path = next((p for p in ckpt_candidates if os.path.exists(p)), None)
    if ckpt_path:
        print(f"Tai mo hinh da huan luyen tu {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Khong tim thay checkpoint. Tien hanh huan luyen nhanh 1 epoch...")
        trainer = Trainer(model, train_loader, val_loader, tokenizer, config)
        trainer.train(num_epochs=1)
            
    traditional_pipeline = TraditionalPipeline()
    snr_levels = [-5, 0, 5, 10, 15, 20]
    
    # Chạy các thí nghiệm
    all_results = []
    all_results.extend(run_experiment_1(model, test_loader, tokenizer, device, config))
    all_results.extend(run_experiment_2(model, test_loader, tokenizer, device, config, snr_levels))
    all_results.extend(run_experiment_3(traditional_pipeline, test_loader, snr_levels))
    
    # Đánh giá Metrics
    print("\nTinh toan cac chi so do do (Metrics)...")
    final_data = []
    for res in tqdm(all_results, desc="Tinh toan Metrics"):
        try:
            bleu = calculate_bleu(res['tgt_text'], res['pred_text'])
            bertscore_result = calculate_bertscore(res['tgt_text'], res['pred_text'])
            if isinstance(bertscore_result, dict):
                bertscore = bertscore_result.get('F1', bertscore_result.get('f1', 0.0))
            else:
                bertscore = float(bertscore_result)
            sem_sim = calculate_semantic_similarity(res['tgt_text'], res['pred_text'])
            
            res['bleu'] = bleu
            res['bertscore'] = bertscore
            res['semantic_similarity'] = sem_sim
            if 'ber_traditional' not in res:
                res['ber_traditional'] = 0.0
            final_data.append(res)
        except Exception:
            res['bleu'] = 0.0
            res['bertscore'] = 0.0
            res['semantic_similarity'] = 0.0
            if 'ber_traditional' not in res:
                res['ber_traditional'] = 0.0
            final_data.append(res)
            
    # Lưu kết quả
    df = pd.DataFrame(final_data)
    csv_path = os.path.join(config['training'].results_dir, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nDa luu ket qua tai: {csv_path}")
    
    # Tạo biểu đồ so sánh cho plot_all_results
    try:
        print("Tao cac bieu do so sanh...")
        save_dir = config['training'].results_dir
        
        # Nhóm kết quả theo SNR và Method để tạo bảng so sánh tương thích
        summary_rows = []
        for s in snr_levels:
            sem_subset = df[(df['snr'] == s) & (df['method'] == 'Semantic')]
            trad_subset = df[(df['snr'] == s) & (df['method'] == 'Traditional')]
            
            row = {'snr': s}
            if not sem_subset.empty:
                row['semantic_bleu'] = sem_subset['bleu'].mean()
                row['semantic_similarity'] = sem_subset['semantic_similarity'].mean()
            if not trad_subset.empty:
                row['traditional_bleu'] = trad_subset['bleu'].mean()
                row['traditional_similarity'] = trad_subset['semantic_similarity'].mean()
                row['traditional_ber'] = trad_subset['ber_traditional'].mean()
            summary_rows.append(row)
            
        summary_df = pd.DataFrame(summary_rows)
        plot_all_results(summary_df, save_dir)
        print(f"Bieu do da duoc luu trong thu muc: {save_dir}")
    except Exception as e:
        print(f"Khong the tao bieu do: {e}")

if __name__ == '__main__':
    main()
