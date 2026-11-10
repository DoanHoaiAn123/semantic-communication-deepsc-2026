"""Fine-tune a saved standard model with FGSM and C&W on local data."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch

from models.semantic_model import SemanticCommModel
from training.config import get_default_config
from training.dataset import create_dataloaders
from training.train_adversarial import AdversarialTrainer


def file_sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    parser.add_argument('--dataset', choices=['all', 'opus', 'tatoeba', 'combined'], default='all')
    parser.add_argument('--samples', type=int, default=0, help='0 loads all supported local samples.')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cw_steps', type=int, default=15, help='C&W training iterations (1 to 15).')
    parser.add_argument('--save_every_steps', type=int, default=25)
    parser.add_argument('--threads', type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_dir', default=None)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.save_every_steps, args.threads) < 1:
        parser.error('epochs, batch_size, save_every_steps and threads must be positive')
    if not 1 <= args.cw_steps <= 15 or args.lr <= 0 or args.samples < 0:
        parser.error('cw_steps must be 1..15, lr positive, and samples nonnegative')

    os.chdir(ROOT)
    checkpoint_path = Path(args.checkpoint).resolve()
    vocab_path = ROOT / 'checkpoints' / 'vocab.json'
    for path in (checkpoint_path, vocab_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required pretrained model/tokenizer file missing: {path}')

    vn = timezone(timedelta(hours=7))
    started_at = datetime.now(vn)
    run_dir = Path(args.run_dir or f'results/adversarial_{started_at:%Y%m%d_%H%M%S}').resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / 'run.json'
    if manifest_path.exists():
        raise FileExistsError(f'Use a new run directory; existing run found: {manifest_path}')

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    config = get_default_config()
    data_cfg, train_cfg, model_cfg = config['data'], config['training'], config['model']
    data_cfg.dataset_name = {
        'all': 'all_raw', 'opus': 'opus_raw',
        'tatoeba': 'tatoeba_raw', 'combined': 'combined_raw',
    }[args.dataset]
    data_cfg.max_samples = args.samples
    train_cfg.epochs = args.epochs
    train_cfg.batch_size = args.batch_size
    train_cfg.lr = args.lr
    train_cfg.save_every_steps = args.save_every_steps
    train_cfg.progress_path = str(run_dir / 'progress.json')
    config['adversarial'].cw_max_iter = args.cw_steps

    print(f'Adversarial fine-tuning started at {started_at.isoformat()}', flush=True)
    print(f'Pretrained model: {checkpoint_path}', flush=True)
    print(f'Device: {"cuda" if torch.cuda.is_available() else "cpu"}; threads={args.threads}', flush=True)
    print(f'FGSM + C&W: {args.epochs} epochs, batch={args.batch_size}, C&W steps={args.cw_steps}', flush=True)
    print('Using saved vocab; creating a reproducible split with seed', args.seed, flush=True)
    print('The previous standard run did not save its split; this split may differ.', flush=True)
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
        data_cfg, batch_size=args.batch_size, rebuild_vocab=False, save_vocab=False,
    )
    counts = {
        'train': len(train_loader.dataset),
        'validation': len(val_loader.dataset),
        'test': len(test_loader.dataset),
    }
    print(f'Dataset sizes: {counts}; train batches/epoch={len(train_loader)}', flush=True)
    model_cfg.vocab_size = tokenizer.vocab_size
    model = SemanticCommModel(**asdict(model_cfg))
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    source_epoch = checkpoint.get('epoch')
    del checkpoint
    print(f'Loaded standard checkpoint successfully (epoch={source_epoch}).', flush=True)

    # Preserve the previous robust models before writing results for this run.
    backup_dir = run_dir / 'previous_robust'
    backup_dir.mkdir(exist_ok=True)
    for name in ('best_robust_model.pth', 'last_robust_model.pth'):
        source = ROOT / train_cfg.checkpoint_dir / name
        if source.exists():
            shutil.copy2(source, backup_dir / name)
    shutil.copy2(vocab_path, run_dir / 'vocab.json')
    manifest = {
        'started_at': started_at.isoformat(), 'pid': os.getpid(),
        'arguments': vars(args), 'source_checkpoint': str(checkpoint_path),
        'source_checkpoint_sha256': file_sha256(checkpoint_path),
        'source_epoch': source_epoch, 'vocab_sha256': file_sha256(vocab_path),
        'dataset_counts': counts, 'train_batches_per_epoch': len(train_loader),
        'planned_updates': len(train_loader) * args.epochs,
        'planned_sample_exposures': counts['train'] * args.epochs,
        'data_config': asdict(data_cfg), 'model_config': asdict(model_cfg),
        'training_config': asdict(train_cfg), 'adversarial_config': asdict(config['adversarial']),
        'split_seed': args.seed, 'split_matches_standard_run': False,
        'split_note': 'Standard checkpoint has no split or RNG state. This is a new seeded split of the same supported corpus; validation/test are not independent of standard pretraining.',
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    train_cfg.run_metadata = manifest
    trainer = AdversarialTrainer(model, train_loader, val_loader, tokenizer, config)
    try:
        history = trainer.train(num_epochs=args.epochs)
    except Exception:
        trainer._write_progress('failed')
        raise
    (run_dir / 'history.json').write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print(f'Run artifacts: {run_dir}', flush=True)


if __name__ == '__main__':
    main()
