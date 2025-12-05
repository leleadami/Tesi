"""
Cross-Building Training Script (Phase 2 Online Only)

Train on one building (Phase 2 online) → Test on OTHER TWO buildings FUSED (Phase 2 online)

Examples:
    Train on B1 → Test on B2+B3 fused
    Train on B2 → Test on B1+B3 fused
    Train on B3 → Test on B1+B2 fused
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import yaml
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Import project modules
from data.preprocessing import prepare_dataset, prepare_train_val_split
from data.dataloader import get_dataloaders, get_test_dataloader
from models.lstm import LSTMModel
from models.lstm_encoder_decoder import LSTMEncoderDecoder
from models.lstm_attention import LSTMAttention
from models.transformer import TransformerModel
from models.lstm_autoencoder import LSTMAutoencoder
from training.trainer import Trainer, predict
from evaluation.metrics import calculate_all_metrics, denormalize_predictions, print_metrics
from utils.experiment import get_cross_building_experiment_name


def load_config(model_type: str) -> dict:
    """Load configuration for specified model type"""
    config_map = {
        'baseline': '../../configs/baseline.yaml',
        'encoder_decoder': '../../configs/encoder_decoder.yaml',
        'attention': '../../configs/attention.yaml',
        'transformer': '../../configs/transformer.yaml',
        'autoencoder': '../../configs/autoencoder.yaml'
    }

    config_path = Path(__file__).parent / config_map[model_type]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_type: str, input_size: int, config: dict):
    """Create model based on type"""
    if model_type == 'baseline':
        return LSTMModel(
            input_size=input_size,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            output_size=config['model']['output_size']
        )
    elif model_type == 'encoder_decoder':
        return LSTMEncoderDecoder(
            input_size=input_size,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            output_size=config['model']['output_size']
        )
    elif model_type == 'attention':
        return LSTMAttention(
            input_size=input_size,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            output_size=config['model']['output_size']
        )
    elif model_type == 'transformer':
        return TransformerModel(
            input_size=input_size,
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            num_layers=config['model']['num_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            output_size=config['model']['output_size']
        )
    elif model_type == 'autoencoder':
        return LSTMAutoencoder(
            input_size=input_size,
            hidden_size=config['model']['hidden_size'],
            bottleneck_size=config['model']['bottleneck_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            output_size=config['model']['output_size']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def fuse_test_buildings(building_ids: list, phase: str, target_variable: str,
                        sequence_length: int, forecast_horizon: int, scalers: dict):
    """
    Fuse test data from multiple buildings into single dataset

    Args:
        building_ids: List of building IDs to fuse [2, 3]
        phase: Phase name
        target_variable: Target variable name
        sequence_length: Sequence length
        forecast_horizon: Forecast horizon
        scalers: Scalers fitted on training building

    Returns:
        X_fused, y_fused: Concatenated test data
    """
    X_list = []
    y_list = []

    for building_id in building_ids:
        X, y, _, _ = prepare_dataset(
            building_id=building_id,
            phase=phase,
            target_variable=target_variable,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            scalers=scalers,
            fit_scalers=False
        )
        X_list.append(X)
        y_list.append(y)

    # Concatenate along batch dimension
    X_fused = np.concatenate(X_list, axis=0)
    y_fused = np.concatenate(y_list, axis=0)

    return X_fused, y_fused


def main():
    parser = argparse.ArgumentParser(
        description='Cross-Building Training (Phase 2 online only, fused test)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on B1, test on B2+B3 fused
  python train_cross_building.py --train_building 1 --model encoder_decoder --hidden_size 128

  # Train on B2, test on B1+B3 fused
  python train_cross_building.py --train_building 2 --model attention --hidden_size 256

  # Full experiment for all 3 buildings
  for b in 1 2 3; do
    python train_cross_building.py --train_building $b --model encoder_decoder --hidden_size 128
  done

Results saved to: results/cross_building/
        """
    )

    parser.add_argument('--train_building', type=int, required=True,
                       choices=[1, 2, 3],
                       help='Building to train on')
    parser.add_argument('--model', type=str, required=True,
                       choices=['baseline', 'encoder_decoder', 'attention', 'transformer', 'autoencoder'],
                       help='Model type')
    parser.add_argument('--hidden_size', type=int, default=None,
                       help='Hidden size for LSTM models (overrides config)')
    parser.add_argument('--d_model', type=int, default=None,
                       help='Model dimension for Transformer (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout (overrides config)')

    args = parser.parse_args()

    # Always use phase_2_online for cross-building
    phase = 'phase_2_online'

    # Determine test buildings (the OTHER two buildings)
    all_buildings = [1, 2, 3]
    test_buildings = [b for b in all_buildings if b != args.train_building]

    print("="*80)
    print("CROSS-BUILDING TRAINING (Phase 2 Online, Fused Test)")
    print("="*80)
    print(f"\n✓ Train Building: {args.train_building}")
    print(f"✓ Test Buildings (fused): {test_buildings}")
    print(f"✓ Model: {args.model}")
    print(f"✓ Phase: {phase}")

    # Load config
    config = load_config(args.model)

    # Override config
    if args.hidden_size is not None:
        config['model']['hidden_size'] = args.hidden_size
        print(f"✓ Overriding hidden_size: {args.hidden_size}")
    if args.d_model is not None:
        config['model']['d_model'] = args.d_model
        print(f"✓ Overriding d_model: {args.d_model}")
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.dropout is not None:
        config['model']['dropout'] = args.dropout

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Device: {device}")

    # Set seed
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])

    # =========================================================================
    # 1. LOAD TRAINING DATA (one building, Phase 2 online)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: Loading training data")
    print("="*80)

    # Load data and split into train/val (split BEFORE normalization to avoid data leakage - FIXED!)
    print(f"\nLoading {phase} data (Building {args.train_building}) and splitting train/val (80/20)...")
    X_train, X_val, y_train, y_val, scalers, feature_names = prepare_train_val_split(
        building_id=args.train_building,
        phase=phase,
        target_variable=config['data']['target_variable'],
        sequence_length=config['data']['sequence_length'],
        forecast_horizon=config['data']['forecast_horizon'],
        val_split=0.2
    )
    print(f"✓ Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"✓ Val set: X={X_val.shape}, y={y_val.shape}")
    print(f"✓ Features: {feature_names}")

    # DataLoaders
    train_loader, val_loader = get_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=config['training']['batch_size'],
        shuffle_train=True
    )

    # =========================================================================
    # 2. CREATE MODEL
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: Creating model")
    print("="*80)

    input_size = X_train.shape[2]
    model = create_model(args.model, input_size, config)
    print(f"✓ Model: {args.model}")
    print(f"  - Parameters: {model.get_model_size():,}")

    # =========================================================================
    # 3. TRAIN MODEL
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: Training")
    print("="*80)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Create experiment directory with standardized naming
    print("\nCreating experiment directory...")
    experiment_name = get_cross_building_experiment_name(
        model_type=args.model,
        config=config,
        train_building=args.train_building,
        test_buildings=test_buildings
    )

    # Create directory structure
    project_root = Path(__file__).parent.parent.parent
    experiment_dir = project_root / 'results' / 'cross_building' / 'phase_2_online' / experiment_name
    checkpoint_dir = experiment_dir / 'checkpoints'
    figures_dir = experiment_dir / 'figures'

    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"✓ Experiment: {experiment_name}")
    print(f"✓ Directory: {experiment_dir}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        experiment_name='model'
    )

    print(f"✓ Trainer initialized")
    print(f"✓ Training...")

    history = trainer.train(
        num_epochs=config['training']['num_epochs'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        clip_grad=config['training']['gradient_clip'],
        verbose=True
    )

    # =========================================================================
    # 4. LOAD BEST MODEL AND TEST ON FUSED BUILDINGS
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: Testing on fused buildings")
    print("="*80)

    # Load best model
    best_model_path = Path(checkpoint_dir) / "model_best.pth"
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded best model (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.6f})")

    # Fuse test buildings
    print(f"\n✓ Fusing test data from Buildings {test_buildings}...")
    X_test_fused, y_test_fused = fuse_test_buildings(
        building_ids=test_buildings,
        phase=phase,
        target_variable=config['data']['target_variable'],
        sequence_length=config['data']['sequence_length'],
        forecast_horizon=config['data']['forecast_horizon'],
        scalers=scalers
    )
    print(f"✓ Fused test data: X={X_test_fused.shape}, y={y_test_fused.shape}")

    # Create test loader
    test_loader = get_test_dataloader(X_test_fused, y_test_fused,
                                     batch_size=config['training']['batch_size'])

    # Predict
    y_pred, y_true = predict(model, test_loader, device)

    # Denormalize
    target_scaler = scalers[config['data']['target_variable']]
    y_pred_orig = denormalize_predictions(y_pred, target_scaler)
    y_true_orig = denormalize_predictions(y_true, target_scaler)

    # Metrics
    metrics = calculate_all_metrics(y_true_orig, y_pred_orig)
    print_metrics(metrics, title=f"Cross-Building Test (B{args.train_building} → B{test_buildings})")

    # =========================================================================
    # 5. SAVE RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: Saving results")
    print("="*80)

    # Save config
    config_save = {
        'experiment_name': experiment_name,
        'timestamp': experiment_dir.name.split('_')[-2] + '_' + experiment_dir.name.split('_')[-1],
        'train_building': args.train_building,
        'test_buildings': test_buildings,
        'phase': phase,
        'model': {
            'type': args.model,
            'parameters': model.get_model_size()
        },
        'training': {
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'num_epochs': config['training']['num_epochs']
        },
        'data': {
            'target_variable': config['data']['target_variable'],
            'sequence_length': config['data']['sequence_length'],
            'forecast_horizon': config['data']['forecast_horizon'],
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test_fused)
        },
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': checkpoint['val_loss']
    }

    # Add model-specific parameters
    if args.model == 'transformer':
        config_save['model']['d_model'] = config['model']['d_model']
        config_save['model']['nhead'] = config['model']['nhead']
    elif args.model == 'autoencoder':
        config_save['model']['hidden_size'] = config['model']['hidden_size']
        config_save['model']['bottleneck_size'] = config['model']['bottleneck_size']
        config_save['model']['num_layers'] = config['model']['num_layers']
        config_save['model']['dropout'] = config['model']['dropout']
    else:
        config_save['model']['hidden_size'] = config['model']['hidden_size']
        config_save['model']['num_layers'] = config['model']['num_layers']
        config_save['model']['dropout'] = config['model']['dropout']

    config_path = experiment_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_save, f, indent=2)

    # Save metrics
    metrics_df = pd.DataFrame({'cross_building_test': metrics}).T
    metrics_path = experiment_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path)

    # Save training history
    history_path = experiment_dir / "training_history.json"
    with open(history_path, 'w') as f:
        history_json = {k: [float(v) for v in vals] for k, vals in history.items()}
        json.dump(history_json, f, indent=2)

    print(f"✓ Config saved: {config_path}")
    print(f"✓ Metrics saved: {metrics_path}")
    print(f"✓ Training history saved: {history_path}")

    # Generate plots
    print("\nGenerating plots...")

    # Training history plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss (MSE)', fontweight='bold')
    axes[0].set_title('Training History', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    axes[1].plot(history['learning_rate'], color='orange', linewidth=2)
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Learning Rate', fontweight='bold')
    axes[1].set_title('Learning Rate Schedule', fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    history_plot_path = figures_dir / "loss_curve.png"
    plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history plot saved: {history_plot_path}")

    # Predictions scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true_orig, y_pred_orig, alpha=0.95, s=40, edgecolors='darkblue', linewidth=0.6, color='#1E5F8C')

    min_val = min(y_true_orig.min(), y_pred_orig.min())
    max_val = max(y_true_orig.max(), y_pred_orig.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Ground Truth (kW)', fontweight='bold', fontsize=12)
    plt.ylabel('Prediction (kW)', fontweight='bold', fontsize=12)
    plt.title(f'Cross-Building Test: Ground Truth vs Predictions\nNMAE: {metrics["nmae"]:.4f} | NRMSE: {metrics["nrmse"]:.4f}',
              fontweight='bold', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    pred_plot_path = figures_dir / "predictions.png"
    plt.savefig(pred_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Predictions scatter plot saved: {pred_plot_path}")

    print("\n" + "="*80)
    print("✓ CROSS-BUILDING TRAINING COMPLETED!")
    print("="*80)
    print(f"\nExperiment: {experiment_name}")
    print(f"Test RMSE: {metrics['rmse']:.4f} kW")
    print(f"Test R²: {metrics['r2']:.4f}")
    print(f"Test MAE: {metrics['mae']:.4f} kW")
    print(f"\nAll results saved in:")
    print(f"  {experiment_dir}")
    print(f"\nContents:")
    print(f"  - config.json: Experiment configuration")
    print(f"  - metrics.csv: Performance metrics")
    print(f"  - training_history.json: Training history data")
    print(f"  - checkpoints/model_best.pth: Best model weights")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
