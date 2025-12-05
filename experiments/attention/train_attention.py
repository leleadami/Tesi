"""
Training script for LSTM Attention model

This script:
1. Loads and preprocesses data (Building 1, Phase 1 for training)
2. Creates LSTM Attention model
3. Trains the model
4. Evaluates on Phase 2 (local and online evaluation sets)
5. Saves results and visualizations
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
import matplotlib
matplotlib.use('Agg')  # Backend non-GUI per salvare immagini senza display
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import yaml
import json
from datetime import datetime

# Import project modules
from data.preprocessing import prepare_train_val_split, prepare_dataset
from data.dataloader import get_dataloaders, get_test_dataloader
from models.lstm_attention import LSTMAttention
from training.trainer import Trainer, predict
from evaluation.metrics import calculate_all_metrics, denormalize_predictions, print_metrics
from utils.experiment import create_experiment_dir


def load_config(config_path: str = '../../configs/attention.yaml') -> dict:
    """Load configuration from YAML file"""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train LSTM Attention model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_attention.py
  python train_attention.py --hidden_size 256
  python train_attention.py --phase phase_2_online --building_id 1
  python train_attention.py --phase phase_2_online --building_id 2 --hidden_size 256 --lr 0.0005

Checkpoints saved to: results/checkpoints/{phase}/
        """
    )
    parser.add_argument('--hidden_size', type=int, default=None,
                       help='LSTM hidden size (default: from config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (default: from config)')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout rate (default: from config)')
    parser.add_argument('--phase', type=str, default='phase_1',
                       choices=['phase_1', 'phase_2_local', 'phase_2_online'],
                       help='Training phase (default: phase_1)')
    parser.add_argument('--building_id', type=int, default=1,
                       choices=[1, 2, 3],
                       help='Building ID (default: 1)')
    args = parser.parse_args()

    print("="*80)
    print("LSTM Attention Training")
    print("="*80)

    # Load configuration
    config = load_config()

    # Override config with command line arguments
    if args.hidden_size is not None:
        config['model']['hidden_size'] = args.hidden_size
        print(f"\n✓ Overriding hidden_size: {args.hidden_size}")
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
        print(f"✓ Overriding batch_size: {args.batch_size}")
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
        print(f"✓ Overriding learning_rate: {args.lr}")
    if args.dropout is not None:
        config['model']['dropout'] = args.dropout
        print(f"✓ Overriding dropout: {args.dropout}")

    # Phase and building info
    phase = args.phase
    building_id = args.building_id
    print(f"✓ Training phase: {phase}")
    print(f"✓ Building ID: {building_id}")

    print("\n✓ Configuration loaded")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])

    # =========================================================================
    # 1. LOAD AND PREPROCESS DATA
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: Loading and preprocessing data")
    print("="*80)

    # Load data and split into train/val (split BEFORE normalization to avoid data leakage)
    print(f"\nLoading {phase} data (Building {building_id}) and splitting train/val (80/20)...")
    X_train, X_val, y_train, y_val, scalers, feature_names = prepare_train_val_split(
        building_id=building_id,
        phase=phase,
        target_variable=config['data']['target_variable'],
        sequence_length=config['data']['sequence_length'],
        forecast_horizon=config['data']['forecast_horizon'],
        val_split=0.2
    )
    print(f"✓ Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"✓ Val set: X={X_val.shape}, y={y_val.shape}")
    print(f"✓ Features: {feature_names}")

    # Create DataLoaders
    print("\nCreating DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=config['training']['batch_size'],
        shuffle_train=True
    )
    print(f"✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # =========================================================================
    # 2. CREATE MODEL
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: Creating LSTM Attention model")
    print("="*80)

    # Get input size from data
    input_size = X_train.shape[2]  # Number of features

    # Create LSTM Attention model
    model = LSTMAttention(
        input_size=input_size,
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        output_size=config['model']['output_size']
    )
    print(f"\n✓ Model created")
    print(f"  - Input size: {input_size}")
    print(f"  - Hidden size: {config['model']['hidden_size']}")
    print(f"  - Num layers: {config['model']['num_layers']}")
    print(f"  - Dropout: {config['model']['dropout']}")
    print(f"  - Total parameters: {model.get_model_size():,}")

    # =========================================================================
    # 3. SETUP TRAINING
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: Setup training")
    print("="*80)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    print(f"\n✓ Optimizer: Adam (lr={config['training']['learning_rate']})")

    # Create experiment directory structure with standardized naming
    print("\nCreating experiment directory...")
    experiment_paths = create_experiment_dir(
        model_type='attention',
        config=config,
        building_id=building_id,
        phase=phase,
        experiment_type='single_building'
    )

    experiment_name = experiment_paths['experiment_name']
    experiment_dir = experiment_paths['experiment_dir']
    checkpoint_dir = experiment_paths['checkpoints']
    figures_dir = experiment_paths['figures']

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

    # =========================================================================
    # 4. TRAIN MODEL
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: Training model")
    print("="*80)

    # Train
    history = trainer.train(
        num_epochs=config['training']['num_epochs'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        clip_grad=config['training']['gradient_clip'],
        verbose=True
    )

    # =========================================================================
    # 5. LOAD BEST MODEL AND EVALUATE
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: Evaluation on test sets")
    print("="*80)

    # Load best model
    best_model_path = Path(checkpoint_dir) / "model_best.pth"
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✓ Loaded best model (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.6f})")

    # Evaluate on validation set (20% of Phase 1)
    print("\n--- Validation Set (Phase 1, last 20%) ---")
    y_pred_val, y_true_val = predict(model, val_loader, device)

    # Denormalize predictions (back to original scale)
    target_scaler = scalers[config['data']['target_variable']]
    y_pred_val_orig = denormalize_predictions(y_pred_val, target_scaler)
    y_true_val_orig = denormalize_predictions(y_true_val, target_scaler)

    # Calculate metrics
    metrics_val = calculate_all_metrics(y_true_val_orig, y_pred_val_orig)
    print_metrics(metrics_val, title="Validation Metrics (Original Scale)")

    # Evaluate on Phase 2 test sets
    # Note: Using phase_2_online (2208h dataset)_1
    test_phases = ['phase_2_local', 'phase_2_online']
    test_results = {}

    for phase in test_phases:
        print(f"\n--- Test Set ({phase}) ---")

        # Load test data
        X_test, y_test, _, _ = prepare_dataset(
            building_id=1,
            phase=phase,
            target_variable=config['data']['target_variable'],
            sequence_length=config['data']['sequence_length'],
            forecast_horizon=config['data']['forecast_horizon'],
            scalers=scalers,
            fit_scalers=False
        )

        # Create test loader
        test_loader = get_test_dataloader(X_test, y_test, batch_size=config['training']['batch_size'])

        # Predict
        y_pred_test, y_true_test = predict(model, test_loader, device)

        # Denormalize
        y_pred_test_orig = denormalize_predictions(y_pred_test, target_scaler)
        y_true_test_orig = denormalize_predictions(y_true_test, target_scaler)

        # Calculate metrics
        metrics_test = calculate_all_metrics(y_true_test_orig, y_pred_test_orig)
        print_metrics(metrics_test, title=f"{phase} Metrics")

        # Store results
        test_results[phase] = {
            'y_true': y_true_test_orig,
            'y_pred': y_pred_test_orig,
            'metrics': metrics_test
        }

    # =========================================================================
    # 6. SAVE RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 6: Saving results")
    print("="*80)

    # Create directories if they don't exist (NEW STRUCTURE)
    project_root = Path(__file__).parent.parent.parent

    # Save experiment configuration (hyperparameters)
    config_save = {
        'experiment_name': experiment_name,
        'phase': phase,
        'building_id': building_id,
        'timestamp': experiment_dir.name.split('_')[-2] + '_' + experiment_dir.name.split('_')[-1],
        'model': {
            'type': 'LSTM_Attention',
            'hidden_size': config['model']['hidden_size'],
            'num_layers': config['model']['num_layers'],
            'dropout': config['model']['dropout'],
            'total_parameters': model.get_model_size()
        },
        'training': {
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'weight_decay': config['training']['weight_decay'],
            'num_epochs': config['training']['num_epochs'],
            'early_stopping_patience': config['training']['early_stopping_patience'],
            'gradient_clip': config['training']['gradient_clip'],
            'optimizer': 'Adam'
        },
        'data': {
            'target_variable': config['data']['target_variable'],
            'sequence_length': config['data']['sequence_length'],
            'forecast_horizon': config['data']['forecast_horizon'],
            'train_samples': len(y_train),
            'val_samples': len(y_val)
        },
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': checkpoint['val_loss'],
        'device': device
    }

    config_path = experiment_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_save, f, indent=2)
    print(f"\n✓ Configuration saved to: {config_path}")

    # Combine all metrics
    all_metrics = {
        'validation': metrics_val,
        **{phase: test_results[phase]['metrics'] for phase in test_phases}
    }

    metrics_df = pd.DataFrame(all_metrics).T
    metrics_path = experiment_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path)
    print(f"✓ Metrics saved to: {metrics_path}")

    # Save training history as JSON
    history_path = experiment_dir / "training_history.json"
    with open(history_path, 'w') as f:
        history_json = {k: [float(v) for v in vals] for k, vals in history.items()}
        json.dump(history_json, f, indent=2)
    print(f"✓ Training history saved to: {history_path}")

    # Save training history plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss (MSE)', fontweight='bold')
    axes[0].set_title('Training History', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Learning rate
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
    print(f"✓ Training history plot saved to: {history_plot_path}")

    # Save predictions scatter plot (ground truth vs predictions)
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true_val_orig, y_pred_val_orig, alpha=0.95, s=40, edgecolors='darkblue', linewidth=0.6, color='#1E5F8C')

    # Perfect prediction line (45-degree)
    min_val = min(y_true_val_orig.min(), y_pred_val_orig.min())
    max_val = max(y_true_val_orig.max(), y_pred_val_orig.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Ground Truth (kW)', fontweight='bold', fontsize=12)
    plt.ylabel('Prediction (kW)', fontweight='bold', fontsize=12)
    plt.title(f'Validation Set: Ground Truth vs Predictions\nNMAE: {metrics_val["nmae"]:.4f} | NRMSE: {metrics_val["nrmse"]:.4f}',
              fontweight='bold', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    pred_plot_path = Path(figures_dir) / "predictions.png"
    plt.savefig(pred_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Predictions scatter plot saved to: {pred_plot_path}")

    print("\n" + "="*80)
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nExperiment: {experiment_name}")
    print(f"Best validation RMSE: {metrics_val['rmse']:.4f}")
    print(f"Best validation R²: {metrics_val['r2']:.4f}")
    print(f"Best validation NMAE: {metrics_val['nmae']:.4f}")
    print(f"Best validation NRMSE: {metrics_val['nrmse']:.4f}")
    print(f"\nAll results saved in:")
    print(f"  {experiment_dir}")
    print(f"\nContents:")
    print(f"  - config.json: Experiment configuration")
    print(f"  - metrics.csv: Performance metrics")
    print(f"  - training_history.json: Training history data")
    print(f"  - checkpoints/model_best.pth: Best model weights")
    print(f"  - figures/loss_curve.png: Training curves")
    print(f"  - figures/predictions.png: Ground truth vs predictions scatter plot")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
