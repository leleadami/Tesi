"""
Training script for LSTM Autoencoder model

This script implements 2-phase training:
1. Pre-training: Learn to reconstruct input (unsupervised)
2. Fine-tuning: Use bottleneck for forecasting (supervised)

Steps:
1. Loads and preprocesses data
2. Phase 1: Pre-train autoencoder on reconstruction task
3. Phase 2: Fine-tune forecasting head using frozen encoder
4. Evaluates on Phase 2 test sets
5. Saves results and visualizations
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import yaml
import json
from datetime import datetime
from tqdm import tqdm

# Import project modules
from data.preprocessing import prepare_train_val_split, prepare_dataset
from data.dataloader import get_dataloaders, get_test_dataloader
from models.lstm_autoencoder import LSTMAutoencoder
from evaluation.metrics import calculate_all_metrics, denormalize_predictions, print_metrics
from utils.experiment import create_experiment_dir


def load_config(config_path: str = '../../configs/autoencoder.yaml') -> dict:
    """Load configuration from YAML file"""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def pretrain_autoencoder(model, train_loader, val_loader, optimizer, device, num_epochs, patience=15):
    """
    Phase 1: Pre-train autoencoder on reconstruction task

    Returns:
        history: Training history
    """
    print("\n" + "="*80)
    print("PHASE 1: PRE-TRAINING AUTOENCODER (Reconstruction)")
    print("="*80)

    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False, ncols=100,
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| [{n_fmt}/{total_fmt}]',
                  ascii=False, colour='cyan') as pbar:
            for X, _ in pbar:  # Ignoriamo y, usiamo X per ricostruzione
                X = X.to(device)

                optimizer.zero_grad()

                # Forward: ricostruisci input
                reconstruction = model(X, mode='reconstruction')

                # Loss: differenza tra input e ricostruzione
                loss = criterion(reconstruction, X)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X, _ in val_loader:
                X = X.to(device)
                reconstruction = model(X, mode='reconstruction')
                loss = criterion(reconstruction, X)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  New best reconstruction loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nPre-training completed! Best reconstruction loss: {best_val_loss:.6f}")
    return history


def finetune_forecasting(model, train_loader, val_loader, optimizer, device, num_epochs, patience=15):
    """
    Phase 2: Fine-tune forecasting head (encoder frozen)

    Returns:
        history: Training history
    """
    print("\n" + "="*80)
    print("PHASE 2: FINE-TUNING FORECASTING HEAD")
    print("="*80)

    # MODIFICA: UNFREEZE encoder per permettere fine-tuning end-to-end
    # Questo permette all'encoder di adattarsi al task di forecasting
    # (prima era freezato e imparava solo dalla fase di reconstruction)
    for param in model.encoder.parameters():
        param.requires_grad = True  # ← UNFROZEN!
    for param in model.encoder_fc.parameters():
        param.requires_grad = True  # ← UNFROZEN!

    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False, ncols=100,
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| [{n_fmt}/{total_fmt}]',
                  ascii=False, colour='green') as pbar:
            for X, y in pbar:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()

                # Forward: predici usando bottleneck
                output = model(X, mode='forecast')

                # Loss: differenza tra predizione e target
                loss = criterion(output, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X, mode='forecast')
                loss = criterion(output, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  New best forecasting loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nFine-tuning completed! Best forecasting loss: {best_val_loss:.6f}")
    return history


def predict_autoencoder(model, data_loader, device):
    """Make predictions using autoencoder"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for X, _ in data_loader:
            X = X.to(device)
            output = model(X, mode='forecast')
            predictions.append(output.cpu().numpy())

    return np.concatenate(predictions)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train LSTM Autoencoder model')
    parser.add_argument('--hidden_size', type=int, default=None,
                       help='LSTM hidden size (default: from config)')
    parser.add_argument('--bottleneck_size', type=int, default=None,
                       help='Bottleneck size (default: from config)')
    parser.add_argument('--use_skip_connections', action='store_true',
                       help='Enable skip connections (default: False)')
    parser.add_argument('--phase', type=str, default='phase_1',
                       choices=['phase_1', 'phase_2_local', 'phase_2_online'],
                       help='Training phase (default: phase_1)')
    parser.add_argument('--building_id', type=int, default=1,
                       choices=[1, 2, 3],
                       help='Building ID (default: 1)')
    args = parser.parse_args()

    phase = args.phase
    building_id = args.building_id

    print("="*80)
    print("LSTM Autoencoder Training")
    print("="*80)

    # Load config
    config = load_config()

    if args.hidden_size is not None:
        config['model']['hidden_size'] = args.hidden_size
        print(f"\nOverriding hidden_size: {args.hidden_size}")
    if args.bottleneck_size is not None:
        config['model']['bottleneck_size'] = args.bottleneck_size
        print(f"Overriding bottleneck_size: {args.bottleneck_size}")
    if args.use_skip_connections:
        config['model']['use_skip_connections'] = True
        print(f"Overriding use_skip_connections: True")

    print(f"Training phase: {phase}")
    print(f"Building ID: {building_id}")
    print("\nConfiguration loaded")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create experiment directory structure
    print("\nCreating experiment directory...")
    experiment_paths = create_experiment_dir(
        model_type='autoencoder',
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

    # Load data
    print("\n" + "="*80)
    print("Loading and preprocessing data")
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

    train_loader, val_loader = get_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=config['training']['batch_size']
    )

    # Create model
    print("\n" + "="*80)
    print("Creating LSTM Autoencoder model")
    print("="*80)

    input_size = X_train.shape[2]
    model = LSTMAutoencoder(
        input_size=input_size,
        hidden_size=config['model']['hidden_size'],
        bottleneck_size=config['model']['bottleneck_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        output_size=config['model']['output_size'],
        use_skip_connections=config['model'].get('use_skip_connections', False)  # Default: False
    ).to(device)

    print(f"\nModel created:")
    print(f"  - Input size: {input_size}")
    print(f"  - Hidden size: {config['model']['hidden_size']}")
    print(f"  - Bottleneck size: {config['model']['bottleneck_size']}")
    print(f"  - Skip connections: {config['model'].get('use_skip_connections', False)}")
    print(f"  - Total parameters: {model.get_model_size():,}")

    # PHASE 1: Pre-training (reconstruction)
    optimizer_pretrain = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    pretrain_history = pretrain_autoencoder(
        model, train_loader, val_loader, optimizer_pretrain, device,
        num_epochs=config['training']['pretrain_epochs'],
        patience=config['training']['early_stopping_patience']
    )

    # PHASE 2: Fine-tuning (forecasting)
    # MODIFICA: Learning rate differenziato
    # - Forecast head: LR alto (0.001) - deve imparare da zero
    # - Encoder/bottleneck: LR basso (0.0001) - già pre-trained, solo fine-tuning
    optimizer_finetune = optim.Adam([
        {'params': model.forecast_fc.parameters(), 'lr': config['training']['learning_rate']},  # 0.001
        {'params': model.encoder.parameters(), 'lr': config['training']['learning_rate'] * 0.1},  # 0.0001
        {'params': model.encoder_fc.parameters(), 'lr': config['training']['learning_rate'] * 0.1}  # 0.0001
    ])
    finetune_history = finetune_forecasting(
        model, train_loader, val_loader, optimizer_finetune, device,
        num_epochs=config['training']['finetune_epochs'],
        patience=config['training']['early_stopping_patience']
    )

    # Evaluation
    print("\n" + "="*80)
    print("Evaluation on test sets")
    print("="*80)

    # Validation
    y_pred_val = predict_autoencoder(model, val_loader, device)
    y_true_val_orig = denormalize_predictions(y_val, scalers, config['data']['target_variable'])
    y_pred_val_orig = denormalize_predictions(y_pred_val, scalers, config['data']['target_variable'])

    metrics_val = calculate_all_metrics(y_true_val_orig, y_pred_val_orig)
    print("\n--- Validation Set ---")
    print_metrics(metrics_val, title="Validation Metrics")

    # Test sets
    test_phases = ['phase_2_local', 'phase_2_online']
    test_results = {}

    for test_phase in test_phases:
        X_test, y_test, _, _ = prepare_dataset(
            building_id=building_id,
            phase=test_phase,
            target_variable=config['data']['target_variable'],
            sequence_length=config['data']['sequence_length'],
            forecast_horizon=config['data']['forecast_horizon'],
            fit_scalers=False,
            scalers=scalers
        )

        test_loader = get_test_dataloader(X_test, y_test, batch_size=config['training']['batch_size'])

        y_pred_test = predict_autoencoder(model, test_loader, device)
        y_true_test_orig = denormalize_predictions(y_test, scalers, config['data']['target_variable'])
        y_pred_test_orig = denormalize_predictions(y_pred_test, scalers, config['data']['target_variable'])

        metrics_test = calculate_all_metrics(y_true_test_orig, y_pred_test_orig)

        print(f"\n--- Test Set ({test_phase}) ---")
        print_metrics(metrics_test, title=f"{test_phase} Metrics")

        test_results[test_phase] = {
            'metrics': metrics_test,
            'y_true': y_true_test_orig,
            'y_pred': y_pred_test_orig
        }

    # Save results
    print("\n" + "="*80)
    print("Saving results")
    print("="*80)

    # Save config
    config_save = {
        'experiment_name': experiment_name,
        'phase': phase,
        'building_id': building_id,
        'timestamp': experiment_dir.name.split('_')[-2] + '_' + experiment_dir.name.split('_')[-1],
        'model': {
            'type': 'LSTM_Autoencoder',
            'hidden_size': config['model']['hidden_size'],
            'bottleneck_size': config['model']['bottleneck_size'],
            'num_layers': config['model']['num_layers'],
            'dropout': config['model']['dropout'],
            'total_parameters': model.get_model_size()
        },
        'training': {
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'pretrain_epochs': config['training']['pretrain_epochs'],
            'finetune_epochs': config['training']['finetune_epochs'],
            'optimizer': 'Adam'
        },
        'data': {
            'target_variable': config['data']['target_variable'],
            'sequence_length': config['data']['sequence_length'],
            'forecast_horizon': config['data']['forecast_horizon'],
            'train_samples': len(y_train),
            'val_samples': len(y_val)
        },
        'device': device
    }

    config_path = experiment_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_save, f, indent=2)
    print(f"\nConfiguration saved to: {config_path}")

    # Save metrics
    all_metrics = {
        'validation': metrics_val,
        **{phase: test_results[phase]['metrics'] for phase in test_phases}
    }

    metrics_df = pd.DataFrame(all_metrics).T
    metrics_path = experiment_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path)
    print(f"Metrics saved to: {metrics_path}")

    # Save model checkpoint
    checkpoint_path = Path(checkpoint_dir) / "model_best.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config_save,
        'metrics': all_metrics
    }, checkpoint_path)
    print(f"Model checkpoint saved to: {checkpoint_path}")

    # Save training history JSON
    history_combined = {
        'pretrain': pretrain_history,
        'finetune': finetune_history
    }
    history_json_path = experiment_dir / "training_history.json"
    with open(history_json_path, 'w') as f:
        json.dump(history_combined, f, indent=2)
    print(f"Training history saved to: {history_json_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pre-training history
    axes[0].plot(pretrain_history['train_loss'], label='Train Loss (Reconstruction)', linewidth=2)
    axes[0].plot(pretrain_history['val_loss'], label='Val Loss (Reconstruction)', linewidth=2)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss (MSE)', fontweight='bold')
    axes[0].set_title('Phase 1: Pre-training (Reconstruction)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Fine-tuning history
    axes[1].plot(finetune_history['train_loss'], label='Train Loss (Forecasting)', linewidth=2, color='green')
    axes[1].plot(finetune_history['val_loss'], label='Val Loss (Forecasting)', linewidth=2, color='orange')
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Loss (MSE)', fontweight='bold')
    axes[1].set_title('Phase 2: Fine-tuning (Forecasting)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    history_plot_path = figures_dir / "loss_curve.png"
    plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {history_plot_path}")

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

    pred_plot_path = figures_dir / "predictions.png"
    plt.savefig(pred_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Predictions scatter plot saved to: {pred_plot_path}")

    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
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
    print(f"  - training_history.json: Pre-training and fine-tuning history")
    print(f"  - checkpoints/model_best.pth: Best model weights")
    print(f"  - figures/loss_curve.png: Training curves (pretrain + finetune)")
    print(f"  - figures/predictions.png: Ground truth vs predictions scatter plot")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
