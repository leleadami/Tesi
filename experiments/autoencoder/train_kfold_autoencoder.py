"""
K-Fold Sliding Window CV - LSTM Autoencoder

This script implements K-fold cross-validation with 2-phase training:
1. Pre-training: Learn to reconstruct input (unsupervised)
2. Fine-tuning: Use bottleneck for forecasting (supervised)

Usage:
    python train_kfold_autoencoder.py --building_id 1 --target cooling_demand --k_folds 5
    python train_kfold_autoencoder.py --hidden_size 256 --bottleneck_size 64 --building_id 2
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import yaml
import json
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import KFold
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from data.preprocessing import prepare_dataset
from data.dataloader import get_dataloaders
from models.lstm_autoencoder import LSTMAutoencoder
from evaluation.metrics import calculate_metrics, calculate_metrics_with_std, print_metrics_with_std
from evaluation.plotting import plot_scatter_predictions, plot_kfold_scatter
from utils.experiment import create_experiment_dir
from utils.cross_validation import TimeSeriesSlidingWindow, print_split_info


def load_config() -> dict:
    """Load autoencoder model configuration"""
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'autoencoder.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_kfold_config() -> dict:
    """Load K-fold experiment configuration"""
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'kfold.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def pretrain_autoencoder(model, train_loader, val_loader, optimizer, device, num_epochs, patience=15):
    """
    Phase 1: Pre-train autoencoder on reconstruction task

    Returns:
        best_val_loss: Best validation loss achieved
        history: Training history
    """
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        with tqdm(train_loader, desc=f'Pretrain {epoch+1}/{num_epochs}', leave=False, ncols=100,
                  bar_format='{desc}: {percentage:3.0f}%|{bar}|', ascii=False, colour='cyan') as pbar:
            for X, _ in pbar:
                X = X.to(device)
                optimizer.zero_grad()
                reconstruction = model(X, mode='reconstruction')
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

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_loss, history


def finetune_forecasting(model, train_loader, val_loader, optimizer, device, num_epochs, patience=15):
    """
    Phase 2: Fine-tune forecasting head (encoder frozen)

    Returns:
        best_val_loss: Best validation loss achieved
        best_epoch: Epoch with best validation loss
        history: Training history
    """
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.encoder_fc.parameters():
        param.requires_grad = False

    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        with tqdm(train_loader, desc=f'Finetune {epoch+1}/{num_epochs}', leave=False, ncols=100,
                  bar_format='{desc}: {percentage:3.0f}%|{bar}|', ascii=False, colour='green') as pbar:
            for X, y in pbar:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(X, mode='forecast')
                loss = criterion(y_pred.squeeze(), y)
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
                y_pred = model(X, mode='forecast')
                loss = criterion(y_pred.squeeze(), y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_loss, best_epoch, history


def predict_autoencoder(model, loader, device):
    """Make predictions using the autoencoder in forecast mode"""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y_pred = model(X, mode='forecast')
            predictions.append(y_pred.cpu().numpy())
            targets.append(y.numpy())

    predictions = np.concatenate(predictions).flatten()
    targets = np.concatenate(targets).flatten()

    return predictions, targets


def main():
    parser = argparse.ArgumentParser(
        description='K-Fold Sliding Window CV - LSTM Autoencoder',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--building_id', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--target', type=str, default='cooling_demand',
                       choices=['cooling_demand', 'carbon_intensity', 'solar_generation'])
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--phase', type=str, default='phase_1',
                       choices=['phase_1', 'phase_2_local', 'phase_2_online'])
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--bottleneck_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)

    args = parser.parse_args()

    print("=" * 80)
    print("TIME SERIES CROSS VALIDATION - LSTM AUTOENCODER")
    print("=" * 80)
    print(f"Building: {args.building_id}")
    print(f"Target: {args.target}")
    print(f"Phase: {args.phase}")
    print("=" * 80)
    print()

    kfold_config = load_kfold_config()
    model_config = load_config()

    if args.hidden_size is not None:
        model_config['model']['hidden_size'] = args.hidden_size
    if args.bottleneck_size is not None:
        model_config['model']['bottleneck_size'] = args.bottleneck_size
    if args.batch_size is not None:
        model_config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        model_config['training']['learning_rate'] = args.lr
    if args.num_epochs is not None:
        model_config['training']['num_epochs'] = args.num_epochs

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    torch.manual_seed(kfold_config['experiment']['random_state'])
    np.random.seed(kfold_config['experiment']['random_state'])

    # =========================================================================
    # STEP 1: Prepare full dataset
    # =========================================================================
    print("STEP 1: Loading dataset...")
    print("-" * 80)

    X_full, y_full, scalers, feature_names = prepare_dataset(
        building_id=args.building_id,
        phase=args.phase,
        target_variable=args.target,
        sequence_length=kfold_config['data']['sequence_length'],
        forecast_horizon=kfold_config['data']['forecast_horizon'],
        weather_features=kfold_config['data']['weather_features']
    )

    print(f"Total samples: {len(X_full)}")
    print(f"Features: {X_full.shape[2]}")
    print(f"Feature names: {feature_names}")
    print()

    # =========================================================================
    # STEP 2: Time Series Cross Validation
    # =========================================================================
    print(f"STEP 2: Time Series Cross Validation")
    print("-" * 80)

    # Use sliding window for time series (avoids temporal leakage)
    if kfold_config['experiment'].get('use_sliding_window', True):
        # If k_folds is specified via CLI, use n_splits parameter
        if args.k_folds is not None:
            cv = TimeSeriesSlidingWindow(
                train_size=kfold_config['experiment']['train_size'],
                test_size=kfold_config['experiment']['test_size'],
                n_splits=args.k_folds
            )
        else:
            # Otherwise use step from config
            cv = TimeSeriesSlidingWindow(
                train_size=kfold_config['experiment']['train_size'],
                test_size=kfold_config['experiment']['test_size'],
                step=kfold_config['experiment']['step']
            )
        print_split_info(cv, len(X_full))
        n_folds = cv.get_n_splits(X_full)
    else:
        # Legacy K-fold (has temporal leakage - not recommended)
        print("WARNING: Using standard K-fold (has temporal leakage)")
        if kfold_config['experiment']['shuffle']:
            cv = KFold(n_splits=args.k_folds, shuffle=True, random_state=kfold_config['experiment']['random_state'])
        else:
            cv = KFold(n_splits=args.k_folds, shuffle=False)
        n_folds = args.k_folds

    fold_results = []
    y_true_folds = []
    y_pred_folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_full), 1):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx}/{n_folds}")
        print(f"{'='*80}")

        # Split data
        X_train, X_val = X_full[train_idx], X_full[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]

        print(f"Train samples: {len(X_train)}")
        print(f"Val samples: {len(X_val)}")

        # Create dataloaders
        train_loader, val_loader = get_dataloaders(
            X_train, y_train, X_val, y_val,
            batch_size=model_config['training']['batch_size']
        )

        # Create model
        input_size = X_train.shape[2]
        model = LSTMAutoencoder(
            input_size=input_size,
            hidden_size=model_config['model']['hidden_size'],
            bottleneck_size=model_config['model']['bottleneck_size'],
            num_layers=model_config['model']['num_layers'],
            dropout=model_config['model']['dropout'],
            output_size=model_config['model']['output_size']
        )
        model = model.to(device)

        # Phase 1: Pre-training
        print(f"\nPhase 1: Pre-training autoencoder (reconstruction)...")
        optimizer_pretrain = optim.Adam(
            model.parameters(),
            lr=model_config['training']['learning_rate'],
            weight_decay=model_config['training']['weight_decay']
        )

        pretrain_val_loss, pretrain_history = pretrain_autoencoder(
            model, train_loader, val_loader, optimizer_pretrain, device,
            num_epochs=model_config['training']['pretrain_epochs'],
            patience=model_config['training']['early_stopping_patience']
        )
        print(f"✓ Pre-training completed. Best reconstruction loss: {pretrain_val_loss:.6f}")

        # Phase 2: Fine-tuning
        print(f"\nPhase 2: Fine-tuning forecasting head...")
        optimizer_finetune = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=model_config['training']['learning_rate'],
            weight_decay=model_config['training']['weight_decay']
        )

        finetune_val_loss, best_epoch, finetune_history = finetune_forecasting(
            model, train_loader, val_loader, optimizer_finetune, device,
            num_epochs=model_config['training']['num_epochs'],
            patience=model_config['training']['early_stopping_patience']
        )
        print(f"✓ Fine-tuning completed. Best forecast loss: {finetune_val_loss:.6f} at epoch {best_epoch}")

        # Predict on validation set
        y_pred, y_true = predict_autoencoder(model, val_loader, device)

        # Denormalize
        y_true_denorm = scalers[args.target].inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_denorm = scalers[args.target].inverse_transform(y_pred.reshape(-1, 1)).flatten()

        # Calculate metrics
        metrics = calculate_metrics(y_true_denorm, y_pred_denorm)

        print(f"\nFold {fold_idx} Metrics:")
        print(f"  MAE:   {metrics['mae']:.4f}")
        print(f"  RMSE:  {metrics['rmse']:.4f}")
        print(f"  NMAE:  {metrics['nmae']:.4f}")
        print(f"  NRMSE: {metrics['nrmse']:.4f}")
        print(f"  R²:    {metrics['r2']:.4f}")

        # Store results
        fold_results.append({
            'fold': fold_idx,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'pretrain_val_loss': float(pretrain_val_loss),
            'best_epoch': best_epoch,
            'best_val_loss': float(finetune_val_loss),
            'metrics': metrics,
            'training_history': {
                'train_loss': [float(x) for x in finetune_history['train_loss']],
                'val_loss': [float(x) for x in finetune_history['val_loss']]
            },
        })

        y_true_folds.append(y_true_denorm)
        y_pred_folds.append(y_pred_denorm)

    # =========================================================================
    # STEP 3: Aggregate Results
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 3: Aggregating Results")
    print("=" * 80)

    metrics_aggregated = calculate_metrics_with_std(y_true_folds, y_pred_folds)
    print_metrics_with_std(metrics_aggregated, f"{n_folds}-Fold Sliding Window CV Results")

    # =========================================================================
    # STEP 4: Save Results
    # =========================================================================
    print("\nSTEP 4: Saving Results")
    print("-" * 80)

    # Create experiment directory with standardized naming
    print("\nCreating experiment directory...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_paths = create_experiment_dir(
        model_type='autoencoder',
        config=model_config,
        building_id=args.building_id,
        phase=args.phase,
        experiment_type='kfold',
        timestamp=timestamp
    )

    experiment_name = experiment_paths['experiment_name']
    experiment_dir = experiment_paths['experiment_dir']
    checkpoint_dir = experiment_paths['checkpoints']
    figures_dir = experiment_paths['figures']

    # Add k_folds to experiment name
    experiment_name = experiment_name.replace(f"_{timestamp}", f"_{n_folds}fold_{timestamp}")

    print(f"✓ Experiment: {experiment_name}")
    print(f"✓ Directory: {experiment_dir}")

    # Save experiment info
    experiment_info = {
        'experiment_name': experiment_name,
        'model_type': 'autoencoder',
        'building_id': args.building_id,
        'target': args.target,
        'phase': args.phase,
        'n_folds': n_folds,
        'cv_method': 'sliding_window' if kfold_config['experiment'].get('use_sliding_window', True) else 'kfold',
        'cv_config': {
            'train_size': kfold_config['experiment'].get('train_size', None),
            'test_size': kfold_config['experiment'].get('test_size', None),
            'step': kfold_config['experiment'].get('step', None)
        } if kfold_config['experiment'].get('use_sliding_window', True) else {},
        'timestamp': timestamp,
        'model_config': model_config,
        'fold_results': [
            {
                'fold': r['fold'],
                'train_samples': r['train_samples'],
                'val_samples': r['val_samples'],
                'pretrain_val_loss': r['pretrain_val_loss'],
                'best_epoch': r['best_epoch'],
                'best_val_loss': r['best_val_loss'],
                'metrics': {k: float(v) for k, v in r['metrics'].items()},
                'training_history': r['training_history']
            }
            for r in fold_results
        ],
        'aggregated_metrics': {k: float(v) for k, v in metrics_aggregated.items()}
    }

    config_path = experiment_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(experiment_info, f, indent=2)
    print(f"✓ Config saved: {config_path}")

    # Save aggregated metrics CSV
    metrics_df = pd.DataFrame({
        'metric': list(metrics_aggregated.keys()),
        'value': [float(v) for v in metrics_aggregated.values()],
        
    })
    metrics_csv_path = experiment_dir / "aggregated_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"✓ Aggregated metrics saved: {metrics_csv_path}")

    # =========================================================================
    # STEP 5: Generate Plots
    # =========================================================================
    print("\nSTEP 5: Generating Plots")
    print("-" * 80)

    # Plot 1: K-Fold grid scatter
    fig, axes = plot_kfold_scatter(
        y_true_folds, y_pred_folds,
        fold_names=[f'Fold {i+1}' for i in range(n_folds)],
        title=f"LSTM Autoencoder - {args.target.replace('_', ' ').title()} - {n_folds}-Fold Sliding Window",
        save_path=figures_dir / 'fold_comparison.png',
        metrics_aggregated=metrics_aggregated
    )
    plt.close(fig)

    # Plot 2: Aggregated scatter (all folds combined)
    y_true_all = np.concatenate(y_true_folds)
    y_pred_all = np.concatenate(y_pred_folds)

    fig, ax = plot_scatter_predictions(
        y_true_all, y_pred_all,
        title=f"LSTM Autoencoder - {args.target.replace('_', ' ').title()}\n{n_folds}-Fold Sliding Window CV",
        save_path=figures_dir / 'aggregated_predictions.png',
        nmae_value=metrics_aggregated['nmae'],
        nrmse_value=metrics_aggregated['nrmse'],
        nstd_value=metrics_aggregated.get('nstd', None),
        
    )
    plt.close(fig)

    # Plot 3: Individual scatter for each fold (optional)
    if kfold_config['results']['save_individual_folds']:
        for i, (y_true, y_pred) in enumerate(zip(y_true_folds, y_pred_folds), 1):
            fig, ax = plot_scatter_predictions(
                y_true, y_pred,
                title=f"LSTM Autoencoder - Fold {i}",
                save_path=figures_dir / f'fold_{i}_predictions.png'
            )
            plt.close(fig)

        print(f"✓ Individual fold plots saved to: {figures_dir}")

    print(f"\n{'='*80}")
    print("TIME SERIES CROSS VALIDATION COMPLETED!")
    print(f"\nMethod: Sliding Window (avoids temporal leakage)")
    print(f"Folds: {n_folds}")
    print("=" * 80)
    print(f"\nResults saved to: {experiment_dir}")
    print(f"\nSummary (Metrics on ALL concatenated folds):")
    print(f"  NMAE:  {metrics_aggregated['nmae']:.4f}")
    print(f"  NRMSE: {metrics_aggregated['nrmse']:.4f}")
    print(f"  NSTD:  {metrics_aggregated['nstd']:.4f}")
    print(f"  R²:    {metrics_aggregated['r2']:.4f}")
    print()


if __name__ == '__main__':
    main()
