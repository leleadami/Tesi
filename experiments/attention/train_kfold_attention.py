"""
K-Fold Cross Validation - LSTM Attention

Usage:
    python train_kfold_attention.py --building_id 1 --target cooling_demand --k_folds 5 --num_epochs 50
    python train_kfold_attention.py --hidden_size 256 --building_id 2
    python train_kfold_attention.py --target solar_generation
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import KFold

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from data.preprocessing import prepare_dataset
from data.dataloader import get_dataloaders
from models.lstm_attention import LSTMAttention
from training.trainer import Trainer, predict
from evaluation.metrics import calculate_metrics, calculate_metrics_with_std, print_metrics_with_std
from evaluation.plotting import plot_scatter_predictions, plot_kfold_scatter
from utils.experiment import create_experiment_dir
from utils.cross_validation import TimeSeriesSlidingWindow, print_split_info


def load_config() -> dict:
    """Load attention model configuration"""
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'attention.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_kfold_config() -> dict:
    """Load K-fold experiment configuration"""
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'kfold.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='K-Fold Cross Validation - LSTM Attention',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--building_id', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--target', type=str, default='cooling_demand',
                       choices=['cooling_demand', 'carbon_intensity', 'solar_generation'])
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--phase', type=str, default='phase_1',
                       choices=['phase_1', 'phase_2_local', 'phase_2_online'])
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)

    args = parser.parse_args()

    print("=" * 80)
    print("TIME SERIES CROSS VALIDATION - LSTM ATTENTION")
    print("=" * 80)
    print(f"Building: {args.building_id}")
    print(f"Target: {args.target}")
    print(f"K-Folds: {args.k_folds}")
    print(f"Phase: {args.phase}")
    print("=" * 80)
    print()

    kfold_config = load_kfold_config()
    model_config = load_config()

    if args.hidden_size is not None:
        model_config['model']['hidden_size'] = args.hidden_size
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
    # STEP 2: Time Series Cross Validation Setup
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

    # Create experiment directory BEFORE fold loop
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_config_with_kfold = model_config.copy()
    model_config_with_kfold['k_folds'] = args.k_folds

    experiment_paths = create_experiment_dir(
        model_type='attention',
        config=model_config,
        building_id=args.building_id,
        phase=args.phase,
        experiment_type='kfold',
        timestamp=timestamp
    )

    # Rename directory to include n_folds
    experiment_name_base = experiment_paths['experiment_name']
    experiment_name_with_folds = experiment_name_base.replace(f"_{timestamp}", f"_{n_folds}fold_{timestamp}")

    experiment_dir_base = experiment_paths['experiment_dir']
    experiment_dir_with_folds = experiment_dir_base.parent / experiment_name_with_folds

    # Rename directory
    experiment_dir_base.rename(experiment_dir_with_folds)

    # Update paths
    experiment_paths['experiment_dir'] = experiment_dir_with_folds
    experiment_paths['experiment_name'] = experiment_name_with_folds
    checkpoint_dir = experiment_dir_with_folds / 'checkpoints'
    figures_dir = experiment_dir_with_folds / 'figures'

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_full), 1):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx}/{n_folds}")
        print(f"{'='*80}")

        X_train, X_val = X_full[train_idx], X_full[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]

        print(f"Train samples: {len(X_train)}")
        print(f"Val samples: {len(X_val)}")

        train_loader, val_loader = get_dataloaders(
            X_train, y_train, X_val, y_val,
            batch_size=model_config['training']['batch_size']
        )

        input_size = X_train.shape[2]
        model = LSTMAttention(
            input_size=input_size,
            hidden_size=model_config['model']['hidden_size'],
            num_layers=model_config['model']['num_layers'],
            dropout=model_config['model']['dropout'],
            output_size=model_config['model']['output_size']
        )
        model = model.to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=model_config['training']['learning_rate'],
            weight_decay=model_config['training']['weight_decay']
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            checkpoint_dir=str(checkpoint_dir)
        )

        print(f"\nTraining fold {fold_idx}...\n")

        history = trainer.train(
            num_epochs=model_config['training']['num_epochs'],
            early_stopping_patience=model_config['training']['early_stopping_patience'],
            clip_grad=model_config['training']['gradient_clip']
        )

        best_val_loss = min(history['val_loss'])
        best_epoch = history['val_loss'].index(best_val_loss) + 1
        print(f"\n✓ Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

        y_pred, y_true = predict(model, val_loader, device)

        y_true_denorm = scalers[args.target].inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_denorm = scalers[args.target].inverse_transform(y_pred.reshape(-1, 1)).flatten()

        metrics = calculate_metrics(y_true_denorm, y_pred_denorm)

        print(f"\nFold {fold_idx} Metrics:")
        print(f"  MAE:   {metrics['mae']:.4f}")
        print(f"  RMSE:  {metrics['rmse']:.4f}")
        print(f"  NMAE:  {metrics['nmae']:.4f}")
        print(f"  NRMSE: {metrics['nrmse']:.4f}")
        print(f"  R²:    {metrics['r2']:.4f}")

        fold_results.append({
            'fold': fold_idx,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'metrics': metrics,
            'training_history': {
                'train_loss': [float(x) for x in history['train_loss']],
                'val_loss': [float(x) for x in history['val_loss']],
                'learning_rate': [float(x) for x in history['learning_rate']]
            }
        })

        y_true_folds.append(y_true_denorm)
        y_pred_folds.append(y_pred_denorm)

    print(f"\n{'='*80}")
    print("STEP 4: Aggregating Results")
    print("=" * 80)

    metrics_aggregated = calculate_metrics_with_std(y_true_folds, y_pred_folds)
    print_metrics_with_std(metrics_aggregated, f"{n_folds}-Fold Cross Validation Results")

    print("\nSTEP 5: Saving Results")
    print("-" * 80)

    # Extract experiment info (already created before fold loop with n_folds in name)
    experiment_name = experiment_paths['experiment_name']
    experiment_dir = experiment_paths['experiment_dir']

    print(f"✓ Experiment: {experiment_name}")
    print(f"✓ Directory: {experiment_dir}")

    experiment_info = {
        'experiment_name': experiment_name,
        'model_type': 'attention',
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
                'best_epoch': r['best_epoch'],
                'best_val_loss': float(r['best_val_loss']),
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

    metrics_df = pd.DataFrame({
        'metric': list(metrics_aggregated.keys()),
        'value': [float(v) for v in metrics_aggregated.values()],
        
    })
    metrics_csv_path = experiment_dir / "aggregated_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"✓ Aggregated metrics saved: {metrics_csv_path}")

    print("\nSTEP 6: Generating Plots")
    print("-" * 80)

    fig, axes = plot_kfold_scatter(
        y_true_folds, y_pred_folds,
        fold_names=[f'Fold {i+1}' for i in range(n_folds)],
        title=f"LSTM Attention - {args.target.replace('_', ' ').title()} - {n_folds}-Fold Sliding Window",
        save_path=figures_dir / 'fold_comparison.png',
        metrics_aggregated=metrics_aggregated
    )
    plt.close(fig)

    y_true_all = np.concatenate(y_true_folds)
    y_pred_all = np.concatenate(y_pred_folds)

    fig, ax = plot_scatter_predictions(
        y_true_all, y_pred_all,
        title=f"LSTM Attention - {args.target.replace('_', ' ').title()}\n{n_folds}-Fold Sliding Window CV",
        save_path=figures_dir / 'aggregated_predictions.png',
        nmae_value=metrics_aggregated['nmae'],
        nrmse_value=metrics_aggregated['nrmse'],
        nstd_value=metrics_aggregated.get('nstd', None),

    )
    plt.close(fig)

    if kfold_config['results']['save_individual_folds']:
        for i, (y_true, y_pred) in enumerate(zip(y_true_folds, y_pred_folds), 1):
            fig, ax = plot_scatter_predictions(
                y_true, y_pred,
                title=f"LSTM Attention - Fold {i}",
                save_path=figures_dir / f'fold_{i}_predictions.png'
            )
            plt.close(fig)

        print(f"✓ Individual fold plots saved to: {figures_dir}")

    print(f"\n{'='*80}")
    print("TIME SERIES CROSS VALIDATION COMPLETED!")
    print("=" * 80)
    print(f"\nMethod: Sliding Window (avoids temporal leakage)")
    print(f"Folds: {n_folds}")
    print(f"\nResults saved to: {experiment_dir}")
    print(f"\nSummary (Metrics on ALL concatenated folds):")
    print(f"  NMAE:  {metrics_aggregated['nmae']:.4f}")
    print(f"  NRMSE: {metrics_aggregated['nrmse']:.4f}")
    print(f"  NSTD:  {metrics_aggregated['nstd']:.4f}")
    print(f"  R²:    {metrics_aggregated['r2']:.4f}")
    print()


if __name__ == '__main__':
    main()
