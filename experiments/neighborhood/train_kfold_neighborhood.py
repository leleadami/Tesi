"""
K-Fold Sliding Window CV - Neighborhood Aggregation

Aggrega i dati dai 3 building (B1+B2+B3) e fa K-fold cross validation.

Usage:
    python train_kfold_neighborhood.py --model encoder_decoder --target solar_generation --k_folds 5
    python train_kfold_neighborhood.py --model baseline --target carbon_intensity --k_folds 5
    python train_kfold_neighborhood.py --model transformer --target solar_generation
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

from data.preprocessing import prepare_dataset, prepare_carbon_intensity_dataset
from data.dataloader import get_dataloaders
from models.lstm import LSTMModel
from models.lstm_encoder_decoder import LSTMEncoderDecoder
from models.lstm_attention import LSTMAttention
from models.transformer import TransformerModel
from models.lstm_autoencoder import LSTMAutoencoder
from training.trainer import Trainer, predict
from evaluation.metrics import calculate_metrics, calculate_metrics_with_std, print_metrics_with_std
from evaluation.plotting import plot_scatter_predictions, plot_kfold_scatter
from utils.experiment import create_experiment_dir
from utils.cross_validation import TimeSeriesSlidingWindow, print_split_info


def load_config(model_type: str) -> dict:
    """Load configuration for specified model type"""
    config_map = {
        'baseline': 'baseline.yaml',
        'encoder_decoder': 'encoder_decoder.yaml',
        'attention': 'attention.yaml',
        'transformer': 'transformer.yaml',
        'autoencoder': 'autoencoder.yaml'
    }

    config_path = Path(__file__).parent.parent.parent / 'configs' / config_map[model_type]
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_kfold_config() -> dict:
    """Load K-fold experiment configuration"""
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'kfold.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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


def fuse_all_buildings(phase: str, target_variable: str,
                       sequence_length: int, forecast_horizon: int,
                       weather_features: list, fit_scalers: bool = True):
    """
    Fuse data from all 3 buildings into single aggregated dataset

    Args:
        phase: Phase name
        target_variable: Target variable name
        sequence_length: Sequence length
        forecast_horizon: Forecast horizon
        weather_features: Weather features to include
        fit_scalers: Whether to fit new scalers

    Returns:
        X_fused, y_fused, scalers, feature_names: Concatenated data
    """
    # Caso speciale: carbon_intensity è già aggregato a livello di rete
    if target_variable == 'carbon_intensity':
        print(f"Loading carbon_intensity dataset (network-level, no per-building fusion)...")
        print("-" * 80)

        X, y, scalers, feature_names = prepare_carbon_intensity_dataset(
            phase=phase,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            scalers=None,
            fit_scalers=fit_scalers
        )

        print(f"  Total samples: {len(X)}")
        print()

        return X, y, scalers, feature_names

    # Caso normale: cooling_demand, solar_generation (per-building → fusi)
    X_list = []
    y_list = []
    scalers = None
    feature_names = None

    print(f"Loading and fusing data from 3 buildings...")
    print("-" * 80)

    # Load and fuse all 3 buildings
    for building_id in [1, 2, 3]:
        X, y, scaler, features = prepare_dataset(
            building_id=building_id,
            phase=phase,
            target_variable=target_variable,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            weather_features=weather_features,
            scalers=scalers,
            fit_scalers=(fit_scalers and building_id == 1)  # Fit only on first building
        )
        X_list.append(X)
        y_list.append(y)

        print(f"  Building {building_id}: {len(X)} samples")

        if building_id == 1:
            scalers = scaler
            feature_names = features

    # Ensure scalers were initialized (should always be set from building_id == 1)
    assert scalers is not None, "Scalers were not initialized during data loading"
    assert feature_names is not None, "Feature names were not initialized during data loading"

    # Concatenate along batch dimension
    X_fused = np.concatenate(X_list, axis=0)
    y_fused = np.concatenate(y_list, axis=0)

    print(f"  Total fused: {len(X_fused)} samples")
    print()

    return X_fused, y_fused, scalers, feature_names


def main():
    parser = argparse.ArgumentParser(
        description='K-Fold Sliding Window CV - Neighborhood Aggregation (B1+B2+B3)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', type=str, required=True,
                       choices=['baseline', 'encoder_decoder', 'attention', 'transformer', 'autoencoder'],
                       help='Model type')
    parser.add_argument('--target', type=str, default='solar_generation',
                       choices=['cooling_demand', 'carbon_intensity', 'solar_generation', 'heating_demand'],
                       help='Target variable to predict (default: solar_generation)')
    parser.add_argument('--k_folds', type=int, default=5,
                       help='Number of folds for cross validation (default: 5)')
    parser.add_argument('--phase', type=str, default='phase_1',
                       choices=['phase_1', 'phase_2_local', 'phase_2_online'],
                       help='Data phase (default: phase_1)')

    # Optional hyperparameter overrides
    parser.add_argument('--hidden_size', type=int, default=None,
                       help='Hidden size for LSTM (overrides config, default: 128)')
    parser.add_argument('--d_model', type=int, default=None,
                       help='d_model for Transformer (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')

    args = parser.parse_args()

    print("=" * 80)
    print("TIME SERIES CROSS VALIDATION - NEIGHBORHOOD AGGREGATION (B1+B2+B3)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Target: {args.target}")
    print(f"Phase: {args.phase}")
    print("=" * 80)
    print()

    kfold_config = load_kfold_config()
    model_config = load_config(args.model)

    # Override configs with command line args
    if args.model == 'transformer':
        if args.d_model is not None:
            model_config['model']['d_model'] = args.d_model
    else:
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

    # =========================================================================
    # STEP 1: Prepare aggregated dataset from all 3 buildings
    # =========================================================================
    print("STEP 1: Loading and aggregating dataset from B1+B2+B3...")
    print("-" * 80)

    X_full, y_full, scalers, feature_names = fuse_all_buildings(
        phase=args.phase,
        target_variable=args.target,
        sequence_length=kfold_config['data']['sequence_length'],
        forecast_horizon=kfold_config['data']['forecast_horizon'],
        weather_features=kfold_config['data']['weather_features']
    )

    print(f"Total aggregated samples: {len(X_full)}")
    print(f"Features: {X_full.shape[2]}")
    print(f"Feature names: {feature_names}")
    print()

    # =========================================================================
    # STEP 2: Time Series Cross Validation Setup
    # =========================================================================
    print(f"STEP 2: Time Series Cross Validation on Aggregated Data")
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
        model_type=args.model,
        config=model_config,
        building_id=0,  # 0 indicates neighborhood aggregation (all buildings)
        phase=args.phase,
        experiment_type='kfold',
        timestamp=timestamp
    )

    # Rename directory to include n_folds
    experiment_name_base = experiment_paths['experiment_name']
    experiment_name_with_folds = experiment_name_base.replace(f"_{timestamp}", f"_{n_folds}fold_{timestamp}")

    experiment_dir_base = experiment_paths['experiment_dir']
    experiment_dir_with_folds = experiment_dir_base.parent / experiment_name_with_folds

    # Rename directory (only if source exists and destination doesn't)
    if experiment_dir_base.exists() and not experiment_dir_with_folds.exists():
        experiment_dir_base.rename(experiment_dir_with_folds)
    elif experiment_dir_with_folds.exists():
        # Destination already exists, use it
        pass
    else:
        # Neither exists, keep base name
        experiment_dir_with_folds = experiment_dir_base

    # Update paths
    experiment_paths['experiment_dir'] = experiment_dir_with_folds
    experiment_paths['experiment_name'] = experiment_name_with_folds
    checkpoint_dir = experiment_dir_with_folds / 'checkpoints'
    figures_dir = experiment_dir_with_folds / 'figures'

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
        model = create_model(args.model, input_size, model_config)
        model = model.to(device)

        # Optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=model_config['training']['learning_rate'],
            weight_decay=model_config['training']['weight_decay']
        )

        # Trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            checkpoint_dir=str(checkpoint_dir)
        )

        print(f"\nTraining fold {fold_idx}...\n")

        # Train
        history = trainer.train(
            num_epochs=model_config['training']['num_epochs'],
            early_stopping_patience=model_config['training']['early_stopping_patience'],
            clip_grad=model_config['training']['gradient_clip']
        )

        # Extract best epoch and loss
        best_val_loss = min(history['val_loss'])
        best_epoch = history['val_loss'].index(best_val_loss) + 1
        print(f"\n✓ Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

        # Predict on validation set
        y_pred, y_true = predict(model, val_loader, device)

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
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'metrics': metrics,
            'training_history': {
                'train_loss': [float(x) for x in history['train_loss']],
                'val_loss': [float(x) for x in history['val_loss']],
                'learning_rate': [float(x) for x in history['learning_rate']]
            },
        })

        y_true_folds.append(y_true_denorm)
        y_pred_folds.append(y_pred_denorm)

    # =========================================================================
    # STEP 4: Aggregate Results
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 4: Aggregating Results")
    print("=" * 80)

    metrics_aggregated = calculate_metrics_with_std(y_true_folds, y_pred_folds)
    print_metrics_with_std(metrics_aggregated, f"{n_folds}-Fold Sliding Window CV Results (Aggregated B1+B2+B3)")

    # =========================================================================
    # STEP 5: Save Results
    # =========================================================================
    print("\nSTEP 5: Saving Results")
    print("-" * 80)

    # Extract experiment info (already created before fold loop with n_folds in name)
    experiment_name = experiment_paths['experiment_name']
    experiment_dir = experiment_paths['experiment_dir']

    print(f"✓ Experiment: {experiment_name}")
    print(f"✓ Directory: {experiment_dir}")

    # Save experiment info
    experiment_info = {
        'experiment_name': experiment_name,
        'model_type': args.model,
        'neighborhood_aggregation': True,
        'buildings': [1, 2, 3],
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

    # Save aggregated metrics CSV
    metrics_df = pd.DataFrame({
        'metric': list(metrics_aggregated.keys()),
        'value': [float(v) for v in metrics_aggregated.values()],
        
    })
    metrics_csv_path = experiment_dir / "aggregated_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"✓ Aggregated metrics saved: {metrics_csv_path}")

    # =========================================================================
    # STEP 6: Generate Plots
    # =========================================================================
    print("\nSTEP 6: Generating Plots")
    print("-" * 80)

    # Map model types to display names
    model_display_names = {
        'baseline': 'LSTM Baseline',
        'encoder_decoder': 'LSTM Encoder-Decoder',
        'attention': 'LSTM Attention',
        'transformer': 'Transformer'
    }
    model_display_name = model_display_names.get(args.model, args.model.upper())

    # Plot 1: K-Fold grid scatter
    fig, axes = plot_kfold_scatter(
        y_true_folds, y_pred_folds,
        fold_names=[f'Fold {i+1}' for i in range(n_folds)],
        title=f"{model_display_name} - {args.target.replace('_', ' ').title()} (B1+B2+B3 Aggregated) - {n_folds}-Fold Sliding Window",
        save_path=figures_dir / 'fold_comparison.png',
        metrics_aggregated=metrics_aggregated
    )
    plt.close(fig)

    # Plot 2: Aggregated scatter (all folds combined)
    y_true_all = np.concatenate(y_true_folds)
    y_pred_all = np.concatenate(y_pred_folds)

    fig, ax = plot_scatter_predictions(
        y_true_all, y_pred_all,
        title=f"{model_display_name} - {args.target.replace('_', ' ').title()} (B1+B2+B3 Aggregated)\n{n_folds}-Fold Sliding Window CV",
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
                title=f"{model_display_name} (B1+B2+B3) - Fold {i}",
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
    print(f"\nSummary (Aggregated B1+B2+B3):")
    print(f"  NMAE:  {metrics_aggregated['nmae']:.4f}")
    print(f"  NRMSE: {metrics_aggregated['nrmse']:.4f}")
    print(f"  NSTD:  {metrics_aggregated['nstd']:.4f}")
    print(f"  R²:    {metrics_aggregated['r2']:.4f}")
    print()


if __name__ == '__main__':
    main()
