"""
K-Fold Sliding Window CV - Cross-Building

Train on Building 1 (K-fold splits) → Test on Building 2+3 (fused, always same)
Validates transfer learning robustness across different training subsets

Usage:
    python train_kfold_cross_building.py --model encoder_decoder --target cooling_demand
    python train_kfold_cross_building.py --model baseline --k_folds 5
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

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from data.preprocessing import prepare_dataset
from data.dataloader import get_dataloaders, get_test_dataloader
from models.lstm import LSTMModel
from models.lstm_encoder_decoder import LSTMEncoderDecoder
from models.lstm_attention import LSTMAttention
from models.transformer import TransformerModel
from models.lstm_autoencoder import LSTMAutoencoder
from training.trainer import Trainer, predict
from evaluation.metrics import calculate_metrics, denormalize_predictions
from evaluation.plotting import plot_scatter_predictions, plot_kfold_scatter
from utils.experiment import create_experiment_dir
from utils.cross_validation import TimeSeriesSlidingWindow, print_split_info


def load_model_config(model_type: str) -> dict:
    """Load model configuration"""
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


def create_model(model_type: str, input_size: int, config: dict, device: str):
    """Create model based on type"""
    if model_type == 'baseline':
        model = LSTMModel(
            input_size=input_size,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            output_size=config['model']['output_size']
        )
    elif model_type == 'encoder_decoder':
        model = LSTMEncoderDecoder(
            input_size=input_size,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            output_size=config['model']['output_size']
        )
    elif model_type == 'attention':
        model = LSTMAttention(
            input_size=input_size,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            output_size=config['model']['output_size']
        )
    elif model_type == 'transformer':
        model = TransformerModel(
            input_size=input_size,
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            num_layers=config['model']['num_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            output_size=config['model']['output_size'],
            use_weighted_pooling=config['model'].get('use_weighted_pooling', False)
        )
    elif model_type == 'autoencoder':
        model = LSTMAutoencoder(
            input_size=input_size,
            hidden_size=config['model']['hidden_size'],
            bottleneck_size=config['model']['bottleneck_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            output_size=config['model']['output_size']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


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
        description='K-Fold Sliding Window CV - Cross-Building',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--train_building', type=int, default=1, choices=[1, 2, 3],
                       help='Building to train on (default: 1)')
    parser.add_argument('--model', type=str, required=True,
                       choices=['baseline', 'encoder_decoder', 'attention', 'transformer', 'autoencoder'],
                       help='Model type')
    parser.add_argument('--target', type=str, default='cooling_demand',
                       choices=['cooling_demand'],
                       help='Target variable (only cooling_demand for cross-building)')
    parser.add_argument('--k_folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--phase', type=str, default='phase_1',
                       choices=['phase_1', 'phase_2_online'],
                       help='Phase to use (default: phase_1 for comparison with Single Building)')

    args = parser.parse_args()

    # Determine test buildings
    all_buildings = [1, 2, 3]
    test_buildings = [b for b in all_buildings if b != args.train_building]

    print("=" * 80)
    print("K-FOLD CROSS-VALIDATION - CROSS-BUILDING")
    print("=" * 80)
    print(f"Train Building: {args.train_building}")
    print(f"Test Buildings (fused): {test_buildings}")
    print(f"Model: {args.model}")
    print(f"Target: {args.target}")
    print(f"Phase: {args.phase}")
    print("=" * 80)
    print()

    kfold_config = load_kfold_config()
    model_config = load_model_config(args.model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    torch.manual_seed(kfold_config['experiment']['random_state'])
    np.random.seed(kfold_config['experiment']['random_state'])

    # STEP 1: Load training building dataset (for K-fold)
    print("STEP 1: Loading training dataset...")
    print("-" * 80)

    X_train_full, y_train_full, scalers, feature_names = prepare_dataset(
        building_id=args.train_building,
        phase=args.phase,
        target_variable=args.target,
        sequence_length=model_config['data']['sequence_length'],
        forecast_horizon=model_config['data']['forecast_horizon'],
        fit_scalers=True
    )

    print(f"Total samples: {len(X_train_full)}")
    print(f"Features: {X_train_full.shape[2]}")
    print(f"Feature names: {feature_names}\n")

    # STEP 2: Load test buildings dataset (fused, always same for all folds)
    print("STEP 2: Loading test dataset (fused)...")
    print("-" * 80)

    X_test_fused, y_test_fused = fuse_test_buildings(
        building_ids=test_buildings,
        phase=args.phase,
        target_variable=args.target,
        sequence_length=model_config['data']['sequence_length'],
        forecast_horizon=model_config['data']['forecast_horizon'],
        scalers=scalers
    )

    print(f"Test samples (fused B{test_buildings[0]}+B{test_buildings[1]}): {len(X_test_fused)}")
    print()

    # STEP 3: Time Series Cross Validation on training building
    print("STEP 3: Time Series Cross Validation (Training Building)")
    print("-" * 80)
    print()

    # Calculate step size
    n_folds = args.k_folds
    train_size = kfold_config['experiment']['train_size']
    test_size = kfold_config['experiment']['test_size']
    step = (1.0 - train_size - test_size) / (n_folds - 1)

    tscv = TimeSeriesSlidingWindow(
        train_size=train_size,
        test_size=test_size,
        step=step
    )

    n_splits = tscv.get_n_splits(X_train_full)
    print_split_info(tscv, len(X_train_full))

    # Storage for fold results
    fold_results = []
    all_y_test_true = []
    all_y_test_pred = []
    all_errors = []

    # K-Fold loop
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train_full), 1):
        print("\n" + "=" * 80)
        print(f"FOLD {fold_idx}/{n_splits}")
        print("=" * 80)

        # Split data for this fold (on training building)
        X_train_fold = X_train_full[train_idx]
        y_train_fold = y_train_full[train_idx]
        X_val_fold = X_train_full[val_idx]
        y_val_fold = y_train_full[val_idx]

        print(f"Train samples (B{args.train_building}): {len(X_train_fold)}")
        print(f"Val samples (B{args.train_building}): {len(X_val_fold)}")
        print(f"Test samples (B{test_buildings[0]}+B{test_buildings[1]}): {len(X_test_fused)}\n")

        # Create dataloaders
        train_loader, val_loader = get_dataloaders(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold,
            batch_size=model_config['training']['batch_size']
        )

        test_loader = get_test_dataloader(
            X_test_fused, y_test_fused,
            batch_size=model_config['training']['batch_size']
        )

        # Create model
        input_size = X_train_fold.shape[2]
        model = create_model(args.model, input_size, model_config, device)

        # Optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=model_config['training']['learning_rate'],
            weight_decay=model_config['training'].get('weight_decay', 0)
        )

        # Trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            checkpoint_dir='/tmp/kfold_checkpoints',  # Temporary dir for K-fold
            experiment_name=f'fold_{fold_idx}'
        )

        print(f"Training fold {fold_idx}...\n")
        history = trainer.train(
            num_epochs=model_config['training']['num_epochs'],
            early_stopping_patience=model_config['training']['early_stopping_patience'],
            clip_grad=model_config['training'].get('gradient_clip', 1.0)
        )

        best_epoch = np.argmin(history['val_loss']) + 1
        best_val_loss = history['val_loss'][best_epoch - 1]
        print(f"\n✓ Best val loss: {best_val_loss:.4f} at epoch {best_epoch}\n")

        # Predict on test buildings (fused)
        y_test_pred, _ = predict(model, test_loader, device)

        # Denormalize
        y_test_true_denorm = denormalize_predictions(
            y_test_fused, scalers, args.target
        )
        y_test_pred_denorm = denormalize_predictions(
            y_test_pred, scalers, args.target
        )

        # Calculate metrics for this fold
        metrics = calculate_metrics(y_test_true_denorm, y_test_pred_denorm)

        print(f"Fold {fold_idx} Metrics (Cross-Building Test):")
        print(f"  MAE:   {metrics['mae']:.4f}")
        print(f"  RMSE:  {metrics['rmse']:.4f}")
        print(f"  NMAE:  {metrics['nmae']:.4f}")
        print(f"  NRMSE: {metrics['nrmse']:.4f}")
        print(f"  R²:    {metrics['r2']:.4f}")

        # Store results
        fold_results.append({
            'fold': fold_idx,
            'metrics': metrics,
            'training_history': history
        })

        # Accumulate for aggregated metrics
        all_y_test_true.append(y_test_true_denorm)
        all_y_test_pred.append(y_test_pred_denorm)
        errors = y_test_pred_denorm - y_test_true_denorm
        all_errors.append(errors)

    # STEP 4: Aggregate results across all folds
    print("\n" + "=" * 80)
    print("STEP 4: Aggregating Results")
    print("=" * 80)

    # Concatenate all predictions
    all_y_test_true_concat = np.concatenate(all_y_test_true)
    all_y_test_pred_concat = np.concatenate(all_y_test_pred)
    all_errors_concat = np.concatenate(all_errors)

    # Calculate aggregated metrics
    aggregated_metrics = calculate_metrics(all_y_test_true_concat, all_y_test_pred_concat)

    # Calculate NSTD
    data_range = all_y_test_true_concat.max() - all_y_test_true_concat.min()
    nstd = np.std(all_errors_concat) / data_range if data_range > 0 else 0
    aggregated_metrics['nstd'] = nstd

    print()
    print("=" * 70)
    print(f"{n_splits}-Fold Sliding Window CV Results (Cross-Building)")
    print("=" * 70)
    print(f"Train: Building {args.train_building} | Test: Buildings {test_buildings[0]}+{test_buildings[1]} (fused)")
    print("Metriche calcolate su TUTTE le predizioni dei fold concatenate:")
    print("-" * 70)
    print(f"MAE       :   {aggregated_metrics['mae']:.4f}")
    print(f"RMSE      :   {aggregated_metrics['rmse']:.4f}")
    print(f"NMAE      :   {aggregated_metrics['nmae']:.4f}")
    print(f"NRMSE     :   {aggregated_metrics['nrmse']:.4f}")
    print(f"NSTD      :   {aggregated_metrics['nstd']:.4f}  (STD normalizzata degli errori)")
    print(f"R2        :   {aggregated_metrics['r2']:.4f}")
    print("=" * 70)

    # STEP 5: Save results
    print("\n\nSTEP 5: Saving Results")
    print("-" * 80)

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build experiment name
    if args.model == 'baseline':
        hs = model_config['model']['hidden_size']
        nl = model_config['model']['num_layers']
        dr = model_config['model']['dropout']
        exp_name = f"baseline_hs{hs}_nl{nl}_dr{dr}_train_b{args.train_building}_test_b{''.join(map(str, test_buildings))}_{n_folds}fold_{timestamp}"
    elif args.model == 'encoder_decoder':
        hs = model_config['model']['hidden_size']
        nl = model_config['model']['num_layers']
        dr = model_config['model']['dropout']
        exp_name = f"encoder_decoder_hs{hs}_nl{nl}_dr{dr}_train_b{args.train_building}_test_b{''.join(map(str, test_buildings))}_{n_folds}fold_{timestamp}"
    elif args.model == 'attention':
        hs = model_config['model']['hidden_size']
        nl = model_config['model']['num_layers']
        dr = model_config['model']['dropout']
        exp_name = f"attention_hs{hs}_nl{nl}_dr{dr}_train_b{args.train_building}_test_b{''.join(map(str, test_buildings))}_{n_folds}fold_{timestamp}"
    elif args.model == 'transformer':
        dm = model_config['model']['d_model']
        nh = model_config['model']['nhead']
        dr = model_config['model']['dropout']
        exp_name = f"transformer_dm{dm}_nh{nh}_dr{dr}_train_b{args.train_building}_test_b{''.join(map(str, test_buildings))}_{n_folds}fold_{timestamp}"
    elif args.model == 'autoencoder':
        hs = model_config['model']['hidden_size']
        bn = model_config['model']['bottleneck_size']
        nl = model_config['model']['num_layers']
        dr = model_config['model']['dropout']
        exp_name = f"autoencoder_hs{hs}_bn{bn}_nl{nl}_dr{dr}_train_b{args.train_building}_test_b{''.join(map(str, test_buildings))}_{n_folds}fold_{timestamp}"
    else:
        raise ValueError(f"Unsupported model type: {args.model}. Must be one of: baseline, encoder_decoder, attention, transformer, autoencoder")

    experiment_dir = Path(__file__).parent.parent.parent / 'results' / 'cross_building_kfold' / args.phase / exp_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / 'checkpoints').mkdir(exist_ok=True)
    (experiment_dir / 'figures').mkdir(exist_ok=True)

    print(f"✓ Experiment: {exp_name}")
    print(f"✓ Directory: {experiment_dir}")

    # Save config
    full_config = {
        'experiment_name': exp_name,
        'model_type': args.model,
        'train_building': args.train_building,
        'test_buildings': test_buildings,
        'target_variable': args.target,
        'phase': args.phase,
        'model_config': model_config,
        'cv_method': 'sliding_window',
        'cv_config': {
            'n_folds': n_splits,
            'train_size': train_size,
            'test_size': test_size,
            'step': step
        },
        'aggregated_metrics': aggregated_metrics,
        'fold_results': fold_results,
        'timestamp': timestamp
    }

    config_path = experiment_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(full_config, f, indent=2)
    print(f"✓ Config saved: {config_path}")

    # Save aggregated metrics
    metrics_df = pd.DataFrame([aggregated_metrics]).T
    metrics_df.columns = ['value']
    metrics_path = experiment_dir / 'aggregated_metrics.csv'
    metrics_df.to_csv(metrics_path, index_label='metric')
    print(f"✓ Aggregated metrics saved: {metrics_path}")

    # STEP 6: Generate plots
    print("\nSTEP 6: Generating Plots")
    print("-" * 80)

    # K-Fold comparison plot
    plot_kfold_scatter(
        all_y_test_true, all_y_test_pred,
        save_path=experiment_dir / 'figures' / 'fold_comparison.png',
        title=f'{args.model.upper()} - Cross-Building K-Fold CV\nTrain: B{args.train_building}, Test: B{test_buildings[0]}+B{test_buildings[1]}, Target: {args.target}'
    )
    print(f"✓ K-Fold scatter plot salvato: {experiment_dir / 'figures' / 'fold_comparison.png'}")

    # Aggregated predictions plot
    plot_scatter_predictions(
        all_y_test_true_concat, all_y_test_pred_concat,
        save_path=experiment_dir / 'figures' / 'aggregated_predictions.png',
        title=f'{args.model.upper()} - Cross-Building Aggregated Predictions\n{n_splits}-Fold CV (Train: B{args.train_building}, Test: B{test_buildings[0]}+B{test_buildings[1]})',
        nmae_value=aggregated_metrics.get('nmae'),
        nrmse_value=aggregated_metrics.get('nrmse'),
        nstd_value=aggregated_metrics.get('nstd')
    )
    print(f"✓ Scatter plot salvato: {experiment_dir / 'figures' / 'aggregated_predictions.png'}")

    # Individual fold plots
    for i, (y_true, y_pred) in enumerate(zip(all_y_test_true, all_y_test_pred), 1):
        fold_metrics = fold_results[i-1]['metrics']
        plot_scatter_predictions(
            y_true, y_pred,
            save_path=experiment_dir / 'figures' / f'fold_{i}_predictions.png',
            title=f'Fold {i}/{n_splits} - Cross-Building Predictions',
            nmae_value=fold_metrics.get('nmae'),
            nrmse_value=fold_metrics.get('nrmse'),
            nstd_value=fold_metrics.get('nstd')
        )
    print(f"✓ Individual fold plots saved to: {experiment_dir / 'figures'}")

    print("\n" + "=" * 80)
    print("K-FOLD CROSS-VALIDATION COMPLETED!")
    print("=" * 80)
    print()
    print(f"Method: Sliding Window (avoids temporal leakage)")
    print(f"Folds: {n_splits}")
    print(f"Train: Building {args.train_building}")
    print(f"Test: Buildings {test_buildings[0]}+{test_buildings[1]} (fused)")
    print()
    print(f"Results saved to: {experiment_dir}")
    print()
    print(f"Summary (Metrics on ALL concatenated folds - Cross-Building):")
    print(f"  NMAE:  {aggregated_metrics['nmae']:.4f}")
    print(f"  NRMSE: {aggregated_metrics['nrmse']:.4f}")
    print(f"  NSTD:  {aggregated_metrics['nstd']:.4f}")
    print(f"  R²:    {aggregated_metrics['r2']:.4f}")
    print()


if __name__ == '__main__':
    main()
