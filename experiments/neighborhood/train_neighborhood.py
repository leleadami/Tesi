"""
Neighborhood Aggregation Training Script (Phase 2 Online Only)

Train on ALL 3 buildings FUSED together → Test on EACH building separately

Setup:
- Train: B1+B2+B3 fused (Phase 2 online, 2208h × 3 = 6624h, 80/20 split)
- Test: B1, B2, B3 separately (Phase 2 online, 2208h each)
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
from data.preprocessing import prepare_dataset
from data.dataloader import get_dataloaders, get_test_dataloader
from models.lstm import LSTMModel
from models.lstm_encoder_decoder import LSTMEncoderDecoder
from models.lstm_attention import LSTMAttention
from models.transformer import TransformerModel
from models.lstm_autoencoder import LSTMAutoencoder
from training.trainer import Trainer, predict
from evaluation.metrics import calculate_all_metrics, denormalize_predictions, print_metrics
from utils.experiment import get_neighborhood_experiment_name


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


def fuse_all_buildings(phase: str, target_variable: str,
                       sequence_length: int, forecast_horizon: int, fit_scalers: bool = True):
    """
    Fuse training data from all 3 buildings into single dataset

    NOTA: Per carbon_intensity, non fonde i building (è già aggregato),
    ma usa direttamente prepare_carbon_intensity_dataset()

    Args:
        phase: Phase name (phase_2_online)
        target_variable: Target variable name
        sequence_length: Sequence length
        forecast_horizon: Forecast horizon
        fit_scalers: Whether to fit new scalers

    Returns:
        X_fused, y_fused, scalers, feature_names: Concatenated data (o singolo dataset per carbon_intensity)
    """
    # Caso speciale: carbon_intensity è già aggregato a livello di rete
    if target_variable == 'carbon_intensity':
        from data.preprocessing import prepare_carbon_intensity_dataset
        X, y, scalers, feature_names = prepare_carbon_intensity_dataset(
            phase=phase,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            scalers=None,
            fit_scalers=fit_scalers
        )
        return X, y, scalers, feature_names

    # Caso normale: cooling_demand, solar_generation (per-building → aggregati)
    X_list = []
    y_list = []
    scalers = None
    feature_names = None

    # Load and fuse all 3 buildings
    for building_id in [1, 2, 3]:
        X, y, scaler, features = prepare_dataset(
            building_id=building_id,
            phase=phase,
            target_variable=target_variable,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            scalers=scalers,
            fit_scalers=(fit_scalers and building_id == 1)  # Fit only on first building
        )
        X_list.append(X)
        y_list.append(y)

        if building_id == 1:
            scalers = scaler
            feature_names = features

    # Concatenate along batch dimension
    X_fused = np.concatenate(X_list, axis=0)
    y_fused = np.concatenate(y_list, axis=0)

    return X_fused, y_fused, scalers, feature_names


def main():
    parser = argparse.ArgumentParser(
        description='Neighborhood Aggregation Training (Phase 2 online only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on B1+B2+B3 fused, test on each building separately
  python train_neighborhood.py --model encoder_decoder --hidden_size 128

  # With custom hyperparameters
  python train_neighborhood.py --model attention --hidden_size 256 --lr 0.0005

Results saved to: results/neighborhood/
        """
    )

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
    parser.add_argument('--target', type=str, default='solar_generation',
                       choices=['cooling_demand', 'solar_generation', 'heating_demand'],
                       help='Target variable (default: solar_generation for neighborhood)')

    args = parser.parse_args()

    # Always use phase_2_online for neighborhood
    phase = 'phase_2_online'

    print("="*80)
    print("NEIGHBORHOOD AGGREGATION TRAINING (Phase 2 Online)")
    print("="*80)
    print(f"\n✓ Train: B1+B2+B3 fused")
    print(f"✓ Test: B1, B2, B3 separately")
    print(f"✓ Model: {args.model}")
    print(f"✓ Phase: {phase}")
    print(f"✓ Target: {args.target}")

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
    # 1. LOAD AND FUSE TRAINING DATA (all 3 buildings, Phase 2 online)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: Loading and fusing training data")
    print("="*80)

    # Load and aggregate data with train/val split BEFORE normalization (prevents data leakage - FIXED!)
    print(f"\nLoading and aggregating {phase} data from Buildings 1, 2, 3...")

    # Special case: carbon_intensity is network-level (not per-building)
    if args.target == 'carbon_intensity':
        from data.preprocessing import prepare_train_val_split
        X_train, X_val, y_train, y_val, scalers, feature_names = prepare_train_val_split(
            building_id=1,  # Dummy value (carbon_intensity uses separate dataset)
            phase=phase,
            target_variable=args.target,
            sequence_length=config['data']['sequence_length'],
            forecast_horizon=config['data']['forecast_horizon'],
            val_split=0.2
        )
    else:
        # Normal case: aggregate buildings (cooling_demand, solar_generation)
        from data.preprocessing import prepare_neighborhood_train_val_split
        X_train, X_val, y_train, y_val, scalers, feature_names = prepare_neighborhood_train_val_split(
            building_ids=[1, 2, 3],
            phase=phase,
            target_variable=args.target,
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
    experiment_name = get_neighborhood_experiment_name(
        model_type=args.model,
        config=config,
        target_variable=args.target
    )

    # Create directory structure
    project_root = Path(__file__).parent.parent.parent
    experiment_dir = project_root / 'results' / 'neighborhood' / 'phase_2_online' / experiment_name
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
    # 4. LOAD BEST MODEL AND TEST ON EACH BUILDING SEPARATELY
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: Testing on each building separately")
    print("="*80)

    # Load best model
    best_model_path = Path(checkpoint_dir) / "model_best.pth"
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded best model (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.6f})")

    # Test on each building separately
    test_results = {}

    for building_id in [1, 2, 3]:
        print(f"\n--- Testing on Building {building_id} ---")

        # Load test data
        X_test, y_test, _, _ = prepare_dataset(
            building_id=building_id,
            phase=phase,
            target_variable=args.target,
            sequence_length=config['data']['sequence_length'],
            forecast_horizon=config['data']['forecast_horizon'],
            scalers=scalers,
            fit_scalers=False
        )

        # Create test loader
        test_loader = get_test_dataloader(X_test, y_test,
                                         batch_size=config['training']['batch_size'])

        # Predict
        y_pred, y_true = predict(model, test_loader, device)

        # Denormalize
        target_scaler = scalers[args.target]
        y_pred_orig = denormalize_predictions(y_pred, target_scaler)
        y_true_orig = denormalize_predictions(y_true, target_scaler)

        # Metrics
        metrics = calculate_all_metrics(y_true_orig, y_pred_orig)
        print_metrics(metrics, title=f"Building {building_id} Test Metrics")

        # Store results
        test_results[f'building_{building_id}'] = {
            'y_true': y_true_orig,
            'y_pred': y_pred_orig,
            'metrics': metrics
        }

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
        'train_buildings': [1, 2, 3],
        'test_buildings': [1, 2, 3],
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
            'target_variable': args.target,
            'sequence_length': config['data']['sequence_length'],
            'forecast_horizon': config['data']['forecast_horizon'],
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples_per_building': 2184  # 2208h with sequence_length=24
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
    metrics_dict = {key: test_results[key]['metrics'] for key in test_results}
    metrics_df = pd.DataFrame(metrics_dict).T
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

    # Aggregate predictions from all 3 buildings for scatter plot
    y_true_all = np.concatenate([test_results[f'building_{i}']['y_true'] for i in [1, 2, 3]])
    y_pred_all = np.concatenate([test_results[f'building_{i}']['y_pred'] for i in [1, 2, 3]])

    # Calculate aggregate metrics
    avg_nmae = np.mean([test_results[f'building_{i}']['metrics']['nmae'] for i in [1, 2, 3]])
    avg_nrmse = np.mean([test_results[f'building_{i}']['metrics']['nrmse'] for i in [1, 2, 3]])

    # Predictions scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true_all, y_pred_all, alpha=0.95, s=40, edgecolors='darkblue', linewidth=0.6, color='#1E5F8C')

    min_val = min(y_true_all.min(), y_pred_all.min())
    max_val = max(y_true_all.max(), y_pred_all.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Ground Truth (kW)', fontweight='bold', fontsize=12)
    plt.ylabel('Prediction (kW)', fontweight='bold', fontsize=12)
    plt.title(f'Neighborhood Test (All Buildings): Ground Truth vs Predictions\nNMAE: {avg_nmae:.4f} | NRMSE: {avg_nrmse:.4f}',
              fontweight='bold', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    pred_plot_path = figures_dir / "predictions.png"
    plt.savefig(pred_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Predictions scatter plot saved: {pred_plot_path}")

    # Summary
    print("\n" + "="*80)
    print("✓ NEIGHBORHOOD TRAINING COMPLETED!")
    print("="*80)
    print(f"\nExperiment: {experiment_name}")
    print(f"\nPer-Building Test Results:")
    for building_id in [1, 2, 3]:
        m = test_results[f'building_{building_id}']['metrics']
        print(f"  Building {building_id}: RMSE={m['rmse']:.4f} kW, R²={m['r2']:.4f}")

    avg_rmse = np.mean([test_results[f'building_{i}']['metrics']['rmse'] for i in [1, 2, 3]])
    avg_r2 = np.mean([test_results[f'building_{i}']['metrics']['r2'] for i in [1, 2, 3]])
    print(f"\nAverage: RMSE={avg_rmse:.4f} kW, R²={avg_r2:.4f}")

    print(f"\nAll results saved in:")
    print(f"  {experiment_dir}")
    print(f"\nContents:")
    print(f"  - config.json: Experiment configuration")
    print(f"  - metrics.csv: Performance metrics per building")
    print(f"  - training_history.json: Training history data")
    print(f"  - checkpoints/model_best.pth: Best model weights")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
