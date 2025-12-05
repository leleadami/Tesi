# Building Energy Forecasting with Deep Learning

A comprehensive research project implementing and evaluating LSTM-based and Transformer architectures for multi-step building energy consumption forecasting using the CityLearn Challenge 2023 dataset.

## Overview

This project addresses the critical challenge of short-term energy demand forecasting in residential buildings through deep learning-based time series prediction. The work systematically compares five neural network architectures across three experimental scenarios: single-building forecasting, cross-building transfer learning, and neighborhood-level aggregation. The research employs rigorous time series cross-validation methodologies to ensure temporal integrity and prevent data leakage.

**Domain:** Smart Buildings, Energy Management Systems, Demand-Side Management

**Problem:** Predict hourly building energy consumption with 24-hour lookback window for 1-hour ahead forecasting

**Target Variables:**
- Cooling demand (kWh)
- Solar generation (kW)
- Carbon intensity (kg CO2/kWh)

## Dataset

**CityLearn Challenge 2023** - A benchmark dataset for building energy optimization and demand response.

- **Buildings:** 3 residential buildings in temperate climate zone
- **Temporal Coverage:** Summer period (June-August), hourly resolution
- **Data Phases:**
  - Phase 1: 720 hours (initial training)
  - Phase 2 Local: 2,208 hours (extended validation)
  - Phase 2 Online: 2,208 hours (test set)
- **Features:** 37 variables including weather data (temperature, solar irradiance, humidity), temporal encodings (month, hour, day type), and building characteristics
- **Dataset Source:** [CityLearn Challenge 2023](https://www.aicrowd.com/challenges/citylearn-challenge-2023)

**Data Preprocessing:**
- Removed 5 constant/redundant features (heating demand, HVAC mode, setpoints) achieving 38% dimensionality reduction
- Mixed normalization strategy: MinMaxScaler for bounded variables, StandardScaler for unbounded loads
- Temporal feature engineering with cyclic encoding for periodicity preservation

## Models

The project implements five architectures with increasing complexity:

### 1. LSTM Baseline
Vanilla LSTM architecture serving as performance baseline. Uses the last hidden state for regression output.

**Architecture:** LSTM layers → Dropout → Fully Connected → Output

**Reference:** Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.

### 2. LSTM Encoder-Decoder
Sequence-to-sequence architecture with separate encoding and decoding phases. Best overall performer with 8.1% RMSE improvement over baseline.

**Architecture:** Encoder LSTM → Context Vector → Decoder LSTM → Output

**Key Innovation:** Explicit separation of representation learning and prediction phases

### 3. LSTM Attention
Implements Bahdanau-style additive attention mechanism allowing dynamic focus on relevant input timesteps. Achieves 7.5% RMSE improvement.

**Architecture:** Encoder LSTM → Attention Mechanism → Context Vector → Decoder → Output

**Reference:** Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural Machine Translation by Jointly Learning to Align and Translate. arXiv:1409.0473.

### 4. LSTM Autoencoder
Bottleneck architecture with two-phase training (pre-training + fine-tuning). Underperforms due to information compression constraints.

**Architecture:** Encoder → Bottleneck → Decoder → Forecast Head

**Training:** Unsupervised pre-training followed by supervised fine-tuning

### 5. Transformer
Multi-head self-attention architecture with positional encoding for time series modeling.

**Architecture:** Positional Encoding → Multi-Head Self-Attention → Feed-Forward → Output

**Reference:** Vaswani, A., et al. (2017). Attention is All You Need. In NeurIPS.

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+ (CPU or CUDA-enabled GPU)

### Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd tesi

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install PyTorch (select appropriate version)
# CPU version:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8 version:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 version:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### Single Building Training

Train models on individual buildings for baseline performance assessment:

```bash
# LSTM Encoder-Decoder (recommended - best performer)
python experiments/encoder_decoder/train_encoder_decoder.py --hidden_size 128 --building_id 1

# LSTM Baseline
python experiments/baseline/train_baseline.py --hidden_size 128 --building_id 1

# LSTM Attention
python experiments/attention/train_attention.py --hidden_size 128 --building_id 1

# Transformer
python experiments/transformer/train_transformer.py --d_model 128 --nhead 8 --building_id 1

# LSTM Autoencoder
python experiments/autoencoder/train_autoencoder.py --hidden_size 128 --bottleneck_size 32 --building_id 1
```

### Cross-Building Transfer Learning

Evaluate model generalization across different buildings:

```bash
# Train on Building 1, test on Buildings 2+3 (fused)
python experiments/cross_building/train_cross_building.py \
    --train_building 1 \
    --model encoder_decoder

# Systematic evaluation across all building combinations
for b in 1 2 3; do
    python experiments/cross_building/train_cross_building.py \
        --train_building $b \
        --model encoder_decoder \
        --target cooling_demand
done
```

### Neighborhood Aggregation

Train on aggregated data from all buildings, test on individual buildings:

```bash
# Solar generation (recommended - physically aggregatable)
python experiments/neighborhood/train_neighborhood.py \
    --model encoder_decoder \
    --target solar_generation

# All supported targets
for target in cooling_demand solar_generation carbon_intensity; do
    python experiments/neighborhood/train_neighborhood.py \
        --model encoder_decoder \
        --target $target
done
```

### Time Series Cross-Validation

Sliding window cross-validation with temporal integrity preservation:

```bash
# Single building K-fold CV (default: 5 folds)
python experiments/encoder_decoder/train_kfold_encoder_decoder.py \
    --building_id 1 \
    --target cooling_demand \
    --k_folds 5

# Custom fold configuration
python experiments/encoder_decoder/train_kfold_encoder_decoder.py \
    --building_id 1 \
    --target cooling_demand \
    --k_folds 10

# Neighborhood aggregation K-fold CV
python experiments/neighborhood/train_kfold_neighborhood.py \
    --model encoder_decoder \
    --target solar_generation \
    --k_folds 5
```

## Project Structure

```
tesi/
├── configs/                        # YAML configuration files
│   ├── baseline.yaml              # LSTM baseline hyperparameters
│   ├── encoder_decoder.yaml       # Encoder-decoder configuration
│   ├── attention.yaml             # Attention mechanism settings
│   ├── transformer.yaml           # Transformer hyperparameters
│   ├── autoencoder.yaml           # Autoencoder architecture
│   ├── kfold.yaml                 # Cross-validation settings
│   ├── cross_building.yaml        # Transfer learning configuration
│   └── neighborhood.yaml          # Aggregation experiment settings
│
├── data/                          # Dataset storage (not tracked)
│   ├── citylearn_challenge_2023_phase_1/
│   ├── citylearn_challenge_2023_phase_2_local/
│   └── citylearn_challenge_2023_phase_2_online/
│
├── experiments/                   # Training scripts (14 total)
│   ├── baseline/                  # LSTM baseline experiments
│   │   ├── train_baseline.py
│   │   └── train_kfold_baseline.py
│   ├── encoder_decoder/           # Encoder-decoder experiments
│   ├── attention/                 # Attention mechanism experiments
│   ├── transformer/               # Transformer experiments
│   ├── autoencoder/               # Autoencoder experiments
│   ├── cross_building/            # Transfer learning experiments
│   │   ├── train_cross_building.py
│   │   └── train_kfold_cross_building.py
│   └── neighborhood/              # Aggregation experiments
│       ├── train_neighborhood.py
│       └── train_kfold_neighborhood.py
│
├── src/                           # Source code modules
│   ├── data/                      # Data loading and preprocessing
│   │   └── preprocessing.py       # Complete preprocessing pipeline
│   ├── models/                    # Neural network architectures
│   │   ├── lstm.py               # LSTM baseline
│   │   ├── lstm_encoder_decoder.py
│   │   ├── lstm_attention.py
│   │   ├── lstm_autoencoder.py
│   │   └── transformer.py
│   ├── training/                  # Training utilities
│   │   └── trainer.py            # Training loop with early stopping
│   ├── evaluation/                # Metrics and visualization
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── plotting.py           # Result visualization
│   └── utils/                     # Helper modules
│       ├── constants.py          # Centralized constants
│       ├── error_handling.py     # Custom exception hierarchy
│       ├── experiment.py         # Experiment management
│       └── cross_validation.py   # Time series CV utilities
│
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_data_analysis.ipynb
│   ├── 02_target_selection.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Results_Analysis.ipynb
│
├── scripts/                       # Analysis and visualization scripts
│   ├── compare_training_methods.py
│   └── generate_final_summary.py
│
├── results/                       # Experiment outputs (generated)
│   ├── single_building/
│   ├── cross_building/
│   ├── neighborhood/
│   └── kfold/
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Results and Artifacts

Each experiment generates a timestamped directory containing:

```
results/{experiment_type}/{phase}/{experiment_name}/
├── config.json                    # Complete experiment configuration
├── metrics.csv                    # Performance metrics
├── training_history.json          # Epoch-by-epoch training log
├── checkpoints/
│   └── model_best.pth            # Best model checkpoint
└── figures/
    ├── loss_curve.png            # Training and validation loss
    ├── predictions.png           # True vs predicted scatter plot
    └── residuals.png             # Residual analysis
```

**K-Fold Cross-Validation Results:**

```
results/kfold/{phase}/{experiment_name}/
├── config.json                    # Includes per-fold training history
├── aggregated_metrics.csv         # Metrics computed on concatenated folds
├── figures/
│   ├── fold_comparison.png                   # Grid of all folds
│   ├── aggregated_predictions.png            # Combined predictions
│   ├── training_history_all_folds.png        # Overfitting detection
│   └── fold_N_predictions.png                # Individual fold results
└── checkpoints/                   # Empty (CV does not save models)
```

**Experiment Naming Convention:**

`{model}_{hyperparams}_{timestamp}/`

Hyperparameter abbreviations:
- `hs{N}`: hidden_size
- `nl{N}`: num_layers
- `dr{X}`: dropout
- `dm{N}`: d_model (Transformer)
- `nh{N}`: nhead (Transformer)
- `bn{N}`: bottleneck_size (Autoencoder)
- `b{N}`: building_id

Example: `encoder_decoder_hs128_nl2_dr0.2_b1_20251027_163000`

## Configuration

All models support both YAML configuration files and command-line argument overrides:

```bash
# Use default configuration from configs/encoder_decoder.yaml
python experiments/encoder_decoder/train_encoder_decoder.py --building_id 1

# Override specific hyperparameters
python experiments/encoder_decoder/train_encoder_decoder.py \
    --building_id 1 \
    --hidden_size 256 \
    --num_layers 3 \
    --dropout 0.3 \
    --batch_size 64 \
    --lr 0.0005

# Custom phases
python experiments/encoder_decoder/train_encoder_decoder.py \
    --building_id 1 \
    --phase phase_2_online
```

## Evaluation Metrics

The project implements comprehensive evaluation metrics following best practices in time series forecasting:

### Point Forecast Metrics
- **MAE** (Mean Absolute Error): Average absolute deviation
- **RMSE** (Root Mean Squared Error): Penalizes large errors quadratically
- **NMAE** (Normalized MAE): MAE / range(y), scale-independent comparison
- **NRMSE** (Normalized RMSE): RMSE / range(y), scale-independent comparison
- **MAPE** (Mean Absolute Percentage Error): Percentage-based error
- **SMAPE** (Symmetric MAPE): Bounded percentage error metric

### Statistical Metrics
- **R²** (Coefficient of Determination): Proportion of variance explained
- **NSTD** (Normalized Standard Deviation): std(errors) / range(y) - CV stability metric

### Interpretation Guidelines
- **NMAE**: 5-15% indicates good performance, <5% may indicate overfitting
- **NRMSE**: 8-20% is typical for energy forecasting, <8% requires verification
- **R²**: 0.70-0.90 is excellent for building energy, >0.95 may indicate data leakage

## Key Findings

### Single Building Performance (Phase 1 → Phase 2)

| Model | RMSE | R² | NMAE | Notes |
|-------|------|-----|------|-------|
| **Encoder-Decoder** | 0.7110 | 0.8105 | 9.2% | Best performer, +8.1% vs baseline |
| **Attention** | 0.7154 | 0.8082 | 9.5% | +7.5% vs baseline |
| **Baseline** | 0.7454 | 0.7918 | 10.8% | Reference architecture |
| **Transformer** | 0.7523 | 0.7845 | 11.1% | Limited data regime |
| **Autoencoder** | 0.9825 | 0.5673 | 15.9% | Information bottleneck issue |

### Cross-Building Generalization

- **Average R²:** 0.6105 (19% performance drop from in-building)
- **Best combination:** B3 → B1+B2 (R² = 0.75)
- **Worst combination:** B1 → B2+B3 (R² = 0.52)
- **Key insight:** Architectural heterogeneity limits transfer learning effectiveness

### Neighborhood Aggregation

- **Average R²:** 0.8393 (excellent generalization)
- **Best target:** Solar generation (R² = 0.87, physically meaningful aggregation)
- **Carbon intensity:** R² = 0.60 (network-level variable, weaker correlation)
- **Key advantage:** Diverse training data improves robustness vs cross-building

## Methodology

### Time Series Cross-Validation

Critical methodological contribution: this project implements **sliding window cross-validation** to prevent temporal leakage. Standard K-fold cross-validation is invalid for time series as it violates temporal causality.

**Problem with Standard K-Fold:**
- Training on future data to predict the past
- Observed validation loss variance: 4.7x between folds (0.06 → 0.29)
- Artificially inflated performance metrics

**Sliding Window Solution:**
```
Configuration: train_size=40%, test_size=20%, step=10%
Fold 1: train=[0-40%],   test=[40-60%]
Fold 2: train=[10-50%],  test=[50-70%]
Fold 3: train=[20-60%],  test=[60-80%]
...
```

**Advantages:**
- Temporal ordering preserved (training always precedes testing)
- Constant train/test sizes across folds
- Realistic evaluation of model performance on unseen future data

**Reference:** Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. Information Sciences, 191, 192-213.

### Data Normalization Strategy

Mixed normalization approach based on feature characteristics:

- **Weather variables** (bounded): MinMaxScaler [0,1] - preserves physical constraints
- **Load variables** (unbounded): StandardScaler - handles outliers robustly
- **Temporal features**: Cyclic encoding (sin/cos) for periodicity preservation

### Feature Engineering

**Temporal Encoding:**
- Hour: Cyclic encoding (0-23 → sin/cos)
- Month: One-hot encoding (limited range)
- Day type: Binary encoding (weekday/weekend)

**Feature Selection:**
- Removed 5 constant/redundant features (38% reduction)
- Retained 37 informative features balancing dimensionality vs information content

## Analysis Tools

### Jupyter Notebooks

1. **01_data_analysis.ipynb**: Exploratory data analysis, distribution analysis, correlation matrices
2. **02_target_selection.ipynb**: Target variable selection methodology, autocorrelation analysis
3. **03_Model_Training.ipynb**: Training workflow demonstrations, hyperparameter sensitivity
4. **04_Results_Analysis.ipynb**: Comprehensive results analysis, statistical testing

### Utility Scripts

```bash
# Compare single building vs cross-building vs neighborhood performance
python scripts/compare_training_methods.py --building_id 1

# Generate comprehensive K-fold CV summary with statistical analysis
python scripts/generate_final_summary.py
```

## Dependencies

### Core Scientific Computing
- [NumPy](https://numpy.org/) >= 1.24.0 - Numerical computing
- [Pandas](https://pandas.pydata.org/) >= 2.0.0 - Data manipulation
- [Scikit-learn](https://scikit-learn.org/) >= 1.3.0 - Preprocessing and metrics

### Deep Learning
- [PyTorch](https://pytorch.org/) >= 2.0.0 - Neural network framework (install separately)

### Visualization
- [Matplotlib](https://matplotlib.org/) >= 3.7.0 - Plotting
- [Seaborn](https://seaborn.pydata.org/) >= 0.12.0 - Statistical visualization

### Configuration
- [PyYAML](https://pyyaml.org/) >= 6.0 - YAML parsing

See `requirements.txt` for complete dependency list.

## Usage Notes

### Help and Documentation

All training scripts provide detailed help:

```bash
python experiments/baseline/train_baseline.py --help
python experiments/cross_building/train_cross_building.py --help
python experiments/neighborhood/train_kfold_neighborhood.py --help
```

### Common Pitfalls

1. **Temporal Leakage**: Always use sliding window CV for time series (not standard K-fold)
2. **Data Normalization**: Ensure scalers are fit on training set only, then applied to validation/test
3. **Early Stopping**: Default patience=15 tuned for Phase 1 (720h), adjust for longer sequences
4. **Autoencoder Mode**: Use `mode='forecast'` (default) for prediction tasks

### Performance Optimization

- **Batch Size**: Larger batches (64-128) improve GPU utilization but may reduce generalization
- **Hidden Size**: 128-256 provides good capacity/efficiency tradeoff
- **Sequence Length**: 24 hours (1 day) captures daily patterns without excessive memory

## References

### Scientific Publications

1. **LSTM Architecture:**
   Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.

2. **Attention Mechanism:**
   Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural Machine Translation by Jointly Learning to Align and Translate. arXiv:1409.0473.

3. **Transformer:**
   Vaswani, A., et al. (2017). Attention is All You Need. In Advances in Neural Information Processing Systems (NeurIPS).

4. **Time Series Cross-Validation:**
   Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. Information Sciences, 191, 192-213.

5. **Building Energy Forecasting:**
   Ahmad, T., et al. (2018). A review on renewable energy and electricity demand forecasting models for smart grid and buildings. Sustainable Cities and Society, 43, 657-670.

### Dataset and Frameworks

- [CityLearn Challenge 2023](https://www.aicrowd.com/challenges/citylearn-challenge-2023) - Competition dataset
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Deep learning framework
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html) - Machine learning library
- [Pandas Documentation](https://pandas.pydata.org/docs/) - Data analysis library
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split) - Sklearn implementation

### Additional Resources

- [Energy Forecasting Best Practices](https://www.sciencedirect.com/science/article/pii/S0306261918317082)
- [Deep Learning for Time Series](https://arxiv.org/abs/2004.13408)
- [Building Energy Modeling](https://www.ashrae.org/)

## Development

### Code Quality

- Type annotations throughout codebase (Python 3.8+ syntax)
- Comprehensive docstrings (Google style)
- Centralized error handling with custom exception hierarchy
- Configuration management via YAML + argparse

### Project Documentation

- `README.md`: User-facing documentation (this file)
- `FEATURE_ANALYSIS.md`: Feature engineering methodology
- `CARBON_INTENSITY_IMPLEMENTATION.md`: Carbon intensity integration details

## Author

This project was developed as a thesis research on building energy forecasting using deep learning methods. The work systematically evaluates state-of-the-art neural architectures for short-term energy demand prediction in residential buildings.

**Academic Context:** Thesis project on Smart Buildings and Energy Management Systems

## Acknowledgments

Dataset provided by the CityLearn Challenge 2023 organizers. The project builds upon foundational work in LSTM-based sequence modeling, attention mechanisms, and transformer architectures.
