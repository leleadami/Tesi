"""
Utilità per la gestione degli esperimenti

Questo modulo fornisce funzioni per creare nomi standardizzati di directory
degli esperimenti e gestire gli artefatti (checkpoint, metriche, figure).

Convenzione di naming delle directory:
    {model}_{hyperparams}_{timestamp}/

Abbreviazioni iperparametri:
    - hs{N}: hidden_size (es. hs128)
    - nl{N}: num_layers (es. nl2)
    - dr{X}: dropout (es. dr0.2)
    - b{N}: building_id (es. b1)
    - bs{N}: batch_size (es. bs32) - solo se diverso dal default
    - lr{X}: learning_rate (es. lr0.001) - solo se diverso dal default
    - dm{N}: d_model per Transformer (es. dm128)
    - nh{N}: nhead per Transformer (es. nh8)

Esempi:
    baseline_hs128_nl2_dr0.2_b1_20251026_163000
    encoder_decoder_hs256_nl3_dr0.3_bs64_b2_20251026_170000
    transformer_dm128_nh8_dr0.2_b1_20251026_180000
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union, TypedDict


class ExperimentPaths(TypedDict):
    """Tipo per i percorsi restituiti da create_experiment_dir()."""
    experiment_dir: Path
    checkpoints: Path
    figures: Path
    experiment_name: str


def get_experiment_name(
    model_type: str,
    config: Dict,
    building_id: int,
    phase: str,
    timestamp: Optional[str] = None,
    include_non_default: bool = True
) -> str:
    """
    Genera nome standardizzato dell'esperimento basato su tipo modello e iperparametri.

    Args:
        model_type: Tipo di modello ('baseline', 'encoder_decoder', 'attention', 'autoencoder', 'transformer')
        config: Dizionario di configurazione contenente parametri modello e training
        building_id: ID del building (1, 2, o 3)
        phase: Fase di training ('phase_1', 'phase_2_local', 'phase_2_online')
        timestamp: Timestamp opzionale. Se None, usa il tempo corrente.
        include_non_default: Se includere iperparametri non-default (batch_size, lr)

    Returns:
        Stringa con il nome dell'esperimento

    Esempi:
        >>> config = {'model': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2},
        ...           'training': {'batch_size': 32, 'learning_rate': 0.001}}
        >>> get_experiment_name('baseline', config, building_id=1, phase='phase_1')
        'baseline_hs128_nl2_dr0.2_b1_20251026_163000'
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = [model_type]

    # Iperparametri specifici per tipo modello
    if model_type == 'transformer':
        d_model = config['model'].get('d_model', 128)
        nhead = config['model'].get('nhead', 8)
        dropout = config['model'].get('dropout', 0.2)

        parts.extend([f"dm{d_model}", f"nh{nhead}", f"dr{dropout}"])

    elif model_type == 'autoencoder':
        hidden_size = config['model'].get('hidden_size', 128)
        bottleneck_size = config['model'].get('bottleneck_size', 32)
        num_layers = config['model'].get('num_layers', 2)
        dropout = config['model'].get('dropout', 0.2)

        parts.extend([f"hs{hidden_size}", f"bn{bottleneck_size}",
                      f"nl{num_layers}", f"dr{dropout}"])

    else:  # Modelli LSTM (baseline, encoder_decoder, attention)
        hidden_size = config['model'].get('hidden_size', 128)
        num_layers = config['model'].get('num_layers', 2)
        dropout = config['model'].get('dropout', 0.2)

        parts.extend([f"hs{hidden_size}", f"nl{num_layers}", f"dr{dropout}"])

    # Iperparametri training non-default (opzionale)
    if include_non_default:
        batch_size = config['training'].get('batch_size', 32)
        learning_rate = config['training'].get('learning_rate', 0.001)

        if batch_size != 32:
            parts.append(f"bs{batch_size}")
        if learning_rate != 0.001:
            lr_str = f"{learning_rate}".rstrip('0').rstrip('.')
            parts.append(f"lr{lr_str}")

    # Building e timestamp
    parts.extend([f"b{building_id}", timestamp])

    return "_".join(parts)


def create_experiment_dir(
    model_type: str,
    config: Dict,
    building_id: int,
    phase: str,
    experiment_type: str = 'single_building',
    timestamp: Optional[str] = None,
    base_dir: Optional[Path] = None
) -> ExperimentPaths:
    """
    Crea la struttura di directory dell'esperimento e restituisce i percorsi.

    Args:
        model_type: Tipo di modello ('baseline', 'encoder_decoder', 'attention', ecc.)
        config: Dizionario di configurazione
        building_id: ID del building
        phase: Fase di training
        experiment_type: Tipo di esperimento ('single_building', 'cross_building', 'neighborhood', 'kfold')
        timestamp: Timestamp opzionale. Se None, usa il tempo corrente.
        base_dir: Directory base per i risultati. Se None, usa ../../../results da questo file.

    Returns:
        Dizionario con percorsi:
            - 'experiment_dir': Directory principale dell'esperimento (Path)
            - 'checkpoints': Subdirectory checkpoint (Path)
            - 'figures': Subdirectory figure (Path)
            - 'experiment_name': Stringa nome esperimento (str)

    Struttura directory creata:
        results/{experiment_type}/{phase}/{experiment_name}/
            ├── checkpoints/
            └── figures/
    """
    if base_dir is None:
        # Default: usa results/ nella radice del progetto
        base_dir = Path(__file__).parent.parent.parent / 'results'

    # Generazione nome esperimento standardizzato
    experiment_name = get_experiment_name(
        model_type=model_type,
        config=config,
        building_id=building_id,
        phase=phase,
        timestamp=timestamp
    )

    # Costruzione struttura directory
    experiment_dir = base_dir / experiment_type / phase / experiment_name
    checkpoints_dir = experiment_dir / 'checkpoints'
    figures_dir = experiment_dir / 'figures'

    # Creazione directory (parents=True crea percorsi intermedi)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    return {
        'experiment_dir': experiment_dir,
        'checkpoints': checkpoints_dir,
        'figures': figures_dir,
        'experiment_name': experiment_name
    }


def get_cross_building_experiment_name(
    model_type: str,
    config: Dict,
    train_building: int,
    test_buildings: list,
    timestamp: Optional[str] = None
) -> str:
    """
    Genera nome esperimento per esperimenti cross-building.

    Args:
        model_type: Tipo di modello
        config: Dizionario di configurazione
        train_building: Building usato per training (1, 2, o 3)
        test_buildings: Lista di building usati per testing (es. [2, 3])
        timestamp: Timestamp opzionale

    Returns:
        Stringa con nome esperimento

    Esempio:
        >>> get_cross_building_experiment_name('encoder_decoder', config, 1, [2, 3])
        'encoder_decoder_hs128_train_b1_test_b23_20251026_163000'
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = [model_type]

    # Aggiunge iperparametri chiave
    if model_type == 'transformer':
        d_model = config['model'].get('d_model', 128)
        parts.append(f"dm{d_model}")
    else:
        hidden_size = config['model'].get('hidden_size', 128)
        parts.append(f"hs{hidden_size}")

    # Aggiunge info sui building di train/test
    test_buildings_str = ''.join(map(str, sorted(test_buildings)))
    parts.append(f"train_b{train_building}")
    parts.append(f"test_b{test_buildings_str}")

    # Aggiunge timestamp
    parts.append(timestamp)

    return "_".join(parts)


def get_neighborhood_experiment_name(
    model_type: str,
    config: Dict,
    target_variable: str,
    timestamp: Optional[str] = None
) -> str:
    """
    Genera nome esperimento per esperimenti di aggregazione neighborhood.

    Args:
        model_type: Tipo di modello
        config: Dizionario di configurazione
        target_variable: Variabile target predetta
        timestamp: Timestamp opzionale

    Returns:
        Stringa con nome esperimento

    Esempio:
        >>> get_neighborhood_experiment_name('encoder_decoder', config, 'solar_generation')
        'encoder_decoder_hs128_target_solar_20251026_163000'
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = [model_type]

    # Aggiunge iperparametri chiave
    if model_type == 'transformer':
        d_model = config['model'].get('d_model', 128)
        parts.append(f"dm{d_model}")
    else:
        hidden_size = config['model'].get('hidden_size', 128)
        parts.append(f"hs{hidden_size}")

    # Aggiunge variabile target (abbreviata)
    target_short = target_variable.split('_')[0]  # es. 'solar' da 'solar_generation'
    parts.append(f"target_{target_short}")

    # Aggiunge timestamp
    parts.append(timestamp)

    return "_".join(parts)
