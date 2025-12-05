"""
Gestione errori personalizzata per il progetto

Questo modulo definisce eccezioni custom che forniscono messaggi di errore
più chiari e informativi rispetto alle eccezioni standard di Python.

Ogni eccezione personalizzata include:
- Messaggio descrittivo del problema
- Suggerimenti per risolvere l'errore
- Informazioni di contesto utili
"""

from pathlib import Path
from typing import List, Optional


class TesiBaseException(Exception):
    """
    Classe base per tutte le eccezioni custom del progetto.

    Fornisce un template comune per gestire gli errori in modo consistente.
    """
    def __init__(self, message: str, suggestion: Optional[str] = None):
        """
        Inizializza l'eccezione.

        Args:
            message: Descrizione del problema
            suggestion: Suggerimento per risolvere (opzionale)
        """
        self.message = message
        self.suggestion = suggestion

        # Costruzione messaggio formattato
        full_message = f"\n{'='*70}\nERRORE: {message}\n"
        if suggestion:
            full_message += f"\nSUGGERIMENTO: {suggestion}\n"
        full_message += f"{'='*70}\n"

        super().__init__(full_message)


class DataLoadError(TesiBaseException):
    """
    Errore durante il caricamento dei dati.

    Sollevata quando:
    - File CSV non trovato
    - File CSV corrotto o mal formattato
    - Colonne mancanti nel dataset
    """
    pass


class InvalidPhaseError(TesiBaseException):
    """
    Errore per fase dataset non valida.

    Sollevata quando si specifica una fase (es. "phase_1") che non esiste.
    """
    pass


class InvalidBuildingError(TesiBaseException):
    """
    Errore per building ID non valido.

    Sollevata quando si specifica un building ID diverso da 1, 2, o 3.
    """
    pass


class InvalidTargetError(TesiBaseException):
    """
    Errore per variabile target non valida.

    Sollevata quando si specifica una variabile target che non esiste
    nel dataset (es. una colonna non presente nel CSV).
    """
    pass


class PreprocessingError(TesiBaseException):
    """
    Errore durante il preprocessing dei dati.

    Sollevata quando:
    - Fallisce la normalizzazione
    - Fallisce la creazione delle sequenze
    - Dati mancanti o NaN non gestibili
    """
    pass


class ModelError(TesiBaseException):
    """
    Errore relativo ai modelli.

    Sollevata quando:
    - Checkpoint del modello non trovato
    - Architettura del modello incompatibile
    - Errori durante il caricamento dei pesi
    """
    pass


class ConfigError(TesiBaseException):
    """
    Errore nella configurazione.

    Sollevata quando:
    - File config YAML mal formattato
    - Parametri mancanti o non validi
    - Valori fuori range
    """
    pass


# ============================================================================
# FUNZIONI HELPER PER CREARE ERRORI INFORMATIVI
# ============================================================================

def raise_file_not_found(
    path: Path,
    file_type: str = "file",
    available_files: Optional[List[str]] = None
):
    """
    Solleva DataLoadError quando un file non viene trovato.

    Args:
        path: Percorso del file mancante
        file_type: Tipo di file (es. "CSV", "checkpoint", "config")
        available_files: Lista di file disponibili nella directory (opzionale)

    Esempio:
        >>> raise_file_not_found(
        ...     Path("data/Building_4.csv"),
        ...     file_type="CSV dati building",
        ...     available_files=["Building_1.csv", "Building_2.csv", "Building_3.csv"]
        ... )
    """
    message = f"{file_type} non trovato: {path}"

    # Suggerimento con lista file disponibili se presenti
    suggestion = "Verifica che il percorso sia corretto."
    if available_files:
        suggestion += "\n\nFile disponibili nella directory:\n"
        for f in available_files:
            suggestion += f"  - {f}\n"

    raise DataLoadError(message, suggestion)


def raise_invalid_phase(phase: str, valid_phases: List[str]):
    """
    Solleva InvalidPhaseError quando la fase specificata non è valida.

    Args:
        phase: Fase specificata dall'utente
        valid_phases: Lista delle fasi valide

    Esempio:
        >>> raise_invalid_phase("phase_4", ["phase_1", "phase_2_local", "phase_2_online"])
    """
    message = f"Fase non valida: '{phase}'"
    suggestion = f"Fasi valide: {', '.join(valid_phases)}"
    raise InvalidPhaseError(message, suggestion)


def raise_invalid_building(building_id: int, valid_buildings: List[int] = [1, 2, 3]):
    """
    Solleva InvalidBuildingError quando il building ID non è valido.

    Args:
        building_id: ID building specificato
        valid_buildings: Lista dei building ID validi

    Esempio:
        >>> raise_invalid_building(4)
    """
    message = f"Building ID non valido: {building_id}"
    suggestion = f"Building validi: {', '.join(map(str, valid_buildings))}"
    raise InvalidBuildingError(message, suggestion)


def raise_invalid_target(target: str, valid_targets: List[str]):
    """
    Solleva InvalidTargetError quando la variabile target non è valida.

    Args:
        target: Nome variabile target specificata
        valid_targets: Lista delle variabili target valide

    Esempio:
        >>> raise_invalid_target("invalid_target", ["cooling_demand", "heating_demand"])
    """
    message = f"Variabile target non valida: '{target}'"
    suggestion = f"Target validi: {', '.join(valid_targets)}"
    raise InvalidTargetError(message, suggestion)


def raise_csv_parse_error(path: Path, original_error: Exception):
    """
    Solleva DataLoadError quando il parsing CSV fallisce.

    Args:
        path: Percorso del file CSV
        original_error: Eccezione originale di pandas

    Esempio:
        >>> try:
        ...     df = pd.read_csv(path)
        ... except pd.errors.ParserError as e:
        ...     raise_csv_parse_error(path, e)
    """
    message = f"Errore parsing file CSV: {path}"
    suggestion = (
        f"Dettagli: {str(original_error)}\n\n"
        f"Possibili cause:\n"
        f"  - File corrotto o incompleto\n"
        f"  - Formato CSV invalido\n"
        f"  - Separatore non standard\n\n"
        f"Soluzione: Verifica file o riscarica dataset originale."
    )
    raise DataLoadError(message, suggestion)


def raise_missing_columns(path: Path, missing_cols: List[str], available_cols: List[str]):
    """
    Solleva DataLoadError quando mancano colonne nel CSV.

    Args:
        path: Percorso del file CSV
        missing_cols: Colonne mancanti
        available_cols: Colonne presenti nel file

    Esempio:
        >>> raise_missing_columns(
        ...     Path("data/Building_1.csv"),
        ...     missing_cols=["cooling_demand"],
        ...     available_cols=["timestamp", "temperature"]
        ... )
    """
    message = f"Colonne mancanti nel file {path.name}"
    suggestion = (
        f"Colonne richieste ma mancanti: {', '.join(missing_cols)}\n\n"
        f"Colonne disponibili nel file:\n"
    )
    for col in available_cols:
        suggestion += f"  - {col}\n"

    raise DataLoadError(message, suggestion)


def raise_checkpoint_not_found(checkpoint_path: Path):
    """
    Solleva ModelError quando il checkpoint del modello non viene trovato.

    Args:
        checkpoint_path: Percorso del checkpoint mancante

    Esempio:
        >>> raise_checkpoint_not_found(Path("results/experiment/model_best.pth"))
    """
    message = f"Checkpoint del modello non trovato: {checkpoint_path}"
    suggestion = (
        f"Possibili cause:\n"
        f"  - Il modello non è ancora stato trainato\n"
        f"  - Il percorso del checkpoint è errato\n"
        f"  - Il file è stato eliminato\n\n"
        f"Soluzione: Esegui il training prima di caricare il checkpoint."
    )
    raise ModelError(message, suggestion)
