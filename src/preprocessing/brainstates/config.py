from pathlib import Path

SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-05")
NETWORKS = ("VIS", "SMN", "DAN", "SAL", "LIM", "FPC", "DN")
TR_SECONDS = 1.49
TOKENS = {
    "VIS": "_Vis_", # Visual network
    "SMN": "_SomMot_", # Somatomotor network
    "DAN": "_DorsAttn_", # Dorsal attention network
    "SAL": "_SalVentAttn_", # Ventral attention network/salience network
    "LIM": "_Limbic_", # Limbic network
    "FPC": "_Cont_", # Frontoparietal Control network
    "DN": "_Default_", # Default mode network
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data/algonauts_2025.competitors"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CACHE_ROOT = PROJECT_ROOT / "data/cache"

