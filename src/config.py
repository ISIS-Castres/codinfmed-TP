from pathlib import Path

# ════════════════════ ENTITÉS ════════════════════
ENTITY_TYPES = ["PERSON","PLACE","SPORT"]

# ════════════════════ CHEMINS PRINCIPAUX ════════════════════
BASE_DIR = Path(__file__).resolve().parent.parent

FILE_NAME = "example"
DATA_PATH = BASE_DIR / "input" / f"{FILE_NAME}.txt"
OUTPUT_DIR = BASE_DIR / "output"
INPUT_DIR = BASE_DIR / "input"

PRED_PATH = OUTPUT_DIR / f"{FILE_NAME}_predictions.json"
TRUE_PATH = INPUT_DIR / f"{FILE_NAME}_gold_standard.json"
NOMENCLATURE_PATH = INPUT_DIR / "nomenclature.json"

# ════════════════════ MODÈLE ET SEUILS ════════════════════
MODELS = [
    {"name": "urchade/gliner_small", "backend": "gliner"},
    {"name": "numind/NuNER-large", "backend": "nunER"},
    {"name": "GAIR/uniner-base", "backend": "uniner"},
]
SMALL_GLINER = "knowledgator/gliner-bi-small-v1.0"
THRESHOLD = 0.6  # seuil de prédiction

# ════════════════════ VIZ COLORS ════════════════════
PRED_COLORS = ["#85C1E9","#FF6961","#F8C471","#77DD77","#CBAACB","#FFB347","#AEC6CF","#CFCFC4"]
    
TRUE_COLORS = ["#1B4F72", "#8B0000", "#B9770E", "#228B22", "#800080", "#FF8C00", "#779ECB", "#808080"]
