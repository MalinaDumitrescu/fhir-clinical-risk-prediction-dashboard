from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

ARTIFACTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

FEATURES_CSV = ARTIFACTS_DIR / "features.csv"

MODEL_PATH = ARTIFACTS_DIR / "risk_model.joblib"

METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

MODEL_COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.json"
OPTUNA_TRIALS_CSV = ARTIFACTS_DIR / "optuna_trials.csv"

RISK_ASSESSMENTS_PATH = ARTIFACTS_DIR / "RiskAssessment.ndjson"