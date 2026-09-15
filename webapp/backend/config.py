import os
import secrets
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("GRAPHDQN_DATA_DIR", WEBAPP_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = DATA_DIR / "graphdqn.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

CHECKPOINT_ROOT = str(DATA_DIR / "checkpoints" / "dqn")
Path(CHECKPOINT_ROOT).mkdir(parents=True, exist_ok=True)

JWT_SECRET = os.environ.get("GRAPHDQN_JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

CORS_ORIGINS = os.environ.get("GRAPHDQN_CORS_ORIGINS", "*").split(",")

MAX_TRAINING_WORKERS = int(os.environ.get("GRAPHDQN_MAX_WORKERS", "4"))
