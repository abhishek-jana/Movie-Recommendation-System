from pathlib import Path
import os

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

API_KEY = os.environ.get("TMDB_API_KEY")
