from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataPreparationConfig:
    root_dir: Path
    source_URL: str
    cache_dir: Path
    movies_data: Path
    links_data: Path
    ratings_data: Path
    final_data_path: Path
    unique_category_path: Path
