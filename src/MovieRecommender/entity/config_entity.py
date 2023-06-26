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
    ratings_data_path: Path
    movies_data_path: Path
    unique_category_path: Path

@dataclass(frozen=True)
class ContentBasedModelConfig:
    root_dir: Path
    movies_data: Path
    content_matrix: Path

@dataclass(frozen=True)
class CollaborativeFilteringConfig:
    root_dir: Path
    movies_data: Path
    ratings_data: Path
    user_movie_matrix: Path
    nearest_neighbors_model: Path
    nn_item_indices: Path
    svd_model: Path
    svd_user_indices: Path
    svd_item_indices: Path
    params_min_user_rating: int
    params_min_movie_rating: int
