from MovieRecommender.constants import *
from MovieRecommender.utils.common import read_yaml, create_directories
from MovieRecommender.entity.config_entity import (
    DataIngestionConfig,
    DataPreparationConfig,
    ContentBasedModelConfig,
    CollaborativeFilteringConfig,
)


class ConfigurationManager:
    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config

    def get_data_preparation_config(self) -> DataPreparationConfig:
        data_preparation = self.config.data_preparation
        final_data_dir = os.path.dirname(data_preparation.ratings_data_path)
        create_directories(
            [
                Path(data_preparation.root_dir),
                Path(data_preparation.cache_dir),
                Path(final_data_dir),
            ]
        )
        folder = self.config.data_ingestion.source_URL.split("/")[-1].split(".zip")[0]
        movies_data = os.path.join(
            self.config.data_ingestion.unzip_dir, folder, "movies.csv"
        )
        links_data = os.path.join(
            self.config.data_ingestion.unzip_dir, folder, "links.csv"
        )
        ratings_data = os.path.join(
            self.config.data_ingestion.unzip_dir, folder, "ratings.csv"
        )
        data_preparation_config = DataPreparationConfig(
            root_dir=Path(data_preparation.root_dir),
            source_URL=data_preparation.source_URL,
            cache_dir=Path(data_preparation.cache_dir),
            movies_data=Path(movies_data),
            links_data=Path(links_data),
            ratings_data=Path(ratings_data),
            ratings_data_path=Path(data_preparation.ratings_data_path),
            movies_data_path=Path(data_preparation.movies_data_path),
            unique_category_path=Path(data_preparation.unique_category_path),
        )

        return data_preparation_config

    def get_content_based_model_config(self) -> ContentBasedModelConfig:
        content_based_model = self.config.content_based_model

        create_directories([content_based_model.root_dir])

        content_based_model_config = ContentBasedModelConfig(
            root_dir=Path(content_based_model.root_dir),
            movies_data=Path(self.config.data_preparation.movies_data_path),
            content_matrix=Path(content_based_model.content_matrix),
        )

        return content_based_model_config

    def get_collaborative_filtering_model_config(self) -> CollaborativeFilteringConfig:
        collab_filter_model = self.config.collaborative_filtering_model

        create_directories([collab_filter_model.root_dir])

        content_based_model_config = CollaborativeFilteringConfig(
            root_dir=Path(collab_filter_model.root_dir),
            movies_data=Path(self.config.data_preparation.movies_data_path),
            ratings_data=Path(self.config.data_preparation.ratings_data_path),
            user_movie_matrix=Path(collab_filter_model.user_movie_matrix),
            nearest_neighbors_model=Path(collab_filter_model.nearest_neighbors_model),
            nn_item_indices=Path(collab_filter_model.nn_item_indices),
            svd_model=Path(collab_filter_model.svd_model),
            svd_user_indices=Path(collab_filter_model.svd_user_indices),
            svd_item_indices=Path(collab_filter_model.svd_item_indices),
            params_min_user_rating=self.params.MIN_USER_RATING,
            params_min_movie_rating=self.params.MIN_MOVIE_RATING,
        )

        return content_based_model_config
