from MovieRecommender.constants import *
from MovieRecommender.utils.common import read_yaml, create_directories
from MovieRecommender.entity.config_entity import (
    DataIngestionConfig,
    DataPreparationConfig,
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
        final_data_dir = os.path.dirname(data_preparation.final_data_path)
        create_directories(
            [
                Path(data_preparation.root_dir),
                Path(data_preparation.cache_dir),
                Path(final_data_dir),
            ]
        )
        movies_data = os.path.join(
            self.config.data_ingestion.unzip_dir, "ml-latest/movies.csv"
        )
        links_data = os.path.join(
            self.config.data_ingestion.unzip_dir, "ml-latest/links.csv"
        )
        ratings_data = os.path.join(
            self.config.data_ingestion.unzip_dir, "ml-latest/ratings.csv"
        )
        data_preparation_config = DataPreparationConfig(
            root_dir=Path(data_preparation.root_dir),
            source_URL=data_preparation.source_URL,
            cache_dir=Path(data_preparation.cache_dir),
            movies_data=Path(movies_data),
            links_data=Path(links_data),
            ratings_data=Path(ratings_data),
            final_data_path=Path(data_preparation.final_data_path),
            unique_category_path=Path(data_preparation.unique_category_path),
        )

        return data_preparation_config
