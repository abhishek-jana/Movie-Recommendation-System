from MovieRecommender.config.configuration import ConfigurationManager
from MovieRecommender.components.data_preparation import DataPreparation

STAGE_NAME = "Data Preparation stage"


class DataPreparationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preparation_config = config.get_data_preparation_config()
        data_preparation = DataPreparation(config=data_preparation_config)
        data_preparation.load_data_from_db()
        data_preparation.preprocess_data()
        data_preparation.get_data_from_tmdb()
        data_preparation.prepare_final_data()
        data_preparation.save_final_data_in_db()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreparationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
