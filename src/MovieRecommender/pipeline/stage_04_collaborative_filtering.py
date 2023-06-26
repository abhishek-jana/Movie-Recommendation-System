from MovieRecommender.config.configuration import ConfigurationManager
from MovieRecommender.components.collaborative_filtering import CollaborativeFiltering
from MovieRecommender.logging import logger

STAGE_NAME = "Collaborative Filtering Model Training"


class CollaborativeFilteringTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        collaborative_filtering_config = config.get_collaborative_filtering_model_config()
        collaborative_filtering = CollaborativeFiltering(
                                    config=collaborative_filtering_config)
        collaborative_filtering.load_data_from_db()
        collaborative_filtering.process_data()
        collaborative_filtering.train_NearestNeighbors_model()  
        collaborative_filtering.train_SVD_model() 

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = CollaborativeFilteringTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
 