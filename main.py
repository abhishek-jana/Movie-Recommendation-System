from MovieRecommender.logging import logger
from MovieRecommender.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)

from MovieRecommender.pipeline.stage_02_data_preparation import (
    DataPreparationTrainingPipeline,
)

from MovieRecommender.pipeline.stage_03_content_based_model import (
    ContentBasedModelTrainingPipeline,
)

from MovieRecommender.pipeline.stage_04_collaborative_filtering import CollaborativeFilteringTrainingPipeline

# from MovieRecommender.pipeline.stage_04_evaluation import EvaluationPipeline

# STAGE_NAME = "Data Ingestion stage"
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     data_ingestion = DataIngestionTrainingPipeline()
#     data_ingestion.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "Data Preparation stage"
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     data_preparation = DataPreparationTrainingPipeline()
#     data_preparation.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = "Content Based Model Training"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    content_based_model = ContentBasedModelTrainingPipeline()
    content_based_model.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Collaborative Filtering Model Training"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    collaborative_filtering_model = CollaborativeFilteringTrainingPipeline()
    collaborative_filtering_model.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e



# STAGE_NAME = "Training stage"
# try:
#     logger.info(f"*******************")
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     model_trainer = ModelTrainingPipeline()
#     model_trainer.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e
