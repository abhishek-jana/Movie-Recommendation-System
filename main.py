from MovieRecommender.logging import logger
from MovieRecommender.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)

from MovieRecommender.pipeline.stage_02_data_preparation import (
    DataPreparationTrainingPipeline,
)

# from MovieRecommender.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
# from MovieRecommender.pipeline.stage_03_training import ModelTrainingPipeline
# from MovieRecommender.pipeline.stage_04_evaluation import EvaluationPipeline

STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Preparation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_preparation = DataPreparationTrainingPipeline()
    data_preparation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
