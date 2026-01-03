# from ChestCancerClassifier import logger
import sys
from pathlib import Path
import dagshub

dagshub.init(
    repo_owner="am-alibee",
    repo_name="Chicken-Disease-Classifier-With-Mlflow-and-DVC",
    mlflow=True
)

# Add src folder to sys.path so Python can find ChestCancerClassifier
sys.path.append(str(Path(__file__).parent / "src"))

from ChestCancerClassifier import logger
from ChestCancerClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from ChestCancerClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from ChestCancerClassifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from ChestCancerClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline

STAGE_NAME = "Data Ingestion Stage"


try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare base model"

try:
    logger.info(f"****************")
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Training"

if __name__ == "__main__":
    try:
        logger.info(f"****************")
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<")
    except Exception as e:
        logger.exception(e)
        raise e
    

STAGE_NAME = "Evaluation"

if __name__ == "__main__":
    try:
        logger.info(f"****************")
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<")
    except Exception as e:
        logger.exception(e)
        raise e