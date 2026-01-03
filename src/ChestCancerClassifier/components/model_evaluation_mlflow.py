import os
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import dagshub
from ChestCancerClassifier.entity.config_entity import EvaluationConfig
from ChestCancerClassifier.utils.common import read_yaml, create_directories, save_json



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config=config
    
    def _valid_generator(self):
        
        datagenerator_kwargs = dict(
            rescale=1./255,
            # validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            # directory=self.config.training_data,
            directory=os.path.join(self.config.training_data, "valid"),
            # subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    
    def save_score(self): 
        scores={"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
 

    def log_into_mlflow(self, registered_model_name="MobileNetV2"):

        mlflow.set_tracking_uri(self.config.mlflow_url)        
        mlflow.set_registry_uri(self.config.mlflow_url)

        tracking_url = mlflow.get_tracking_uri()
        print("MLFLOW tracking unit: ", tracking_url)


        # check if backend is local store or remote
        tracking_url_type_score = urlparse(mlflow.get_tracking_uri()).scheme

        # mlflow.set_experiment("Chest-Cancer-Mlops")

        with mlflow.start_run():
            # log parameters
            mlflow.log_params(self.config.all_params)
            # log metrics
            mlflow.log_metrics({
                "loss": self.score[0], 
                "accuracy": self.score[1]
            })
            # log & register metrics if backend supports it
            if tracking_url_type_score != "file":
                mlflow.keras.log_model(
                    self.model,
                    "model",
                    registered_model_name=registered_model_name
                )
                print(f"Model registered as '{registered_model_name}'")
            else:
                mlflow.keras.log_model(self.model, "model")
                print(f"Model logged to artifacts (manual registration required!)")
