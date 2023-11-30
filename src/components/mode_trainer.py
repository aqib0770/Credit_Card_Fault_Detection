import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import *

@dataclass
class ModelTrainerConfig:
    artifact_folder=os.path.join(artifact_folder)
    trained_model_file_path=os.path.join(artifact_folder,"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        self.utils=MainUtils()
        self.model = RandomForestClassifier(criterion='gini', max_depth=9, max_features='log2', n_estimators=200)
    
    def initiate_model_training(self,X,y):
        try:
            X_train, y_train, X_test, y_test = (
                X[:,:-1], X[:,-1],
                y[:,:-1], y[:,-1]
            )
            logging.info("Training model")
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Model accuracy {accuracy}")
            logging.info("Saving model at path: {0}".format(self.model_trainer_config.trained_model_file_path))
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            self.utils.save_object(filepath=self.model_trainer_config.trained_model_file_path, obj=self.model)
            logging.info("Model saved successfully")
            return self.model_trainer_config.trained_model_file_path
        except Exception as e:
            logging.info("Error occured while initiating model training")
            raise CustomException(e, sys)