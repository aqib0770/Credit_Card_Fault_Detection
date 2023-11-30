import os,sys
import pandas as pd
from src.logger import logging

from src.exception import CustomException
from flask import request
from src.constant import *
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname:str="predictions"
    prediction_file_name:str="prediction.csv"
    model_file_path:str=os.path.join(artifact_folder,"model.pkl")
    preprocessor_file_path:str=os.path.join(artifact_folder,"preprocessor.pkl")
    prediction_output_path:str=os.path.join(prediction_output_dirname,prediction_file_name)


class PredictPipeline:
    def __init__(self,request:request):
        self.request=request
        self.prediction_pipeline_config=PredictionPipelineConfig()
        self.utils=MainUtils()

    def save_prediction(self):
        try:
            pred_file_input_dir="prediction_artifacts"
            os.makedirs(pred_file_input_dir,exist_ok=True)
            input_csv_file=self.request.files["file"]
            pred_file_path=os.path.join(pred_file_input_dir,input_csv_file.filename)
            input_csv_file.save(pred_file_path)
            return pred_file_path
        except Exception as e:
            logging.error("Error occured while saving prediction")
            raise CustomException(e,sys)
        
    def predict(self, features):
            try:
                model_path = self.prediction_pipeline_config.model_file_path
                preprocessor_path = self.prediction_pipeline_config.preprocessor_file_path


                model = self.utils.load_object(file_path=model_path)
                preprocessor = self.utils.load_object(file_path= preprocessor_path)

                transformed_features = preprocessor.transform(features)


                preds = model.predict(transformed_features)

                return preds

            except Exception as e:
                raise CustomException(e, sys)
            
    def get_predicted_dataframe(self, input_dataframe_path:pd.DataFrame):
        try:

            prediction_column_name : str = TARGET_COLUMN
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)
            
            input_dataframe =  input_dataframe.drop(columns="Unnamed: 0") if "Unnamed: 0" in input_dataframe.columns else input_dataframe

            predictions = self.predict(input_dataframe)
            input_dataframe[prediction_column_name] = [pred for pred in predictions]


            
            os.makedirs( self.prediction_pipeline_config.prediction_output_dirname, exist_ok= True)
            input_dataframe.to_csv(self.prediction_pipeline_config.prediction_output_path, index= False)
            logging.info("predictions completed. ")
        except Exception as e:
            raise CustomException(e, sys)
        
    
        
    def run_pipeline(self):
        try:
            input_csv_path = self.save_prediction()
            self.get_predicted_dataframe(input_csv_path)

            return self.prediction_pipeline_config


        except Exception as e:
            raise CustomException(e,sys)