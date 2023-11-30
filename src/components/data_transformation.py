import os,sys
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
from src.constant import *
from src.utils.main_utils import *
import logging


@dataclass
class DataTransformationConfig:
    artifact_folder=os.path.join(artifact_folder)
    transformed_train_file_path=os.path.join(artifact_folder,"train.csv")
    transformed_test_file_path=os.path.join(artifact_folder,"test.csv")
    transformed_obj_file_path=os.path.join(artifact_folder,"preprocessor.pkl")


class DataTransformation:
    def __init__(self,feature_store_file_path):
        self.feature_store_file_path=feature_store_file_path
        self.data_transformation_config=DataTransformationConfig()
        self.utils=MainUtils()

    @staticmethod
    def get_data(feature_store_file_path):
        try:
            logging.info("Getting data")
            df=pd.read_csv(feature_store_file_path)
            logging.info("Data loaded successfully")
            return df
        except Exception as e:
            logging.error("Error occured while getting data")
            raise CustomException(e,sys)
    
    def get_data_transformation_object(self):
        try:
            scaler_step=('scaler',StandardScaler())
            preprocessor=Pipeline(
                steps=[scaler_step]
            )
            return preprocessor
        except Exception as e:
            logging.error("Error occured while getting data transformation object")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self):
        try:
            logging.info("Initiating data transformation")
            df=self.get_data(self.feature_store_file_path)

            X=df.drop(columns=TARGET_COLUMN,axis=1)
            y=df[TARGET_COLUMN]

            X_res,y_res=SMOTE().fit_resample(X,y)
            X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,test_size=0.2,random_state=42)
            preprocessor=self.get_data_transformation_object()

            X_train_transformed=preprocessor.fit_transform(X_train)
            X_test_transformed=preprocessor.transform(X_test)

            preprocessor_path=self.data_transformation_config.transformed_obj_file_path
            os.makedirs(os.path.dirname(preprocessor_path),exist_ok=True)
            self.utils.save_object(filepath=preprocessor_path,obj=preprocessor)

            train_arr=np.c_[X_train_transformed,np.array(y_train)]
            test_arr=np.c_[X_test_transformed,np.array(y_test)]

            logging.info("Saving transformed train data")
            return (train_arr,test_arr,preprocessor_path)
        except Exception as e:
            logging.info("Error occured while initiating data transformation")
            raise CustomException(e,sys)