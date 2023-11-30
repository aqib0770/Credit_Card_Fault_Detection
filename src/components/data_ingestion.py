from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import os,sys
import pandas as pd
from src.utils.main_utils import MainUtils
from src.constant import *
from sklearn.model_selection import train_test_split
from pymongo.mongo_client import MongoClient
import numpy as np

@dataclass
class DataIngestionConfig:
    artifact_folder=os.path.join(artifact_folder)

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
        self.utils=MainUtils()

    def export_collection_as_dataframe(self,collection_name,db_name):
        try:
            mongo_client=MongoClient(MONGO_DB_URL)
            collection=mongo_client[db_name][collection_name]
            df=pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df.drop(columns=["_id"],inplace=True,axis=1)

            return df
        except Exception as e:
            logging.error("Error occured while exporting collection as dataframe")
            raise CustomException(e,sys)
        
    def export_data_into_feature(self)->pd.DataFrame:
        try:
            logging.info("Exporting data into feature")
            raw_file_path=self.data_ingestion_config.artifact_folder
            os.makedirs(raw_file_path,exist_ok=True)

            credit_card_data=self.export_collection_as_dataframe(
                collection_name=MONGO_COLLECTION_NAME,
                db_name=MONGO_DATABASE_NAME
            )

            logging.info("Data exported successfully")
            feature_store_file_path=os.path.join(raw_file_path,"credit_card_data.csv")#Path to store exported data file
            logging.info(f"Saving feature store file {feature_store_file_path}")
            credit_card_data.to_csv(feature_store_file_path,index=False)
            return feature_store_file_path
        except Exception as e:
            logging.error("Error occured while exporting data into feature")
            raise CustomException(e,sys)
        

    def initiate_data_ingestion(self):
        try:
            logging.info("Initiating data ingestion")
            feature_store_file_path=self.export_data_into_feature()
            logging.info("Data ingestion completed successfully")
            return feature_store_file_path
        except Exception as e:
            logging.error("Error occured while initiating data ingestion")
            raise CustomException(e,sys)