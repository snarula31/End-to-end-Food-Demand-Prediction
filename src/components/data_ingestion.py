import os
import sys
import numpy as np
from exception import CustomException
from logger import logging
import pandas as pd
from dataclasses import dataclass 
from sklearn.model_selection import train_test_split

from components.data_transformation import DataTransformationConfig
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainerConfig
from components.model_trainer import ModelTrainer
from components.lstm_data_transformation import LSTMConfig
from components.lstm_data_transformation import LSTMDataProcessor
from components.lstm_model_trainer import LSTMModelTrainerConfig
from components.lstm_model_trainer import LSTMModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    merged_data_path: str = os.path.join('artifacts', 'merged.csv')
    final_test_data_path: str = os.path.join('artifacts', 'final_test_set.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

import os
import sys
import numpy as np
from exception import CustomException
from logger import logging
import pandas as pd
from dataclasses import dataclass 
from sklearn.model_selection import train_test_split

from components.data_transformation import DataTransformationConfig
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainerConfig
from components.model_trainer import ModelTrainer
from components.lstm_data_transformation import LSTMConfig
from components.lstm_data_transformation import LSTMDataProcessor
from components.lstm_model_trainer import LSTMModelTrainerConfig
from components.lstm_model_trainer import LSTMModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    merged_data_path: str = os.path.join('artifacts', 'merged.csv')
    final_test_data_path: str = os.path.join('artifacts', 'final_test_set.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter data ingestion stage")

        try:

            weekly_demand = pd.read_csv('notebook/data/train.csv')
            center_info = pd.read_csv('notebook/data/fulfilment_center_info.csv') 
            meal_info = pd.read_csv('notebook/data/meal_info.csv')
            # test = pd.read_csv('notebook/data/test.csv')
            logging.info("Read the dataset as dataframe")

            # Adding a placeholder column for predictions in the test dataset
            # test['num_orders'] = np.nan
            
            logging.info("Merging all the datasets into a single dataset")
            # merging all the data sets into single datset for analysis
            # data = pd.concat([weekly_demand, test], axis=0)
            data = weekly_demand.copy()
            data = data.merge(center_info, on='center_id', how='left')
            data = data.merge(meal_info, on='meal_id', how='left')

            data['num_orders'] = pd.to_numeric(data['num_orders'], errors='coerce')

            logging.info(f'merged data info:{data.info()}')

            # logging.info(f"test data: {test.head(5)}")
            # logging.info(f"merged data: {data.head(5)}")
            # logging.info(f"merged data shape: {data.shape}")
            # logging.info(f"merged data columns: {data.columns}")

            os.makedirs(os.path.dirname(self.ingestion_config.merged_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.merged_data_path, index=False)

            logging.info("Train test split initiated")
            train_set = data[data['week'].isin(range(1,136))]
            test_set = data[data['week'].isin(range(136,146))]
            # final_test_set = data[data['week'].isin(range(146,155))]

            logging.info(f"train set: {train_set.head(5)}")
            logging.info(f"Train set shape: {train_set.shape}")
            logging.info(f"test set: {test_set.head(5)}")
            logging.info(f"Test set shape: {test_set.shape}")
            # logging.info(f"final test set: {final_test_set.head(5)}")
            # logging.info(f"final test set shape: {final_test_set.shape}")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            # final_test_set.to_csv(self.ingestion_config.final_test_data_path, index=False)

            logging.info("Data Ingestion completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                # self.ingestion_config.final_test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)

    # model_trainer = ModelTrainer()
    # print(model_trainer.initiate_model_trainer(train_arr,test_arr))

    # lstm_trainer = LSTMModelTrainer()
    # print(lstm_trainer.initiate_training(train_data))