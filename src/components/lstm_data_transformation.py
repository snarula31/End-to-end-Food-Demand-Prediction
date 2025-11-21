import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import tensorflow as tf
import pickle
from dataclasses import dataclass

from logger import logging
from exception import CustomException
from components.feature_engineering import FeatureEngineering
from utils import save_object

@dataclass
class LSTMDataTransformationConfig:
    lstm_preprocessor_obj_file_path: str = os.path.join('artifacts', 'lstm_preprocessor.pkl')


class LSTMDataTransformation:
    def __init__(self):
        self.lstm_data_transformation_config = LSTMDataTransformationConfig()
        self.numerical_columns = ['week','base_price','checkout_price','discount_amount','discount_percentage',
                                  'week_of_year','week_sin','week_cos','price_vs_category_avg',
                                #   'center_price_rank','meal_price_rank'
                                  ]
        
        self.categorical_columns = ['category','cuisine','center_type','region_code','city_code',
                                    'center_id','meal_id'
                                    ]

    def create_sequences(self, df: pd.DataFrame, target_col,window_size: int):
        """
        Efficiently converts DataFrame into 3D sequences (Samples, TimeSteps, Features)
        """
        # window_size = self.lstm_data_transformation_config.window_size
        
        X_dynamic = [] 
        X_static = []  
        y = []         
        
        dyn_cols = self.numerical_columns
        stat_cols = self.categorical_columns
        
        grouped = df.groupby(['center_id', 'meal_id'])
        
        logging.info("Generating sequences (Sliding Window)...")
        
        for _, group in grouped:
            if len(group) <= window_size:
                continue
            
            group_dyn = group[dyn_cols].values
            group_stat = group[stat_cols].values
            group_target = np.log1p(group[target_col].values)
            # group_target = group[target_col].values
            
            for i in range(window_size, len(group)):
                X_dynamic.append(group_dyn[i-window_size:i])
                
                X_static.append(group_stat[i]) 
                
                y.append(group_target[i])
                
        return np.array(X_dynamic), np.array(X_static), np.array(y)
    
    def initiate_LSTM_data_transformation(self,train_path,test_path):
        try:
            logging.info('Initiating LSTM data Transformation')
            logging.info('Reading training and testing data')

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Successfully read training and test datasets')

            logging.info('Initiating feature engineering for LSTM model')

            feature_engineering = FeatureEngineering()
            train_df = feature_engineering.derive_features_lstm(train_df)
            test_df = feature_engineering.derive_features_lstm(test_df)

            logging.info('Feature engineering for LSTM model completed')

            logging.info(f"Train df: {train_df.head()}")
            logging.info(f"Train df: {train_df.shape}")
            logging.info(f"Train df columns: {train_df.columns}")
            logging.info(f"Test df: {test_df.head()}")
            logging.info(f"Test df shape: {test_df.shape}")
            logging.info(f"Test df columns: {test_df.columns}")

            train_df = train_df.sort_values(['meal_id','center_id','week']).reset_index(drop=True)
            test_df = test_df.sort_values(['meal_id','center_id','week']).reset_index(drop=True)   
            
            logging.info("Scaling numerical features for LSTM model")
            
            scaler = StandardScaler()
            train_df[self.numerical_columns] = scaler.fit_transform(train_df[self.numerical_columns])
            test_df[self.numerical_columns] = scaler.transform(test_df[self.numerical_columns])
            
            logging.info("Scaled numerical features for LSTM model successfully")

            logging.info("Encoding categorical columns for LSTM model")

            encoder = LabelEncoder()
            for col in self.categorical_columns:

                all_values = pd.concat([train_df[col], test_df[col]], axis=0)
                encoder.fit(all_values)

                train_df[col] = encoder.transform(train_df[col])
                test_df[col] = encoder.transform(test_df[col])

            logging.info("Encoded categorical features for LSTM model successfully")

            logging.info("Creating sequences for LSTM model")

            X_train_dyn,X_train_stat,y_train = self.create_sequences(df=train_df,target_col='num_orders',window_size=10)

            last_train_records = train_df.groupby(['center_id', 'meal_id']).tail(10)
            
        
            test_df_with_history = pd.concat([last_train_records, test_df]).sort_values(['meal_id','center_id','week'])
            
            X_test_dyn, X_test_stat, y_test = self.create_sequences(test_df_with_history, target_col='num_orders',window_size=10)

            def split_static(arr):
                return [arr[:, i] for i in range(arr.shape[1])]

            train_inputs = [X_train_dyn] + split_static(X_train_stat)
            test_inputs = [X_test_dyn] + split_static(X_test_stat)

            logging.info("Created sequences for LSTM model successfully")
            logging.info(f'Train inputs shapes: {[inp.shape for inp in train_inputs]}')
            logging.info(f'train inputs: {train_inputs}')
            logging.info(f'Test inputs shapes: {[inp.shape for inp in test_inputs]}')
            logging.info(f'test inputs: {test_inputs}')

            return train_inputs, y_train, test_inputs, y_test


        except Exception as e:
            logging.error(f"Error occurred during LSTM data transformation: {e}")
            raise CustomException(e, sys) from e
