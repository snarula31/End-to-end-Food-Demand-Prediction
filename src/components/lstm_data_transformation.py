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
                                    # 'center_id','meal_id'
                                    ]

    def get_LSTM_data_transformer_object(self):
        try:
            
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # categorical_transformer = Pipeline(steps=[
            #     ('imputer', SimpleImputer(strategy='most_frequent',fill_value='missing')),
            #     ('label', LabelEncoder())
            # ])
            
            LSTM_preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical', numerical_transformer, self.numerical_columns),
                    # ('categorical', categorical_transformer, self.categorical_columns),
                ],
                remainder='passthrough'
                )

            return LSTM_preprocessor
        
        except Exception as e:
            logging.error(f"Error occurred in LSTM data handler object creation: {e}")
            raise CustomException(e, sys) from e

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
            group_target = group[target_col].values
            
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

            train_df = train_df.sort_values(['meal_id','center_id','week'])
            test_df = test_df.sort_values(['meal_id','center_id','week'])

            target_col_name = 'num_orders'
            train_df[target_col_name] = np.log1p(train_df[target_col_name])
            test_df[target_col_name] = np.log1p(test_df[target_col_name])

            # logging.info("Fetching preprocessor object")            
            # preprocessing_obj = self.get_LSTM_data_transformer_object()

            # X_train_raw = train_df.drop(columns=['id','num_orders'],axis=1)
            # y_train = train_df[target_col_name]
            # X_test_raw = test_df.drop(columns=['id','num_orders'],axis=1)
            # y_test = test_df[target_col_name]

            # logging.info("Applying preprocessing object on training and test dataframes")

            # X_train_transformed = preprocessing_obj.fit_transform(X_train_raw)
            # X_test_transformed = preprocessing_obj.transform(X_test_raw)

            # all_columns = self.numerical_columns + self.categorical_columns

            # train_df_processed = pd.DataFrame(X_train_transformed, columns=all_columns)
            # test_df_processed = pd.DataFrame(X_test_transformed, columns=all_columns)

            # train_df_processed[target_col_name] = train_df[target_col_name].values
            # test_df_processed[target_col_name] = test_df[target_col_name].values

            # logging.info("Creating sequences for Training Data...")
            # X_train_dyn, X_train_stat, y_train = self.create_sequences(train_df_processed, target_col_name)
            
            # logging.info("Creating sequences for Testing Data...")
            # X_test_dyn, X_test_stat, y_test = self.create_sequences(test_df_processed, target_col_name)

            # logging.info(f"Final Train Shapes - Dynamic: {X_train_dyn.shape}, Static: {X_train_stat.shape}, Y: {y_train.shape}")

            # # 8. Structure output for Keras Multi-Input Model
            # # We split the static array into a list of individual arrays (one per categorical feature)
            # # This allows separate Embedding layers for each ID type
            
            # def split_static_inputs(X_stat):
            #     # X_stat is (N, 7). We want a list of 7 arrays, each (N,)
            #     return [X_stat[:, i] for i in range(X_stat.shape[1])]

            # train_inputs = [X_train_dyn] + split_static_inputs(X_train_stat)
            # test_inputs = [X_test_dyn] + split_static_inputs(X_test_stat)

            # return (
            #     train_inputs,
            #     y_train,
            #     test_inputs,
            #     y_test,
            #     self.lstm_data_transformation_config.lstm_preprocessor_obj_file_path
            # )



        except Exception as e:
            logging.error(f"Error occurred during LSTM data transformation: {e}")
            raise CustomException(e, sys) from e
