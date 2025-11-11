import os
import sys
from dataclasses import dataclass

from exception import CustomException
from logger import logging
from utils import save_object

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import category_encoders as ce


from components.feature_engineering import FeatureEngineering

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        try:
            numerical_columns = ['week','checkout_price', 'base_price', 'op_area', 'discount_amount',
                                'discount_percentage','weekly_base_price_change','weekly_checkout_price_change','week_of_year',
                                'lag_1', 'lag_2','lag_3', 'lag_4','lag_5','lag_10','lag_15', 
                                'price_vs_category_avg','expanding_base_price_mean', 'expanding_base_price_max',
                                'expanding_base_price_min', 'expanding_checkout_price_mean','expanding_checkout_price_max',
                                'expanding_checkout_price_min','center_price_rank','meal_price_rank','week_sin', 'week_cos',
                                'ewma_2_week_orders','ewma_4_week_orders', 'ewma_5_week_orders', 'ewma_10_week_orders','ewma_15_week_orders']
            
            ohe_categorical_columns = ['emailer_for_promotion', 'homepage_featured','center_type', 'category','cuisine']
            
            target_en_categorical_columns = ['center_id','region_code','city_code','meal_id']

            # Define the numerical and categorical transformers
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            ohe_categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine the transformers into a preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_columns),
                    ('cat', ohe_categorical_transformer, ohe_categorical_columns),
                    ('target_cat',ce.TargetEncoder(), target_en_categorical_columns)
                ],
                remainder='passthrough'
            )

            # logging.info("Data transformation pipeline created successfully.")
            return preprocessor

        except Exception as e:
            logging.error(f"Error occurred while creating data transformation pipeline: {e}")
            raise CustomException(e, sys) from e
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Initiating data transformation")
            logging.info("Reading training and testing data")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Successfully read training and test datasets")

            logging.info("Initiating feature engineering process")
            
            feature_engineering = FeatureEngineering()
            train_df = feature_engineering.derive_features(train_df)
            test_df = feature_engineering.derive_features(test_df)

            logging.info("Feature engineering completed")

            logging.info(f"Train df: {train_df.head()}")
            logging.info(f"Train df: {train_df.shape}")
            logging.info(f"Train df columns: {train_df.columns}")
            logging.info(f"Test df: {test_df.head()}")
            logging.info(f"Test df shape: {test_df.shape}")
            logging.info(f"Test df columns: {test_df.columns}")

            logging.info("Fetching preprocessor object")

            preprocessing_obj = self.get_data_transformer_object()

            # train_df.drop()
            # target_column_name = np.log1p("num_orders")
            numerical_columns = ['week', 'center_id', 'meal_id', 'checkout_price', 'base_price',
                                'region_code', 'op_area', 'discount_amount',
                                'discount_percentage', 'discount_y_n', 'weekly_base_price_change',
                                'weekly_checkout_price_change', 'week_of_year', 'quarter', 'month',
                                '4_week_avg_checkout_price', '4_week_avg_base_price', 'lag_1', 'lag_2',
                                'lag_3', 'lag_4', 'rolling_4_week_mean','rolling_std','rolling_min',
                                'rolling_max','rolling_median']
            
            categorical_columns = ['emailer_for_promotion', 'homepage_featured', 'center_type', 'category',
                                   'cuisine', 'city_cat']

            input_feature_train_df = train_df.drop(columns=['id','num_orders'],axis=1)#x_train
            target_feature_train_df = np.log1p(train_df['num_orders'])#y_train

            input_feature_test_df = test_df.drop(columns=['id','num_orders'],axis=1)#x_test
            target_feature_test_df = np.log1p(test_df['num_orders'])#y_test

            logging.info(f"Input feature train df: {input_feature_train_df.head()}")
            logging.info(f"Target feature train df: {target_feature_train_df.head()}")
            logging.info(f"Input feature test df: {input_feature_test_df.head()}")
            logging.info(f"Target feature test df: {target_feature_test_df.head()}")

            logging.info("Applying preprocessing object on training and test dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df,target_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Applied preprocessing object on training and testing datasets successfully")
            logging.info(f'Train array shape: {train_arr.shape}')
            logging.info(f'Train array: {train_arr}')
            logging.info(f'Test array shape: {test_arr.shape}')
            logging.info(f'Test array: {test_arr}')

            logging.info(f"Saving preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            logging.error(f"Error occurred while initiating data transformation: {e}")
            raise CustomException(e, sys) from e