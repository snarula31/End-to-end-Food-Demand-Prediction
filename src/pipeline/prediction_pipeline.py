import os
import sys
import pandas as pd
import numpy as np

from src.components.feature_engineering import FeatureEngineering

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.merged = pd.read_csv('artifacts/merged.csv')
        self.preprocessor = load_object('artifacts/preprocessor1.pkl')
        self.model_xgb = load_object('artifacts/XGBRegressor_model.pkl')
        self.model_lgbm = load_object('artifacts/LGBM Regressor_model.pkl')
        self.model_cat = load_object('artifacts/CatBoosting Regressor_model.pkl')

        self.reference_df = self.merged[self.merged['week'].isin(range(120,146))]
        self.reference_df = self.reference_df.sort_values(by=['center_id', 'meal_id', 'week'])

    def predict(self, input_data_frame):
        try:
            # 1. RETRIEVE CONTEXT
            # Find history for this specific center/meal
            center_id = input_data_frame['center_id'].iloc[0]
            meal_id = input_data_frame['meal_id'].iloc[0]
            
            history = self.reference_df[
                (self.reference_df['center_id'] == center_id) & 
                (self.reference_df['meal_id'] == meal_id)
            ]

            # 2. APPEND AND ENGINEER
            # Combine history + new input to calculate rolling features
            combined = pd.concat([history, input_data_frame],axis=0, ignore_index=True)
            
            # Run your Feature Engineering logic here
            fe = FeatureEngineering()
            combined_featured = fe.derive_features(combined)
            
            # Select ONLY the last row (the new input) for prediction
            input_row_featured = combined_featured.iloc[[-1]].drop(columns=['id', 'num_orders'])

            # 3. TRANSFORM
            # Use the pre-fitted preprocessor (DO NOT FIT again)
            data_scaled = self.preprocessor.transform(input_row_featured)

            # 4. PREDICT & ENSEMBLE
            pred_xgb = self.model_xgb.predict(data_scaled)
            pred_lgbm = self.model_lgbm.predict(data_scaled)
            pred_cat = self.model_cat.predict(data_scaled)

            final_pred_log = (pred_xgb + pred_lgbm + pred_cat) / 3

            final_pred = np.expm1(final_pred_log)
            
            return final_pred[0]
            
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    # This class maps HTML inputs/JSON to a DataFrame
    def __init__(self, week, center_id, meal_id, checkout_price, base_price, 
                 emailer_for_promotion, homepage_featured, city_code, region_code, 
                 center_type, op_area, category, cuisine):
        self.week = week
        self.center_id = center_id
        self.meal_id = meal_id
        self.checkout_price = checkout_price
        self.base_price = base_price
        self.emailer_for_promotion = emailer_for_promotion
        self.homepage_featured = homepage_featured
        self.city_code = city_code
        self.region_code = region_code
        self.center_type = center_type
        self.op_area = op_area
        self.category = category
        self.cuisine = cuisine


    def get_data_as_data_frame(self):
        try:

            self.data_dict = {
                "week": [self.week],
                "center_id": [self.center_id],
                "meal_id": [self.meal_id],
                "checkout_price": [self.checkout_price],
                "base_price": [self.base_price],
                "emailer_for_promotion": [self.emailer_for_promotion],
                "homepage_featured": [self.homepage_featured],
                "city_code": [self.city_code],
                "region_code": [self.region_code],
                "center_type": [self.center_type],
                "op_area": [self.op_area],
                "category": [self.category],
                "cuisine": [self.cuisine],
                "num_orders": [0] # Placeholder for feature engineering to work
            }
            return pd.DataFrame(self.data_dict)
        except Exception as e:
            logging.error(f"Error occurred while creating DataFrame from custom data: {e}")
            raise CustomException(e, sys)
