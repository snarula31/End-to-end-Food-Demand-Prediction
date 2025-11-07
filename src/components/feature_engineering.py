import os
import sys
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from exception import CustomException
from logger import logging

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self

    def derive_features(self,df:pd.DataFrame):
        try:
            # logging.info("Initiating feature engineering process")

            df['discount_amount'] = round(df['base_price'] - df['checkout_price'],4)
            
            df['discount_percentage'] = round((df['discount_amount'] / df['base_price']) * 100,4)
            
            df['discount_y_n'] = [1 if x > 0 else 0 for x in (df['base_price'] - df['checkout_price'])]
            
            df['weekly_base_price_change'] = round(df.groupby(['meal_id','center_id'])['base_price'].diff().fillna(0),4)
            
            df['weekly_checkout_price_change'] = round(df.groupby(['meal_id','center_id'])['checkout_price'].diff().fillna(0),4)
            
            df['4_week_avg_checkout_price'] = round(df.groupby(['meal_id', 'center_id'])['checkout_price'].transform(lambda x: x.rolling(window=4, min_periods=1).mean()),4)

            df['4_week_avg_base_price'] = round(df.groupby(['meal_id', 'center_id'])['base_price'].transform(lambda x: x.rolling(window=4, min_periods=1).mean()),4)

            df['week_of_year'] = df['week'].apply(lambda x: x % 52 if x % 52 != 0 else 52)
            
            df['quarter'] = df['week_of_year'].apply(lambda x: (x-1) // 13 + 1)
            
            df['month'] = df['week_of_year'].apply(lambda x: (x-1) // 4 + 1)

            #Creating lag feature
            df = df.sort_values(['meal_id','center_id','week'])
            for lag in [1,2,3,4,5,10]:
                df[f'lag_{lag}'] = df.groupby(['meal_id','center_id'])['num_orders'].shift(lag).fillna(0)

            rolling_window = df.groupby(['meal_id','center_id'])['num_orders'].shift(1).rolling(window=4, min_periods=1)

            df['rolling_4_week_mean'] = round(rolling_window.mean().fillna(0),4)
            df['rolling_std'] = round(rolling_window.std().reset_index(drop=True))  # Volatility
            df['rolling_min'] = round(rolling_window.min().reset_index(drop=True),4)
            df['rolling_max'] = round(rolling_window.max().reset_index(drop=True),4)
            df['rolling_median'] = round(rolling_window.median().reset_index(drop=True),4)
            

            avg_price_cat = df.groupby(['week', 'category'])['base_price'].apply(lambda x: x.mean()).reindex(df .set_index(['week', 'category']).index).values

            df['price_vs_category_avg'] = round(df['base_price'] - avg_price_cat,4)

            # Categorizing cities into four groups based on their order volume.
            city4={590:'C1', 526:'C2', 638:'C3'}
            df['city_cat']=df['city_code'].map(city4)
            df['city_cat']=df['city_cat'].fillna('C4')

            df['emailer_for_promotion'] = df['emailer_for_promotion'].astype('object')
            df['homepage_featured'] = df['homepage_featured'].astype('object')

            df['base_price_max'] = df.groupby('meal_id')['base_price'].transform('max')
            df['base_price_min'] = df.groupby('meal_id')['base_price'].transform('min')
            df['base_price_mean'] = round(df.groupby('meal_id')['base_price'].transform('mean'),4)
            
            df['meal_price_max'] = df.groupby('meal_id')['checkout_price'].transform('max')
            df['meal_price_min'] = df.groupby('meal_id')['checkout_price'].transform('min')
            df['meal_price_mean'] = round(df.groupby('meal_id')['checkout_price'].transform('mean'),4)
            
            df['center_cat_count'] = df.groupby(['category','center_id'])['num_orders'].transform('count')
            df['center_cat_price_rank'] = df.groupby(['category','center_id','meal_id'])['base_price'].rank(method='dense').astype('int64')
            df['center_cat_week_count'] = df.groupby(['category','center_id','week'])['num_orders'].transform('count')
            df['center_cuisine_count'] = df.groupby(['cuisine','center_id'])['num_orders'].transform('count')
            df['center_price_rank'] = df.groupby(['meal_id','center_id'])['base_price'].rank(method='dense').astype('int64')


            return df

        except Exception as e:
            logging.error(f"Error occurred during feature engineering: {e}")
            raise CustomException(e, sys) from e

