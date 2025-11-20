import os
import sys
import pandas as pd
import numpy as np
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
            
            # df['discount_y_n'] = [1 if x > 0 else 0 for x in (df['base_price'] - df['checkout_price'])]
            
            df['weekly_base_price_change'] = round(df.groupby(['meal_id','center_id'])['base_price'].diff().fillna(0),4)
            
            df['weekly_checkout_price_change'] = round(df.groupby(['meal_id','center_id'])['checkout_price'].diff().fillna(0),4)
            
            # df['4_week_avg_checkout_price'] = round(df.groupby(['meal_id', 'center_id'])['checkout_price'].transform(lambda x: x.rolling(window=4, min_periods=1).mean()),4)

            # df['4_week_avg_base_price'] = round(df.groupby(['meal_id', 'center_id'])['base_price'].transform(lambda x: x.rolling(window=4, min_periods=1).mean()),4)

            df['week_of_year'] = df['week'].apply(lambda x: x % 52 if x % 52 != 0 else 52)
            
            # df['quarter'] = df['week_of_year'].apply(lambda x: (x-1) // 13 + 1)
            
            # df['month'] = df['week_of_year'].apply(lambda x: (x-1) // 4 + 1)

            #Creating lag feature
            df = df.sort_values(['meal_id','center_id','week'])
            for lag in [1,2,3,4,5,10,15]:
                df[f'lag_{lag}'] = df.groupby(['meal_id','center_id'])['num_orders'].shift(lag).fillna(0)

            # rolling_window = df.groupby(['meal_id','center_id'])['num_orders'].shift(1).rolling(window=4, min_periods=1)

            # df['rolling_4_week_mean'] = round(rolling_window.mean().fillna(0),4)
            # df['rolling_std'] = round(rolling_window.std().reset_index(drop=True))  # Volatility
            # df['rolling_min'] = round(rolling_window.min().reset_index(drop=True),4)
            # df['rolling_max'] = round(rolling_window.max().reset_index(drop=True),4)
            # df['rolling_median'] = round(rolling_window.median().reset_index(drop=True),4)
            

            avg_price_cat = df.groupby(['week', 'category'])['base_price'].apply(lambda x: x.mean()).reindex(df .set_index(['week', 'category']).index).values

            df['price_vs_category_avg'] = round(df['base_price'] - avg_price_cat,4)

            # Categorizing cities into four groups based on their order volume.
            # city4={590:'C1', 526:'C2', 638:'C3'}
            # df['city_cat']=df['city_code'].map(city4)
            # df['city_cat']=df['city_cat'].fillna('C4')

            df['emailer_for_promotion'] = df['emailer_for_promotion'].astype('object')
            df['homepage_featured'] = df['homepage_featured'].astype('object')
            df['city_code'] = df['city_code'].astype('object')
            df['region_code'] = df['region_code'].astype('object')
            df['center_id'] = df['center_id'].astype('object')
            df['meal_id'] = df['meal_id'].astype('object')

            df = df.sort_values(['meal_id','center_id','week'])
            
            base_price_expanding_window = df.groupby(['meal_id','center_id'])['base_price'].expanding(min_periods=1)
            df['expanding_base_price_mean'] = round(base_price_expanding_window.mean().reset_index(level=[0,1],drop=True),4)
            df['expanding_base_price_max'] = base_price_expanding_window.max().reset_index(drop=True)
            df['expanding_base_price_min'] = base_price_expanding_window.min().reset_index(drop=True)
            
            checkout_price_expanding_window = df.groupby(['meal_id','center_id'])['checkout_price'].expanding(min_periods=1)
            df['expanding_checkout_price_mean'] = round(checkout_price_expanding_window.mean().reset_index(level=[0,1],drop=True),4)
            df['expanding_checkout_price_max'] = checkout_price_expanding_window.max().reset_index(drop=True)
            df['expanding_checkout_price_min'] = checkout_price_expanding_window.min().reset_index(drop=True)


            # df['center_cat_count'] = df.groupby(['category','center_id'])['num_orders'].transform('count')
            
            # df['center_cat_price_rank'] = df.groupby(['category','center_id','meal_id'])['base_price'].rank(method='dense').astype('int64')
            # 
            # df['center_cat_week_count'] = df.groupby(['category','center_id','week'])['num_orders'].transform('count')
            
            # df['center_cuisine_count'] = df.groupby(['cuisine','center_id'])['num_orders'].transform('count')
            
            df['center_price_rank'] = df.groupby(['meal_id','center_id'])['base_price'].rank(method='dense').astype('int64')
            
            # df['center_week_price_rank'] = df.groupby(['center_id','week','meal_id'])['base_price']. rank(method='dense').astype('int64')
            
            # df['center_week_order_count'] = df.groupby(['center_id','week'])['num_orders']. transform('count')
            
            # df['city_meal_week_count'] = df.groupby(['city_code','week'])['meal_id'].transform('count')
            
            # df['meal_count'] = df.groupby('meal_id')['num_orders'].transform('count')
            
            # df['meal_city_price_rank'] = df.groupby(['meal_id','city_code'])['base_price'].rank(method='dense').astype('int64')

            df['meal_price_rank'] = df.groupby('meal_id')['base_price'].rank(method='dense').astype('int64')

            # df['meal_region_price_rank'] = df.groupby(['meal_id','region_code'])['base_price'].rank(method='dense').astype('int64')

            # df['meal_week_count'] = df.groupby(['meal_id','week'])['num_orders'].transform('count')

            # df['meal_week_price_rank'] = df.groupby(['meal_id','week'])['base_price'].rank(method='dense').astype('int64')

            # df['region_meal_count'] = df.groupby(['region_code','meal_id'])['num_orders'].transform('count')

            # Cyclical "Week of Year" Features (helps models understand seasonality)
            df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
            df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

            # Creating Exponential Moving Average (EMA) Features
            df = df.sort_values(['meal_id','center_id','week'])
            grouped = df.groupby(['meal_id','center_id'])
            
            spans = [1,2,4,5,10,15]

            for span in spans:
                ewma_orders = grouped['num_orders'].shift(1).ewm(span=span, adjust=True,min_periods=1).mean()
                df[f'ewma_{span}_week_orders'] = round(ewma_orders.fillna(0),4)

            
            return df

        except Exception as e:
            logging.error(f"Error occurred during feature engineering: {e}")
            raise CustomException(e, sys) from e

    def derive_features_lstm(self,df:pd.DataFrame):
        try:
            
            df['city_code'] = df['city_code'].astype('object')
            df['region_code'] = df['region_code'].astype('object')
            df['center_id'] = df['center_id'].astype('object')
            df['meal_id'] = df['meal_id'].astype('object')

            df['discount_amount'] = round(df['base_price'] - df['checkout_price'],4)
                
            df['discount_percentage'] = round((df['discount_amount'] / df['base_price']) * 100,4)

            avg_price_cat = df.groupby(['week', 'category'])['base_price'].apply(lambda x: x.mean()).reindex(df .set_index(['week', 'category']).index).values

            df['price_vs_category_avg'] = round(df['base_price'] - avg_price_cat,4)

            df['week_of_year'] = df['week'].apply(lambda x: x % 52 if x % 52 != 0 else 52)
            df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
            df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
            return df
        except Exception as e:
            logging.error(f"Error occurred during feature engineering for LSTM: {e}")
            raise CustomException(e, sys) from e