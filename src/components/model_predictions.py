import os
import sys
import numpy as np
import pandas as pd

from src.components.feature_engineering import FeatureEngineering
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

# @dataclass
# class ModelPredictionsConfig:
    
#     pass

class ModelPredictions:
    def __init__(self):
        pass

    def initiate_model_predictions(self,test_df):

        logging.info("loadinng test_df")

        # test_df = pd.read_csv('artifacts/test.csv')
        # final_test_df = pd.read_csv('notebook/data/test.csv')
        # # final
        # logging.info(final_test_df.head(5))
        # logging.info(final_test_df.shape)

        logging.info('fetching preprocessor object')

        preprocessor = load_object(
            file_path='artifacts/preprocessor.pkl'
        )

        test_array = preprocessor.transform(test_df)

        logging.info("Enter model predictions stage")
        logging.info("Loading test data for predictions")

        X_test = test_array[:,:-1]
        y_test = test_array[:,-1]

        logging.info(f"X_test shape: {X_test.shape}")
        logging.info("Test data loading completed")
        logging.info("Loading trained models for predictions")

        xgb_model = load_object(
            file_path='artifacts\XGBRegressor_model.pkl'
        )

        lightgbm_model = load_object(
            file_path='artifacts\LGBM Regressor_model.pkl'
        )

        catboost_model = load_object(
            file_path='artifacts\CatBoosting Regressor_model.pkl'
        )

        logging.info("Trained models loading completed")

        y_test_normal = np.expm1(y_test)

        logging.info(f"---------- Making predictions on test data using XGB regressor --------")
        log_predictions_xgb = xgb_model.predict(X_test)
        actual_predictions_xgb = np.expm1(log_predictions_xgb)

        logging.info(f"---------- Making predictions on test data using LightGBM regressor --------")
        log_predictions_lgbm = lightgbm_model.predict(X_test)
        actual_predictions_lgbm = np.expm1(log_predictions_lgbm)

        logging.info(f"---------- Making predictions on test data using CatBoost regressor --------")
        log_predictions_cat = catboost_model.predict(X_test)
        actual_predictions_cat = np.expm1(log_predictions_cat)

        logging.info(f"---------- Calulating avgerage predictions on test data --------")
        avg_predictions = (actual_predictions_xgb + actual_predictions_lgbm + actual_predictions_cat) / 3


        
        # logging.info("Predictions on test data completed")
        logging.info(f"evaluting avgerage predictions of the models")

        r2_square = r2_score(y_test_normal, avg_predictions)
        mae = mean_absolute_error(y_test_normal, avg_predictions)
        rmse = np.sqrt(mean_squared_error(y_test_normal, avg_predictions))
        mape = mean_absolute_percentage_error(y_test_normal, avg_predictions)

        logging.info(f'R2 Score of test data: {r2_square}')
        logging.info(f'Mean Absolute Error of test data: {mae}')
        logging.info(f'Root Mean Squared Error of test data: {rmse}')
        logging.info(f'Mean Absolute Percentage Error of test data: {mape}')


        results = pd.DataFrame({
            'Actual': y_test_normal,
            'Predicted': avg_predictions
        })
        results_df = pd.DataFrame(columns=['id','week','Actual','Predicted'])
        results_df['id'] = test_df[test_df['week'].isin(range(136,146))]['id'].reset_index(drop=True)
        results_df['week'] = test_df[test_df['id'].isin(range(136,146))]['week'].reset_index(drop=True)
        results_df['Actual'] = results['Actual'].values
        results_df['Predicted'] = results['Predicted'].values

        results_df['category'] = test_df[test_df['week'].isin(range(136,146))]['category'].reset_index(drop=True)
        results_df['APE'] = np.abs((results_df['Actual'] - results_df['Predicted']) / results_df['Actual']) #absolute percentage error

        os.makedirs(os.path.dirname(os.path.join('artifacts', 'results_df.csv')), exist_ok=True)

        # 3. Calculate MAPE by Category
        category_mape = results_df.groupby('category')['APE'].mean().sort_values(ascending=False)
        
        logging.info("--- MAPE by category ---")
        logging.info(category_mape)

        return avg_predictions
