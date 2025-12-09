import os 
import sys

import numpy as np

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,tune_model_with_optuna

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model1.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Initiating model trainer")
            logging.info("Splitting training and testing input data")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Training and testing split completed")

            models = {
                # "Linear Regression": LinearRegression(),
                # "Random Forest": RandomForestRegressor(random_state= 58),
                "LGBM Regressor": LGBMRegressor(random_state=58),
                "XGBRegressor": XGBRegressor(verbosity=3,random_state=58),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False,random_state=85)
            }


            # params = {
            #     "Random Forest": {
            #         'n_estimators': [100,150,200],  
            #         'max_features': ['sqrt'],
            #         'max_depth': [7,8,9],
            #         'min_samples_leaf': [2,3,4],     
            #     },
            #     "LGBM Regressor": {
            #         'boosting_type': ['gbdt'],
            #         'learning_rate': [0.01,0.1,0.13,0.16],
            #         'n_estimators': [100,150,200],
            #         'max_depth': [7,8,9],
            #         'device_type': ['gpu']  
            #     },
            #     "Linear Regression": {},
            #     "XGBRegressor": {
            #         'booster': ['gbtree'], 
            #         'tree_method': ['hist'],
            #         'max_depth': [7,8,9],
            #         'learning_rate': [0.01,0.1,0.14],
            #         'n_estimators': [50,100,150,200],
            #         'device': ['cuda']
            #     },
            #     "CatBoosting Regressor": {
            #         'max_depth': [7,8,9],
            #         'learning_rate': [0.01,0.1,0.12],
            #         'iterations': [100,150,200,250],
            #         'l2_leaf_reg': [4,6,8],
            #         'loss_function': ['RMSE'],
            #     }
            # }
            
            logging.info("Model training initiated")
            logging.info("Evaluating models")

            # model_report,best_trained_models = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,param=params) --randomizedsearchcv
            # model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,param=params)

            model_report,best_trained_models = tune_model_with_optuna(X_train, y_train, models=models, n_trials=20)

            logging.info(f'model report: {model_report}')

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            # best_model = best_trained_models[best_model_name]with --randomised searchcv
            best_model = best_trained_models[best_model_name]

            # if best_model_score < 0.6:
            #     raise CustomException("No best model found")
            # logging.info(f"Best model found: {best_model_name} with r2 score: {best_model_score}")

            for model_name, model in best_trained_models.items():
            
                save_object(
                file_path=os.path.join(f'artifacts',f'{model_name}_model.pkl'),
                    obj=model
                )

            logging.info("Model training completed")
            logging.info('predictions on test data')

            y_test_normal = np.expm1(y_test)
            log_predictions = best_model.predict(X_test)
            actual_predictions = np.expm1(log_predictions)
            r2_square = r2_score(y_test_normal, actual_predictions)
            mae = mean_absolute_error(y_test_normal, actual_predictions)
            rmse = np.sqrt(mean_squared_error(y_test_normal, actual_predictions))
            mape = mean_absolute_percentage_error(y_test_normal, actual_predictions)


            logging.info(f'R2 Score of test data: {r2_square}')
            logging.info(f'Mean Absolute Error of test data: {mae}')
            logging.info(f'Root Mean Squared Error of test data: {rmse}')
            logging.info(f'Mean Absolute Percentage Error of test data: {mape}')

            return (r2_square, mae, rmse, mape)
            
            # pass
        except Exception as e:
            raise CustomException(e, sys) from e

