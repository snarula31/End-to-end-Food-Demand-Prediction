import os 
import sys

import numpy as np
from scipy.stats import randint, uniform

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from exception import CustomException
from logger import logging
from utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

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
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state= 58),
                "LGBM Regressor": LGBMRegressor(random_state=58),
                "XGBRegressor": XGBRegressor(verbosity=3,random_state=58),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False,random_state=85)
            }


            params = {
                "Random Forest": {
                    'n_estimators': randint(50, 250),  
                    'max_features': ['sqrt', 'log2'],
                    'max_depth': randint(6, 16),     
                },
                "LGBM Regressor": {
                    'boosting_type': ['gbdt'],
                    'learning_rate': uniform(0.01, 0.2),
                    'n_estimators': randint(50, 250),
                    'max_depth': randint(6, 16),
                    'device_type': ['gpu']  
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'booster': ['gbtree'], 
                    'tree_method': ['hist'],
                    'max_depth': randint(4, 16),
                    'learning_rate': uniform(0.01, 0.2),
                    'n_estimators': randint(50, 250),
                    'device': ['cuda']
                },
                "CatBoosting Regressor": {
                    'depth': randint(6, 16),
                    'learning_rate': uniform(0.01, 0.2),
                    'iterations': randint(50, 250),
                    'l2_leaf_reg': randint(1, 10),
                }
            }

            # logging.info("STARTNG SANITY CHECK RUN")

            # import time

            # model = XGBRegressor(tree_method='gpu_hist', device='cuda',n_estimators=120, verbosity=3)

            # start_time = time.time()
            # model.fit(X_train, y_train)
            # end_time = time.time()

            # logging.info(f"Model training completed in {end_time - start_time:.3f} seconds")

            # sys.exit("exiting after sanity check")

            
            logging.info("Model training initiated")
            logging.info("Evaluating models")

            model_report,best_trained_models = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,param=params)

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = best_trained_models[best_model_name]

            # if best_model_score < 0.6:
            #     raise CustomException("No best model found")
            logging.info(f"Best model found: {best_model_name} with r2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
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

        
#  check categorical columns again. specifically emailer_for_promotion and homepage_featured - DONE
# add more features if possible -  DONE
# check for any null values - DONE
# optimize the hyperparameters for all models - ONGOING
#  improve models predicitions - ONGOING

# find solution for training time - DONE (using GPU based training for some models) REDUCED TO 1.5 - 2 HOURS FROM 5+ HOURS)

# run the optimized training set


# add new features to improve model performance
# -> change categorical encoding technique, if new columns added
# use bayesian optimization for hyperparameter tuning (OPTUNA)
# further reduce model training time 
# -> find a fix for long training time of xgboost even on GPU
#  use different transformation techniques for target variable to make it a normal distribution (quantile transformation)
# check implentaion od ewma on google ai studio
# fix features using expanding window technique