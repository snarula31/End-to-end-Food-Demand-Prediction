import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit

from logger import logging
from exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report_r2 = {}
        # report_mae = {}
        # report_mse = {}
        # report_rmse = {}
        # report_mape = {}
        best_models = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            # gs = GridSearchCV(model,para,cv=3,n_jobs=4,verbose=3)
            # gs.fit(X_train,y_train)
            tscv = TimeSeriesSplit(n_splits=3)

            rs = RandomizedSearchCV(estimator=model,param_distributions=para,
                                    n_iter=5,cv=tscv,n_jobs=6,verbose=5,refit=True,random_state=101)

            logging.info(f'RandomizedSearchCV for {list(models.keys())[i]} started')

            rs.fit(X_train,y_train)

            # logging.info(f'RandomizedSearchCV results: {rs.cv_results_}')

            logging.info(f'Best parameters for {list(models.keys())[i]}: {rs.best_params_}')

            # model.set_params(**rs.best_params_)
            # model.fit(X_train,y_train)

            best_estimator = rs.best_estimator_
            y_train_pred = best_estimator.predict(X_train)

            y_test_pred = best_estimator.predict(X_test)

            train_model_r2_score = r2_score(y_train, y_train_pred)
            test_model_r2_score = r2_score(y_test, y_test_pred)
            
            # train_model_mae = mean_absolute_error(y_train, y_train_pred)
            # test_model_mae = mean_absolute_error(y_test, y_test_pred)

            # train_model_mse = mean_squared_error(y_train, y_train_pred)
            # test_model_mse = mean_squared_error(y_test, y_test_pred)

            # train_model_rmse = np.sqrt(train_model_mse)
            # test_model_rmse = np.sqrt(test_model_mse)

            # train_model_mape = mean_absolute_percentage_error(y_train, y_train_pred)
            # test_model_mape = mean_absolute_percentage_error(y_test, y_test_pred)


            report_r2[list(models.keys())[i]] = test_model_r2_score
            # report_mae[list(models.keys())[i]] = test_model_mae
            # report_mse[list(models.keys())[i]] = test_model_mse
            # report_rmse[list(models.keys())[i]] = test_model_rmse
            # report_mape[list(models.keys())[i]] = test_model_mape
            best_models[list(models.keys())[i]] = best_estimator
            
        logging.info(f'Model report_r2:{report_r2}')
        # logging.info(f'Model report_mae:{report_mae}')
        # logging.info(f'Model report_mse:{report_mse}')
        # logging.info(f'Model report_rmse:{report_rmse}')
        # logging.info(f'Model report_mape:{report_mape}')

        return report_r2,best_models

    except Exception as e:
        logging.info(f"Error occurred in evaluate_models function: {e}")
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info(f"Error occurred in load_object function: {e}")
        raise CustomException(e, sys)