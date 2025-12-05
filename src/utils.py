import os
import sys

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dill
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit,cross_val_score,cross_validate

import optuna
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

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
        tscv = TimeSeriesSplit(n_splits=3)
        report_mae = {}
        report_mse = {}
        report_rmse = {}
        report_mape = {}
        # best_models = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]


            logging.info(f'GridSearchCV for {list(models.keys())[i]} started')
            gs = GridSearchCV(model,para,cv=tscv,n_jobs=6,verbose=3)
            gs.fit(X_train,y_train)

            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            # logging.info(f'RandomizedSearchCV for {list(models.keys())[i]} started')
            # # rs = RandomizedSearchCV(estimator=model,param_distributions=para,
            #                         n_iter=5,cv=tscv,n_jobs=6,verbose=5,refit=True,random_state=101)

            

            # rs.fit(X_train,y_train)

            # logging.info(f'RandomizedSearchCV results: {rs.cv_results_}')

            # logging.info(f'Best parameters for {list(models.keys())[i]}: {rs.best_params_}')

            # best_estimator = rs.best_estimator_
            # y_train_pred = best_estimator.predict(X_train)

            # y_test_pred = best_estimator.predict(X_test)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)


            train_model_r2_score = r2_score(y_train, y_train_pred)
            test_model_r2_score = r2_score(y_test, y_test_pred)
            
            train_model_mae = mean_absolute_error(y_train, y_train_pred)
            test_model_mae = mean_absolute_error(y_test, y_test_pred)

            train_model_mse = mean_squared_error(y_train, y_train_pred)
            test_model_mse = mean_squared_error(y_test, y_test_pred)

            train_model_rmse = np.sqrt(train_model_mse)
            test_model_rmse = np.sqrt(test_model_mse)

            train_model_mape = mean_absolute_percentage_error(y_train, y_train_pred)
            test_model_mape = mean_absolute_percentage_error(y_test, y_test_pred)


            report_r2[list(models.keys())[i]] = test_model_r2_score
            report_mae[list(models.keys())[i]] = test_model_mae
            report_mse[list(models.keys())[i]] = test_model_mse
            report_rmse[list(models.keys())[i]] = test_model_rmse
            report_mape[list(models.keys())[i]] = test_model_mape
            # best_models[list(models.keys())[i]] = best_estimator
            
        logging.info(f'Model report_r2:{report_r2}')
        logging.info(f'Model report_mae:{report_mae}')
        logging.info(f'Model report_mse:{report_mse}')
        logging.info(f'Model report_rmse:{report_rmse}')
        logging.info(f'Model report_mape:{report_mape}')

        return report_r2

    except Exception as e:
        logging.info(f"Error occurred in evaluate_models function: {e}")
        raise CustomException(e, sys)
    


# Functionns for hyperparameter tunning using optuna/

def objective(trial,X,y,model_name,cv):
    try:
        if model_name == 'XGBRegressor':
            params = {
                'booster': 'gbtree',
                'tree_method': 'hist',
                'device': 'cuda',
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 6, 16),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'objective': 'reg:tweedie',
                'eval_metric': 'mape',
                'verbosity': 3
            }
            model = XGBRegressor(**params)

        elif model_name == 'LGBM Regressor':
            params = {
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 6, 16),
            'device_type': 'gpu',
            'objective': 'tweedie',
            'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.0, 1.9),
            'metric': 'mape',
            }
            model = LGBMRegressor(**params)

        elif model_name == 'CatBoosting Regressor':
            params = {
                'depth': trial.suggest_int('depth', 6, 16),
                'learning_rate': trial.suggest_float('learning_rate', 0.01,0.2),
                'iterations': trial.suggest_int('iterations', 50, 500),
                'loss_function': 'Tweedie:variance_power=1.5',
                'eval_metric': 'MAPE',
                # 'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
            }
            model = CatBoostRegressor(**params)

        # elif model_name == 'Random Forest':
        #     params = {
        #         'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        #         'max_depth': trial.suggest_int('max_depth', 6, 16),
        #         'max_features': trial.suggest_categorical('max_features', ['sqrt']),
        #     }

            # model = RandomForestRegressor(**params)
        # else:
        #     param = {}
        #     model = LinearRegression()

        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
        # logging.info(f"Trial completed with R2 scores: {r2_scores}")    
        r2_score_mean = r2_scores.mean()

        return r2_score_mean

    except Exception as e:
        logging.info(f"Error occurred in objective function: {e}")
        raise CustomException(e, sys)
    

def tune_model_with_optuna(X_train, y_train, models, n_trials=20):
    report = {}
    best_models = {}
    
    tscv = TimeSeriesSplit(n_splits=3)

    for model_name in models.keys():
        logging.info(f"--- Starting Optuna Optimization for {model_name} ---")
        
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        
        func = lambda trial: objective(trial, X_train, y_train, model_name, tscv)
        study.optimize(func, n_trials=n_trials)

        # best_trials = study.best_trials
        # chosen_trial = None
        # best_r2_so_far = -float('inf')

        # for trial in best_trials:
        #     r2 = trial.values[0]
        #     mape = trial.values[1]

        #     if mape < 1.0:
        #         if r2 > best_r2_so_far:
        #             best_r2_so_far = r2
        #             chosen_trial = trial
        
        # if chosen_trial is None:
        #     chosen_trial = sorted(best_trials, key=lambda t: t.values[1])[0]

        # logging.info(f"selected trial params: {chosen_trial.params}, values: {chosen_trial.values}")
    

        logging.info(f"Best params for {model_name}: {study.best_params}")
        logging.info(f"Best R2 score for {model_name}: {study.best_value}")
        
        logging.info(f"Refitting {model_name} with best params...")
        
        best_params = study.best_params
        
        if model_name == "LGBM Regressor":
            model = LGBMRegressor(**best_params)
        elif model_name == "XGBRegressor":
            model = XGBRegressor(**best_params)
        elif model_name == "CatBoosting Regressor":
            model = CatBoostRegressor(**best_params)
        # else:
        #     model = RandomForestRegressor(**best_params)

        model.fit(X_train, y_train)
        
        report[model_name] = study.best_value
        best_models[model_name] = model
        
    return report, best_models
    

def evaluate_lstm_model(model, X_test, y_test_log, history=None):
    try:
        logging.info("Starting Model Evaluation...")

        # --- 1. Make Predictions ---
        # The model returns a 2D array (N, 1). We flatten it to 1D (N,).
        logging.info("Generating predictions...")
        predictions_log = model.predict(X_test, batch_size=512).flatten()

        # --- 2. Inverse Transform (Log -> Original) ---
        # We use expm1 because we used log1p during training
        predictions_actual = np.expm1(predictions_log)
        y_test_actual = np.expm1(y_test_log)

        # Sanity check: Ensure no negative predictions (impossible for orders)
        predictions_actual = np.maximum(predictions_actual, 0)

        # --- 3. Calculate Metrics ---
        r2 = r2_score(y_test_actual, predictions_actual)
        mae = mean_absolute_error(y_test_actual, predictions_actual)
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
        mape = mean_absolute_percentage_error(y_test_actual, predictions_actual)

        logging.info(f"--- Evaluation Results ---")
        logging.info(f"R2 Score: {r2:.4f}")
        logging.info(f"MAE: {mae:.4f}")
        logging.info(f"RMSE: {rmse:.4f}")
        logging.info(f"MAPE: {mape:.4f}")

        # --- 4. Plot Training History (Loss Curves) ---
        if history:
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('LSTM Training vs Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss (MSE)')
            plt.legend()
            plt.savefig(os.path.join('artifacts', 'lstm_loss_curve.png'))
            logging.info("Loss curve saved to artifacts/lstm_loss_curve.png")
            plt.close()

        # --- 5. Plot Actual vs Predicted (Snapshot) ---
        # Plotting all 30k points is messy. Let's plot the first 100 points to see the fit.
        plt.figure(figsize=(15, 6))
        plt.plot(y_test_actual[:150], label='Actual Orders', color='blue')
        plt.plot(predictions_actual[:150], label='Predicted Orders', color='orange', linestyle='--')
        plt.title('Actual vs Predicted Orders (First 150 Samples)')
        plt.legend()
        plt.savefig(os.path.join('artifacts', 'lstm_prediction_sample.png'))
        logging.info("Prediction sample plot saved to artifacts/lstm_prediction_sample.png")
        plt.close()

        return r2, mae, rmse, mape

    except Exception as e:
        raise CustomException(e, sys) from e
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info(f"Error occurred in load_object function: {e}")
        raise CustomException(e, sys)