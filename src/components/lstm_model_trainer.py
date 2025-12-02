import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error,mean_squared_error
from keras.models import Model
from keras.losses import Huber
from keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Flatten, Dropout,Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from keras.optimizers import Adam

from exception import CustomException
from logger import logging
from utils import evaluate_lstm_model

@dataclass
class LSTMModelTrainerConfig:
    trained_lstm_model_path: str = os.path.join('artifacts', 'lstm_model1.keras')

class LSTMModelTrainer:
    
    def __init__(self):
        self.lstm_model_trainer_config = LSTMModelTrainerConfig()
        self.cat_names = ['category', 'cuisine', 'center_type', 'region_code', 'city_code', 
                          'center_id', 'meal_id'
                          ]

    def build_lstm_model(self,window_size,num_dynamic_features,cat_vocab_sizes):

        input_dynamic = Input(shape=(window_size, num_dynamic_features), name='input_dynamic')

        x_lstm = LSTM(128, return_sequences=True, name='lstm_layer_1')(input_dynamic)
        x_lstm = Dropout(0.2)(x_lstm)
        x_lstm = LSTM(64,return_sequences=False, name='lstm_layer_2')(x_lstm)

        
        inputs_static = [] 
        embeddings_list = []      

        for feature_info in self.cat_names:
            vocab_size = cat_vocab_sizes[feature_info]

            cat_input = Input(shape=(1,), name=f'input_{feature_info}')
            inputs_static.append(cat_input)
            
            emb_dim = min(50, (vocab_size + 1) // 2)

            emb = Embedding(input_dim=vocab_size + 1, output_dim=emb_dim, name=f'emb_{feature_info}')(cat_input)

            emb = Flatten()(emb)
            
            embeddings_list.append(emb)

        combined = Concatenate(name='concat_layer')([x_lstm] + embeddings_list)

        x = Dense(64, activation='relu')(combined)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)

        output = Dense(1, activation='linear', name='output')(x)

        model = Model(inputs=[input_dynamic] + inputs_static, outputs=output)

        model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(), metrics=['mae', 'mape'])

        return model

        
    def initiate_lstm_model_training(self, train_inputs, y_train, test_inputs, y_test):
        try:
            logging.info("Starting LSTM Model Training")

            window_size = train_inputs[0].shape[1]
            
            num_num_features = train_inputs[0].shape[2] 
            logging.info(f"Window Size: {window_size}")
            logging.info(f"Number of Numerical Features: {num_num_features}")

            vocab_sizes = {}
            
            for i, name in enumerate(self.cat_names):
                vocab_sizes[name] = int(max(train_inputs[i+1].max(), test_inputs[i+1].max()) + 1)
            
            logging.info(f"Vocab Sizes: {vocab_sizes}")

            model = self.build_lstm_model(window_size, num_num_features, vocab_sizes)
            model.summary()

            callbacks = [
                EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
                ModelCheckpoint(self.lstm_model_trainer_config.trained_lstm_model_path, save_best_only=True, monitor='val_loss')
            ]

            # 5. Fit
            history = model.fit(
                x=train_inputs,
                y=y_train,
                validation_data=(test_inputs, y_test),
                epochs=50,
                batch_size=512, 
                callbacks=callbacks,
                verbose=1
            )
            
            logging.info("LSTM Training Completed")
            logging.info("Saving LSTM Model")
            model.save(self.lstm_model_trainer_config.trained_lstm_model_path)

            logging.info("Evaluating LSTM Model")

            r2,mae,rmse,mape = evaluate_lstm_model(self, model, test_inputs, y_test, history)


            return r2,mae,rmse,mape

        except Exception as e:
            raise CustomException(e, sys) from e