# import os
# import sys
# import numpy as np
# import pandas as pd
# import pickle
# import tensorflow as tf
# from dataclasses import dataclass
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
# from keras.models import Model
# from keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Flatten, Dropout
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from keras.optimizers import Adam

# from components.lstm_data_handler import LSTMDataProcessor

# from logger import logging
# from exception import CustomException  
# from utils import save_object, load_object

# from logger import logging
# from exception import CustomException

# # Configure GPU memory growth to avoid OOM errors
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# @dataclass
# class LSTMConfig:
#     model_path: str = os.path.join('artifacts', 'lstm_model.keras')
#     batch_size: int = 512
#     epochs: int = 500

# class LSTMModelTrainer:
#     def __init__(self):
#         self.config = LSTMConfig()

#     def build_model(self, num_numerical_features, num_meals, num_centers):
#         """
#         Constructs the LSTM model with Embeddings using Keras Functional API
#         """
#         # --- Inputs ---
#         # 1. Time Series Data: (Window Size, Features)
#         input_num = Input(shape=(self.config.window_size, num_numerical_features), name='numerical_input')
        
#         # 2. Categorical IDs (Static context)
#         input_meal = Input(shape=(1,), name='meal_input')
#         input_center = Input(shape=(1,), name='center_input')

#         # --- Embeddings ---
#         # Learn a vector representation for meals and centers
#         emb_meal = Embedding(input_dim=num_meals, output_dim=10, name='meal_embedding')(input_meal)
#         emb_center = Embedding(input_dim=num_centers, output_dim=10, name='center_embedding')(input_center)
        
#         # Flatten embeddings to concat with LSTM output
#         flat_meal = Flatten()(emb_meal)
#         flat_center = Flatten()(emb_center)

#         # --- LSTM Layers ---
#         # The LSTM learns the pattern from the numerical sequence
#         x = LSTM(64, return_sequences=False, name='lstm_layer')(input_num)
#         x = Dropout(0.2)(x) # Prevent overfitting

#         # --- Combine ---
#         combined = Concatenate()([x, flat_meal, flat_center])

#         # --- Dense Layers ---
#         x = Dense(32, activation='relu')(combined)
#         x = Dropout(0.1)(x)
#         output = Dense(1, activation='linear', name='output')(x)

#         model = Model(inputs=[input_num, input_meal, input_center], outputs=output)
        
#         optimizer = Adam(learning_rate=0.001)
#         model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
#         return model

#     def initiate_training(self, train_path):
#         try:
#             logging.info("Loading Data...")
#             df = pd.read_csv(train_path)

#             # Feature Engineering (Cyclical time features needed)
#             df['week_of_year'] = df['week'].apply(lambda x: x % 52 if x % 52 != 0 else 52)
#             df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
#             df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

#             # Process Data
#             processor = LSTMDataProcessor()
#             X, y = processor.prepare_data(df)
            
#             # Split Train/Validation
#             # We split strictly by time logic or random (random is okay for X/y arrays here 
#             # because windows were already created safely)
            
#             # Unpacking X for splitting is tricky, so we split indices
#             indices = np.arange(len(y))
#             idx_train, idx_val = train_test_split(indices, test_size=0.2, random_state=42)

#             X_train = [X[0][idx_train], X[1][idx_train], X[2][idx_train]]
#             y_train = y[idx_train]
            
#             X_val = [X[0][idx_val], X[1][idx_val], X[2][idx_val]]
#             y_val = y[idx_val]

#             logging.info(f"Data Shapes - Train Num: {X_train[0].shape}, Val Num: {X_val[0].shape}")

#             # Build Model
#             num_meals = len(processor.meal_encoder.classes_)
#             num_centers = len(processor.center_encoder.classes_)
#             num_features = X_train[0].shape[2]

#             model = self.build_model(num_features, num_meals, num_centers)
#             model.summary()

#             # Callbacks for optimized training
#             callbacks = [
#                 EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
#                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
#             ]

#             logging.info("Starting Training...")
#             history = model.fit(
#                 X_train, y_train,
#                 validation_data=(X_val, y_val),
#                 epochs=self.config.epochs,
#                 batch_size=self.config.batch_size,
#                 callbacks=callbacks,
#                 verbose=1
#             )

#             logging.info("Training Completed. Saving Model...")
#             model.save(self.config.model_path)
            
#             # Evaluation on Validation Set
#             preds_log = model.predict(X_val)
#             preds = np.expm1(preds_log)
#             y_val_actual = np.expm1(y_val)
            
#             mape = mean_absolute_percentage_error(y_val_actual, preds)
#             mae = mean_absolute_error(y_val_actual, preds)
#             r2 = r2_score(y_val_actual, preds)
            
#             logging.info(f"Validation R2: {r2}")
#             logging.info(f"Validation MAE: {mae}")
#             logging.info(f"Validation MAPE: {mape}")


#             return r2, mape, mae

#         except Exception as e:
#             raise CustomException(e, sys) from e