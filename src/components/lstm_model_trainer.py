import os
import sys
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Flatten, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam