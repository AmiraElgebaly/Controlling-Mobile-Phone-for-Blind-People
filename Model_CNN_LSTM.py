
import os
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, TimeDistributed, Conv1D, ZeroPadding1D, GRU
from keras.layers import Lambda, Input, Dropout, Masking, BatchNormalization, Activation
from keras.models import Model, load_model



def cnn_lstm(input_dim, output_dim, dropout=0.2, n_layers=1):
    #     # Input data type
    dtype = 'float32'

    # ---- Network model ----
    input_data = Input(name='the_input', shape=input_dim, dtype=dtype)

    # 1 x 1D convolutional layers with strides 4
    x = Conv1D(filters=256, kernel_size=10, strides=4, name='conv_1')(input_data)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout, name='dropout_1')(x)

    x = LSTM(128, activation='relu', return_sequences=True,
             dropout=dropout, name='lstm_1')(x)
    x = LSTM(128, activation='relu', return_sequences=False,
             dropout=dropout, name='lstm_2')(x)

    #     # 1 fully connected layer DNN ReLu with default 20% dropout
    x = Dense(units=64, activation='relu', name='fc')(x)
    x = Dropout(dropout, name='dropout_2')(x)

    # Output layer with softmax
    y_pred = Dense(units=output_dim, activation='softmax', name='softmax')(x)

    network_model = Model(inputs=input_data, outputs=y_pred)

    return network_model
