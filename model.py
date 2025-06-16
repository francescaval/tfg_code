# -*- coding: utf-8 -*-
"""
@author: Francesca Val Bagli

"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import SGD

def configure_tf_gpu_growth():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def build_ar_mrnn_model():
    model = Sequential([
        InputLayer(shape=(4,)),
        Dense(16, activation='sigmoid'),
        Dense(4,  activation='sigmoid'),
        Dense(1,  activation='sigmoid')
    ])
    model.compile(optimizer=SGD(learning_rate=0.009, momentum=0.95), loss='mse')
    return model
