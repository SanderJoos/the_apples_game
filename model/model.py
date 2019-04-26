from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import numpy as np

LEARNING_RATE = 0.01
DISCOUNT = 0.9


class HarvestModel:

    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            model = Sequential()
            model.add(Conv2D(121, 5, input_shape=(1, 15, 15), data_format="channels_first"))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Conv2D(49, 5, data_format="channels_first"))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(3, 3)))
            model.add(Dense(10, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(3, activation='softmax'))
            self.model = model

    def predict(self, env):
        input = np.zeros((1, 1, 15, 15))
        input[0, 0] = env
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return self.model.predict(input)

    def fit(self, vector, result):
        self.model.fit([vector], [result])

    def train(self, buffer):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            return



