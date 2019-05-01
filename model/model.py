from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import numpy as np
from threading import Thread, Lock
import os
from keras.optimizers import Adam
lock = Lock()
import keras.backend as K
from keras.utils import plot_model
import pickle

LEARNING_RATE = 0.01
DISCOUNT = 0.9

class HarvestModel:

    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)
        model = Sequential()
        model.add(Conv2D(121, 5, input_shape=(1, 15, 15), data_format="channels_first"))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(49, 5, data_format="channels_first"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Flatten())
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='linear'))
        adam = Adam(lr=LEARNING_RATE)
        model.compile(adam, 'mse')
        if os.path.exists("model.h5"):
            model = load_model("model.h5")
        self.model = model
        self.input_shape = (1, 1, 15, 15)

    def predict(self, env):
        input = np.zeros(self.input_shape)
        input[0, 0] = env
        # K.get_session().run(tf.global_variables_initializer())
        return self.model.predict(input)

    def fit(self, state, pred):
        # K.get_session().run(tf.global_variables_initializer())
        self.model.fit(state, pred)

    def train(self, buffer):
        lock.acquire()
        if os.path.exists("model.h5"):
            self.model = load_model("model.h5")
        inp = np.zeros((99, 1, 15, 15))
        predictions = np.zeros((99, 3))
        batchind = 0
        for i in range(len(buffer)):
            if i % 10 == 0 and not i % len(buffer) == 0:
                max_reward, gotten_rewards, first_reward = self.get_rewards(buffer[i:i + 4])
                delta = first_reward + max_reward - gotten_rewards
                dit = buffer[i]
                state = dit["state"]
                pred = dit["predict"]
                move = dit["best_move"]
                if move == 'left':
                    pred[0][0] += delta
                elif move == 'move':
                    pred[0][1] += delta
                else:
                    pred[0][2] += delta
                predictions[batchind] = pred
                inp[batchind, 0] = state
                batchind += 1
        self.fit(inp, predictions)
        self.model.save("model.h5")
        lock.release()
        print("ended fitting")

    def get_rewards(self, bufferslice):
        q_bar = 0
        for i in range(len(bufferslice)):
            dit = bufferslice[i]
            if i == 0:
                first_reward = dit["reward"]
                q = dit["reward"]
            else:
                q_bar += (DISCOUNT ** i) * dit["max_reward"]
                q += (DISCOUNT ** i) * dit["reward"]
        return q_bar, q, first_reward




