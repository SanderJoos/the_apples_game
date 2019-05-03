from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
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
from keras.activations import softmax

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
        # model.add(Dropout(0.5))
        model.add(Conv2D(49, 5, data_format="channels_first"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Flatten())
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='relu'))
        adam = Adam(lr=LEARNING_RATE)
        model.compile(adam, 'mse')
        if os.path.exists("model.h5"):
            model.load_weights("model.h5")
            print("loaded")
        self.model = model
        self.input_shape = (1, 1, 15, 15)

    # def softmax_asix_1(self,x):
    #     return softmax(x, axis=1)

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
            self.model.load_weights("model.h5")
        nb_of_fits = int((len(buffer) / 10) + 1)
        inp = np.zeros((nb_of_fits, 1, 15, 15))
        predictions = np.zeros((nb_of_fits, 3))
        batchind = 0
        for i in range(len(buffer)):
            if i % 10 == 0 and not i % len(buffer) == 0:
                dit = buffer[i]
                state = dit["state"]
                pred = self.get_best_prediction(buffer[i:i + 7])
                # max_reward, gotten_rewards, first_reward = self.get_rewards(buffer[i:i + 4])
                # delta = first_reward + max_reward - gotten_rewards
                # dit = buffer[i]
                # state = dit["state"]
                # pred = dit["predict"]
                # move = dit["best_move"]
                # if move == 'left':
                #     pred[0][0] += delta
                # elif move == 'move':
                #     pred[0][1] += delta
                # else:
                #     pred[0][2] += delta
                predictions[batchind] = pred
                inp[batchind, 0] = state
                batchind += 1
        self.fit(inp, predictions)
        self.model.save_weights("model.h5")
        lock.release()
        print("ended fitting")

    def get_best_prediction(self, bufferslice):
        orientation = bufferslice[0]["orientation"]
        xl, yl, orl, reward_l = self.get_left_state(bufferslice[0], 7, 7, orientation)
        xm, ym, orm, reward_m = self.get_move_state(bufferslice[0], 7, 7, orientation)
        xr, yr, orr, reward_r = self.get_right_state(bufferslice[0], 7, 7, orientation)
        reward = np.zeros((1, 3))
        new_buffer = bufferslice[1:]
        left_reward = reward_l * DISCOUNT + self.get_rewards(new_buffer, xl, yl, orl)
        move_reward = reward_m * DISCOUNT + self.get_rewards(new_buffer, xm, ym, orm)
        right_reward = reward_r * DISCOUNT + self.get_rewards(new_buffer, xr, yr, orr)
        reward[0][0] = left_reward
        reward[0][1] = move_reward
        reward[0][2] = right_reward
        return reward

    def get_rewards(self, bufferslice, x, y, orientation):
        if len(bufferslice) == 0:
            return 0
        else:
            fac = DISCOUNT ** (5 - len(bufferslice))
            xl, yl, orl, reward_l = self.get_left_state(bufferslice[0], x, y, orientation)
            xm, ym, orm, reward_m = self.get_move_state(bufferslice[0], x, y, orientation)
            xr, yr, orr, reward_r = self.get_right_state(bufferslice[0], x, y, orientation)
            new_buffer = bufferslice[1:]
            return max(reward_l * fac + self.get_rewards(new_buffer, xl, yl, orl),
                       reward_m * fac + self.get_rewards(new_buffer, xm, ym, orm),
                       reward_r * fac + self.get_rewards(new_buffer, xr, yr, orr))

    def get_left_state(self, buf, x, y, orientation):
        state = buf["state"]
        if orientation == 'left':
            return x+1, y, 'down', state[x + 1][y]
        elif orientation == 'right':
            return x-1, y, 'up', state[x - 1][y]
        elif orientation == 'down':
            return x, y+1, 'right', state[x][y + 1]
        else:
            return x, y-1, 'left', state[x][y - 1]

    def get_move_state(self, buf, x, y, orientation):
        state = buf["state"]
        if orientation == 'left':
            return x, y-1, 'left', state[x][y-1]
        elif orientation == 'right':
            return x, y+1, 'right', state[x][y+1]
        elif orientation == 'down':
            return x+1, y, 'down', state[x+1][y]
        else:
            return x-1, y, 'up', state[x-1][y]

    def get_right_state(self, buf, x, y, orientation):
        state = buf["state"]
        if orientation == 'left':
            return x-1, y, 'up', state[x-1][y]
        elif orientation == 'right':
            return x+1, y, 'down', state[x+1][y]
        elif orientation == 'down':
            return x, y-1, 'left', state[x][y-1]
        else:
            return x, y+1, 'right', state[x][y+1]

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
