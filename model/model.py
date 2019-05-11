from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, LeakyReLU
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
from random import randint

LEARNING_RATE = 0.001
DISCOUNT = 0.95

NUMBER_OF_BUFFERSLICES = 400
BATCHSIZE = 10
MODELNAME = "model.h5"

class HarvestModel:

    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        leaky = LeakyReLU()
        K.set_session(sess)
        model = Sequential()
        model.add(Dense(225))
        model.add(leaky)
        model.add(Dense(150))
        model.add(leaky)
        model.add(Dense(100))
        model.add(leaky)
        model.add(Dense(50))
        model.add(leaky)
        model.add(Dense(10))
        model.add(leaky)
        model.add(Dense(4))
        model.add(leaky)
        adam = Adam(lr=LEARNING_RATE)
        model.compile(adam, 'mae')
        if os.path.exists(MODELNAME):
            model = load_model(MODELNAME)
            print("loaded")
        self.model = model
        self.input_shape = (1, 225)

    # def softmax_asix_1(self,x):
    #     return softmax(x, axis=1)

    def predict(self, env):
        input = np.zeros(self.input_shape)
        input[0] = env.flatten()
        # K.get_session().run(tf.global_variables_initializer())
        return self.model.predict(input)

    def fit(self, state, pred):
        # K.get_session().run(tf.global_variables_initializer())
        self.model.fit(state, pred, batch_size=BATCHSIZE)

    def train(self, buffer):
        lock.acquire()
        if os.path.exists(MODELNAME):
            self.model.save(MODELNAME)
        # nb_of_fits = int((len(buffer) / 10) + 1)
        inp = np.zeros((NUMBER_OF_BUFFERSLICES, 225))
        predictions = np.zeros((NUMBER_OF_BUFFERSLICES, 4))
        batchind = 0
        for i in range(NUMBER_OF_BUFFERSLICES):
            index = randint(0, len(buffer) - 7)
            dit = buffer[index]
            state = dit["state"]
            pred = self.get_best_prediction(buffer[index:index + 7])
            print(pred)
            predictions[batchind] = pred
            inp[batchind] = state.flatten()
            batchind += 1
        self.fit(inp, predictions)
        self.model.save(MODELNAME)
        lock.release()
        print("ended fitting")

    def get_best_prediction(self, bufferslice):
        orientation = bufferslice[0]["orientation"]
        state = bufferslice[0]["state"]
        players = bufferslice[0]["players"]
        xl, yl, orl, reward_l = self.get_left_state(state, 7, 7, orientation)
        xm, ym, orm, reward_m = self.get_move_state(state, 7, 7, orientation)
        xr, yr, orr, reward_r = self.get_right_state(state, 7, 7, orientation)
        reward_f = self.get_fire_state(orientation, players)
        reward = np.zeros((1, 4))
        new_buffer = bufferslice[1:]
        left_reward = reward_l * DISCOUNT + self.get_rewards(new_buffer, xl, yl, orl, state, players)
        move_reward = reward_m * DISCOUNT + self.get_rewards(new_buffer, xm, ym, orm, state, players)
        right_reward = reward_r * DISCOUNT + self.get_rewards(new_buffer, xr, yr, orr, state, players)
        fire_reward = reward_f * DISCOUNT + self.get_rewards(new_buffer, 7, 7, orientation, state, players)
        reward[0][0] = left_reward
        reward[0][1] = move_reward
        reward[0][2] = right_reward
        reward[0][3] = fire_reward
        # reward[0] = self.softmax(reward[0])
        return reward

    def get_rewards(self, bufferslice, x, y, orientation, state, players):
        if len(bufferslice) == 0:
            return 0
        else:
            fac = DISCOUNT ** (5 - len(bufferslice))
            xl, yl, orl, reward_l = self.get_left_state(state, x, y, orientation)
            xm, ym, orm, reward_m = self.get_move_state(state, x, y, orientation)
            xr, yr, orr, reward_r = self.get_right_state(state, x, y, orientation)
            reward_f = self.get_fire_state(orientation, players)
            new_buffer = bufferslice[1:]
            return max(reward_l * fac + self.get_rewards(new_buffer, xl, yl, orl, state, players),
                       reward_m * fac + self.get_rewards(new_buffer, xm, ym, orm, state, players),
                       reward_r * fac + self.get_rewards(new_buffer, xr, yr, orr, state, players),
                       reward_f * fac + self.get_rewards(new_buffer, x, y, orientation, state, players))

    #TODO: what is the reward for shooting another player?
    #returns the absolute value of the shot player's score
    def get_fire_state(self, orientation, players):
        target_dist = 10
        score_shot_player = 0
        # loop through the whole state
        for player in players:
            i = player["location"][0]
            j = player["location"][1]
            # if a cell has a negative number this is another player
            # check if i can shoot this player and collect the reward
            if orientation == 'right' and i == 7:
                if ((j - 7) > 0) and (j - 7) < target_dist:
                    score_shot_player = player["score"]
                    target_dist = j - 7
            elif orientation == 'left' and i == 7:
                if ((j - 7) < 0) and (abs(7 - j) < target_dist):
                    score_shot_player = player["score"]
                    target_dist = 7 - j
            elif orientation == 'up' and j == 7:
                if ((i - 7) < 0) and (i - 7) < target_dist:
                    score_shot_player = player["score"]
                    target_dist = i - 7
            elif orientation == 'down' and j == 7:
                if ((i - 7) > 0) and (7 - i) < target_dist:
                    score_shot_player = player["score"]
                    target_dist = 7 - i
        reward = score_shot_player / 1000
        return reward

    def get_left_state(self, state, x, y, orientation):
        if orientation == 'left':
            return x+1, y, 'down', state[x + 1][y]
        elif orientation == 'right':
            return x-1, y, 'up', state[x - 1][y]
        elif orientation == 'down':
            return x, y+1, 'right', state[x][y + 1]
        else:
            return x, y-1, 'left', state[x][y - 1]

    def get_move_state(self, state, x, y, orientation):
        if orientation == 'left':
            return x, y-1, 'left', state[x][y-1]
        elif orientation == 'right':
            return x, y+1, 'right', state[x][y+1]
        elif orientation == 'down':
            return x+1, y, 'down', state[x+1][y]
        else:
            return x-1, y, 'up', state[x-1][y]

    def get_right_state(self, state, x, y, orientation):
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
