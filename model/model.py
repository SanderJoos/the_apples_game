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
        self.input_shape = (1, 225)
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
        self.model.build(input_shape=self.input_shape)
        print(self.model.summary())

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
        player_number= bufferslice[0]["player_number"]
        xl, yl, orl, reward_l = self.get_left_state(state, 7, 7, orientation)
        xm, ym, orm, reward_m = self.get_move_state(state, 7, 7, orientation)
        xr, yr, orr, reward_r = self.get_right_state(state, 7, 7, orientation)
        reward_f = self.get_fire_state(orientation, players, player_number)
        reward = np.zeros((1, 4))
        new_buffer = bufferslice[1:]
        left_reward = reward_l * DISCOUNT + self.get_rewards(new_buffer, xl, yl, orl, state, players, player_number)
        move_reward = reward_m * DISCOUNT + self.get_rewards(new_buffer, xm, ym, orm, state, players, player_number)
        right_reward = reward_r * DISCOUNT + self.get_rewards(new_buffer, xr, yr, orr, state, players, player_number)
        fire_reward = reward_f * DISCOUNT + self.get_rewards(new_buffer, 7, 7, orientation, state, players, player_number)
        reward[0][0] = left_reward
        reward[0][1] = move_reward
        reward[0][2] = right_reward
        reward[0][3] = fire_reward
        # reward[0] = self.softmax(reward[0])
        return reward

    def get_rewards(self, bufferslice, x, y, orientation, state, players, player_number):
        if len(bufferslice) == 0:
            return 0
        else:
            fac = DISCOUNT ** (5 - len(bufferslice))
            xl, yl, orl, reward_l = self.get_left_state(state, x, y, orientation)
            xm, ym, orm, reward_m = self.get_move_state(state, x, y, orientation)
            xr, yr, orr, reward_r = self.get_right_state(state, x, y, orientation)
            reward_f = self.get_fire_state(orientation, players, player_number)
            new_buffer = bufferslice[1:]
            return max(reward_l * fac + self.get_rewards(new_buffer, xl, yl, orl, state, players, player_number),
                       reward_m * fac + self.get_rewards(new_buffer, xm, ym, orm, state, players, player_number),
                       reward_r * fac + self.get_rewards(new_buffer, xr, yr, orr, state, players, player_number),
                       reward_f * fac + self.get_rewards(new_buffer, x, y, orientation, state, players, player_number))

    #returns the absolute value of the shot player's score
    def get_fire_state(self, orientation, players, player_number):
        target_dist = 10
        score_shot_player = 0
        my_i = players[player_number-1]["location"][0]
        my_j = players[player_number-1]["location"][1]
        # loop through the whole state
        for player in players:
            i = player["location"][0]
            j = player["location"][1]
            # check if i can shoot this player and collect the reward
            if orientation == 'right' and i == my_i:
                if ((j - my_j) > 0) and (j - my_j) < target_dist:
                    score_shot_player = -1
                    target_dist = j - my_j
            elif orientation == 'left' and i == my_i:
                if ((j - my_j) < 0) and (abs(my_j - j) < target_dist):
                    score_shot_player = -1
                    target_dist = my_j - j
            elif orientation == 'up' and j == my_j:
                if ((i - my_i) < 0) and (i - my_i) < target_dist:
                    score_shot_player = -1
                    target_dist = i - my_i
            elif orientation == 'down' and j == my_i:
                if ((i - my_i) > 0) and (my_i - i) < target_dist:
                    score_shot_player = -1
                    target_dist = my_i - i
        reward = score_shot_player
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
