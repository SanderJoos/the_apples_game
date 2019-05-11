#!/usr/bin/env python3
# encoding: utf-8
"""
agent.py
Template for the Machine Learning Project course at KU Leuven (2018-2019)
of Karl Tuys and Wannes Meert.
Copyright (c) 2019 KU Leuven. All rights reserved.
"""
import sys
import argparse
import logging
import asyncio
import time

import websockets
import json
from model.model import HarvestModel
from collections import defaultdict
import random
import numpy as np
import os
import pickle


logger = logging.getLogger(__name__)
games = {}
agentclass = None

DISCOUNT = 0.9
EXPLORATION_REDUCTION = 0.99
EXPLORATION = True


class Agent:

    def __init__(self, player, nb_rows, nb_cols):
        self.player = {player}
        self.ended = False

        # this is intentional to work with previous error (no need to
        # adjust all other code)
        self.nb_rows = nb_cols
        self.nb_cols = nb_rows

        self.buffer = []
        self.score = 0
        self.model = HarvestModel()
        self.exploration = 0.99

        # var for buffering:
        self.action = -100
        self.reward = 0
        self.state = np.zeros((15, 15))
        self.next_state = np.zeros((15, 15))
        self.discount = DISCOUNT
        self.best_move = ''
        self.orientation = ''
        self.max_reward = -100
        self.pred = np.zeros((1, 4))

    def add_player(self, player):
        self.player.add(player)

    def register_action(self, player_number, players, apples):
        # store action in buffer for training
        self.reward, self.next_state = self.get_environment(player_number, players, apples)
        if not self.action == -100:
            dic = {"state": self.state, "action": self.action, "reward": self.reward,
                "discount": self.discount, "next_state": self.next_state, "best_move": self.best_move,
                   "orientation": self.orientation, "max_reward": self.max_reward, "predict": self.pred}
            self.buffer.append(dic)
            print(len(self.buffer))
        pass

    def next_action(self, player_number, players, apples):
        self.state = self.build_state(player_number, players, apples)
        player = players[player_number - 1]
        self.best_move = self.get_best_move(self.state, player["orientation"])
        self.orientation = player["orientation"]
        move = self.get_move()
        if move == 'left':
            self.action = 0
        elif move == 'move':
            self.action = 1
        elif move == 'right':
            self.action = 2
        elif move == 'fire':
            self.action = 3
        else:
            self.action = 0
        return move

    def do_shoot(self, state, orientation):
        # loop through the whole state
        for i in range(len(state)):
            for j in range(len(state[i])):
                # if a cell has a negative number this is another player
                if state[i][j] < 0:
                    # check if i can shoot this player and collect the reward
                    if orientation == 'right' and i == 7:
                        if ((j - 7) > 0):
                            return True
                    elif orientation == 'left' and i == 7:
                        if ((j - 7) < 0):
                            return True
                    elif orientation == 'up' and j == 7:
                        if ((i - 7) < 0):
                            return True
                    elif orientation == 'down' and j == 7:
                        if ((i - 7) > 0):
                            return True

    def get_move(self):
        rnd = random.random()
        print("exploration chance: ", self.exploration)
        if EXPLORATION and rnd <= self.exploration:
            rnd = random.random()
            if self.do_shoot(self.state, self.orientation):
                if rnd <= 0.25:
                    self.pred[0] = [1, 0, 0, 0]
                    move = 'left'
                elif rnd <= 0.25:
                    self.pred[0] = [0, 1, 0, 0]
                    move = 'move'
                elif rnd <= 0.75:
                    self.pred[0] = [0, 0, 1, 0]
                    move = 'right'
                else:
                    self.pred[0] = [0, 0, 0, 1]
                    move = 'fire'
            else:
                if rnd <= 0.33:
                    self.pred[0] = [1, 0, 0, 0]
                    move = 'left'
                elif rnd <= 0.66:
                    self.pred[0] = [0, 1, 0, 0]
                    move = 'move'
                elif rnd <= 1:
                    self.pred[0] = [0, 0, 1, 0]
                    move = 'right'
        else:
            prob = self.model.predict(self.state)
            print(prob)
            self.pred[0] = prob[0]
            indices = [idx for idx, val in enumerate(prob[0]) if val == max(prob[0])]
            index = random.choice(indices)
            if index == 0:
                move = 'left'
            elif index == 1:
                move = 'move'
            elif index == 2:
                move = 'right'
            elif index == 3:
                move = 'fire'
        self.exploration *= EXPLORATION_REDUCTION
        return move

    def end_game(self):
        self.ended = True
        for i in range(16):
            if i in self.player:
                time.sleep(i*10)
        self.model.train(self.buffer)


    def get_key(self, elem):
        return elem[0]

    def get_best_move(self, state, orientation):
        return_list = []
        left_reward = self.get_left_reward(state, orientation)
        return_list.append((left_reward, "left"))
        right_reward = self.get_right_reward(state, orientation)
        return_list.append((right_reward, "right"))
        move_reward = self.get_move_reward(state, orientation)
        return_list.append((move_reward, "move"))
        fire_reward = self.get_fire_reward(state, orientation)
        return_list.append((fire_reward, "fire"))

        return_list.sort(key=self.get_key, reverse=True)
        return return_list[0][0]
        # if left_reward > right_reward:
        #     if left_reward > move_reward:
        #         if left_reward > fire_reward:
        #             self.max_reward = left_reward
        #             return 'left'
        #
        #     else:
        #         self.max_reward = move_reward
        #         return 'move'
        #         return 'move'
        # else:
        #     if right_reward > move_reward:
        #         self.max_reward = right_reward
        #         return 'right'
        #     else:
        #         self.max_reward = move_reward
        #         return 'move'

    # TODO: what is the reward for shooting another player?
    # returns the absolute value of the shot player's score
    def get_fire_reward(self, state, orientation):
        target_dist = 10
        score_shot_player = 0
        # loop through the whole state
        for i in range(len(state)):
            for j in range(len(state[i])):
                # if a cell has a negative number this is another player
                if state[i][j] < 0:
                    # check if i can shoot this player and collect the reward
                    if orientation == 'right' and i == 7:
                        if ((j - 7) > 0) and (j - 7) < target_dist:
                            score_shot_player = state[i][j]
                            target_dist = j - 7
                    elif orientation == 'left' and i == 7:
                        if ((j - 7) < 0) and (abs(7 - j) < target_dist):
                            score_shot_player = state[i][j]
                            target_dist = 7 - j
                    elif orientation == 'up' and j == 7:
                        if ((i - 7) < 0) and (i - 7) < target_dist:
                            score_shot_player = state[i][j]
                            target_dist = i - 7
                    elif orientation == 'down' and j == 7:
                        if ((i - 7) > 0) and (7 - i) < target_dist:
                            score_shot_player = state[i][j]
                            target_dist = 7 - i
        reward = -score_shot_player
        return reward

    def get_left_reward(self, state, orientation):
        if orientation == 'left':
            return state[8][7]
        elif orientation == 'right':
            return state[6][7]
        elif orientation == 'down':
            return state[7][8]
        else:
            return state[7][6]

    def get_move_reward(self, state, orientation):
        if orientation == 'left':
            return state[7][6]
        elif orientation == 'right':
            return state[7][8]
        elif orientation == 'down':
            return state[8][7]
        else:
            return state[6][7]

    def get_right_reward(self, state, orientation):
        if orientation == 'left':
            return state[6][7]
        elif orientation == 'right':
            return state[8][7]
        elif orientation == 'down':
            return state[7][6]
        else:
            return state[7][8]

    def build_state(self, player_number, players, apples):
        representation = np.zeros((15, 15))
        player = players[player_number - 1]
        row, col = player["location"]
        for index_r in range(-7, 8):
            for index_c in range(-7, 8):
                R = (row + index_r + self.nb_rows) % self.nb_rows
                C = (col + index_c + self.nb_cols) % self.nb_cols
                for a_row, a_col in apples:
                    if a_row == R and a_col == C:
                        representation[index_c + 7, index_r + 7] = 1
                for player in players:
                    p_row, p_col = player["location"]
                    if not p_row == "?" and p_row == R and p_col == C:
                        representation[index_c + 7, index_r + 7] = -player["score"]/100
        return representation

    def get_environment(self, player_number, players, apples):
        representation = self.build_state(player_number, players, apples)
        player = players[player_number - 1]
        score = player["score"]
        reward = score - self.score
        self.score = score
        return reward, representation

## MAIN EVENT LOOP

async def handler(websocket, path):
    logger.info("Start listening")
    game = None
    # msg = await websocket.recv()
    try:
        async for msg in websocket:
            try:
                msg = json.loads(msg)
            except json.decoder.JSONDecodeError as err:
                logger.error(err)
                return False
            game = msg["game"]
            answer = None
            if msg["type"] == "start":
                # Initialize game
                if msg["game"] in games:
                    games[msg["game"]].add_player(msg["player"])
                else:
                    nb_cols, nb_rows = msg["grid"]
                    games[msg["game"]] = agentclass(msg["player"],
                                                    nb_rows,
                                                    nb_cols)
                if msg["player"] == 1:
                    # Start the game
                    nm = games[game].get_move()
                    print('nm = {}'.format(nm))
                    if nm is None:
                        # Game over
                        logger.info("Game over")
                        continue
                    answer = {
                        'type': 'action',
                        'action': nm,
                    }
                else:
                    # Wait for the opponent
                    answer = None

            elif msg["type"] == "action":
                # An action has been played
                if msg["nextplayer"] in games[game].player and msg["nextplayer"] == msg["receiver"]:
                    # Compute your move
                    player_number = msg["nextplayer"]
                    apples = msg["apples"]
                    players = msg["players"]
                    nm = games[game].next_action(player_number, players, apples)
                    if nm is None:
                        # Game over
                        logger.info("Game over")
                        continue
                    answer = {
                        'type': 'action',
                        'action': nm
                    }
                elif msg["player"] in games[game].player:
                    player_number = msg["player"]
                    apples = msg["apples"]
                    players = msg["players"]
                    games[game].register_action(player_number, players, apples)
                else:
                    answer = None

            elif msg["type"] == "end":
                # End the game
                f = open("scores.txt", "a+")
                players = msg["players"]
                nr = msg["receiver"]
                f.write(('score Player: %s: %s \n' % (nr, players[nr - 1]["score"])))
                f.close()
                games[msg["game"]].end_game()
                answer = None
            else:
                logger.error("Unknown message type:\n{}".format(msg))

            if answer is not None:
                print(answer)
                await websocket.send(json.dumps(answer))
    except websockets.exceptions.ConnectionClosed as err:
        logger.info("Connection closed")
    logger.info("Exit handler")


def start_server(port):
    server = websockets.serve(handler, 'localhost', port)
    print("Running on ws://127.0.0.1:{}".format(port))
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()


## COMMAND LINE INTERFACE

def main(argv=None):
    global agentclass
    parser = argparse.ArgumentParser(description='Start agent to play the Apples game')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count', default=0, help='Quiet output')
    parser.add_argument('port', metavar='PORT', type=int, help='Port to use for server')
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    agentclass = Agent
    start_server(args.port)


if __name__ == "__main__":
    sys.exit(main())
