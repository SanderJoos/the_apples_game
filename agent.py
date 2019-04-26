#!/usr/bin/env python3
# encoding: utf-8
"""
agent.py
Template for the Machine Learning Project course at KU Leuven (2017-2018)
of Hendrik Blockeel and Wannes Meert.
Copyright (c) 2018 KU Leuven. All rights reserved.
"""
import sys
import argparse
import logging
import asyncio
import websockets
import json
from model.model import HarvestModel
from collections import defaultdict
import random
import numpy as np


logger = logging.getLogger(__name__)
games = {}
agentclass = None

DISCOUNT = 0.9
EXPLORATION_REDUCTION = 0.01


class Agent:

    def __init__(self, player, nb_rows, nb_cols):
        self.player = {player}
        self.ended = False
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.buffer = []
        self.score = 0
        self.model = HarvestModel()
        self.exploration = 0.95

        # var for buffering:
        self.action = -100
        self.reward = 0
        self.state = np.zeros((15, 15))
        self.next_state = np.zeros((15, 15))
        self.discount = DISCOUNT

    def add_player(self, player):
        self.player.add(player)

    def register_action(self, player_number, players, apples):
        # store action in buffer for training
        self.reward, self.next_state = self.get_environment(player_number, players, apples)
        if not self.action == -100:
            dic = {"state": self.state, "action": self.action, "reward": self.reward,
                "discount": self.discount, "next_state": self.next_state}
            self.buffer.append(dic)
            print(len(self.buffer))
        pass

    def next_action(self, player_number, players, apples):
        self.state = self.build_state(player_number, players, apples)
        move = self.get_move()
        if move == 'left':
            self.action = 0
        elif move == 'move':
            self.action = 1
        elif move == 'right':
            self.action = 2
        else:
            self.action = 0
        return move

    def get_move(self):
        rnd = random.random()     # this or is always true, outcomment to let reduction work
        if rnd <= self.exploration or rnd < 1:
            rnd = random.random()
            if rnd <= 0.33:
                move = 'left'
            elif rnd <= 0.66:
                move = 'move'
            else:
                move = 'right'
        else:
            # get model prediction here by using self.state as environment
            move = 'move'
        self.exploration -= EXPLORATION_REDUCTION
        return move

    def end_game(self):
        self.ended = True
        # upate model here with data in buffer


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
                        representation[index_c + 7, index_r + 7] = -player["score"]
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
                    nb_rows, nb_cols = msg["grid"]
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
                if msg["nextplayer"] in games[game].player:
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
