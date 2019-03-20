#!/usr/bin/env python3
# Brodderick Rodriguez
# Auburn University - CSSE
# 19 Mar. 2019

import logging
import numpy as np

from env0.player import Player
from env0.texas_holdem import TexasHoldem

import logging


class Environment:
    def __init__(self, players):
        self.players = players
        self.game = TexasHoldem(players)

        self.action_space = np.array([0, 1])
        self.observation_space = np.zeros((len(players), len(self.game.deck.GetFullDeck())))
        logging.info(('observation_space shape', self.observation_space.shape))

        pass

    def reset(self):
        pass


    def step(self, player, action):

        logging.info('received action from ' + str(player))

        pass

    def _step(self, action):


        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    plyrs = [Player() for _ in range(3)]

    env = Environment(plyrs)

    env.step(plyrs[0], None)

    env.game.step()


