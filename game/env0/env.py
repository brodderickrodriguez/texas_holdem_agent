#!/usr/bin/env python3
# Brodderick Rodriguez
# Auburn University - CSSE
# 19 Mar. 2019

import logging
from mcts.agent_mcts import Agent
from mcts.model import Residual_CNN
from mcts.mcts_config import reg, lr, hidden_layers, memory_size, episodes
from mcts.memory import Memory
import numpy as np
from texas_holdem import TexasHoldem, GameStage


class Environment:
    def __init__(self):
        self._agents, self._agent_actions, self._game = [], {}, None
        self._called_env_step = False

    def add_agent(self, agent):
        agent.env = self
        self._agents.append(agent)

    def setup_game(self):
        self._game = TexasHoldem(self._agents)
        self._agent_actions = {}

    def step(self, agents):
        if self._game is None:
            raise RuntimeError('game has not been created. Must call Environment.setup_game() first.')

        observation = self._game.step()
        actions = [agent.update(observation, 1 if observation[0] != GameStage.INITIAL else 0) for agent in agents]
        self._agent_actions = {}
        return actions
    
    def reset(self):
        self._agents, self._agent_actions, self._game = [], {}, None
        self._called_env_step = False


class Tests:
    def __init__(self):
        self.current_model = Residual_CNN(reg, lr, np.array([1, 3]).shape, 4, hidden_layers)
        self.best_model    = Residual_CNN(reg, lr, np.array([1, 3]).shape, 4, hidden_layers)
        self.current_agent = Agent()
        self.best_agent = Agent()
        self.env = Environment()
        self.scores = [0, 0, 0]
        try:
            self.memory = pickle.load('/run/memory')
        except:
            self.memory = Memory(memory_size)
        try:
            bm_temp = best_model.read()
            self.current_model.model.set_weights(bm_temp.get_weights())
            self.best_model.model.set_weights(bm_temp.get_weights())
        except:
            self.best_model.model.set_weights(self.current_model.model.get_weights())
        self.current_agent.model = self.current_model
        self.best_agent.model    = self.best_model
        while True:
            self.memory = self.play_matches(self.best_agent, self.best_agent, self.best_agent, episodes, memory=self.memory)


    def play_matches(self, player1, player2, player3, EPISODES, memory=None):
        for e in range(EPISODES):
            self.env.reset()
            
            pi = self.play([player1, player2, player3])

            if memory != None:
                memory.commit_stmemory([player1.state, player2.state, player3.state], [player1.state, player2.state, player3.state], pi)
        return memory


    
    def play(self, agents):
        pi = 0
        while self.env._game.game_stage != GameStage.HAND_COMPLETE:
            actions = self.env.step(agents)
            a, pi, mcts_val, nn_val = actions[0]
        return pi



if __name__ == '__main__':
    logging.basicConfig(level=None)
    Tests()
