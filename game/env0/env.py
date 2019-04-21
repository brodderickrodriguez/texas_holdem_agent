#!/usr/bin/env python3
# Brodderick Rodriguez
# Auburn University - CSSE
# 19 Mar. 2019

import logging
from agent_mcts import MCTS_Agent
from model import Residual_CNN
from mcts_config import reg, lr, hidden_layers, memory_size, episodes
from memory import Memory
import random
import pickle
import numpy as np
from texas_holdem import TexasHoldem, GameStage
import time
import sys

toolbar_width = 30



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
        self.current_model = Residual_CNN(reg, lr, (1,3,5), 3, hidden_layers)
        self.best_model    = Residual_CNN(reg, lr, (1,3,5), 3, hidden_layers)
        self.current_agent = MCTS_Agent()
        self.best_agent = MCTS_Agent()
        self.env = Environment()
        self.scores = [0, 0, 0]
        try:
            self.memory = pickle.load('/run/memory/memory.p')
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
        iter = 0
        while True:
            self.memory, scores = self.play_matches(self.best_agent, self.current_agent, episodes, memory=self.memory)
            self.memory.clear_stmemory()
            print(scores)
            print(len(self.memory.ltmemory))
            if len(self.memory.ltmemory) >= memory_size:
                self.current_agent.replay(self.memory.ltmemory)

                if iter % 5 == 0:
                    pickle.dump(self.memory, open('/run/memory/memory.p', "wb"))

                for s in random.sample(self.memory.ltmemory, min(1000, len(self.memory.ltmemory))):
                    current_value, current_probs, _ = self.current_agent.get_preds(s['state'])
                    best_value, best_probs, _ = self.best_agent.get_preds(s['state'])
            
            iter += 1
                

    def play_matches(self, player1, player2, EPISODES, memory=None):
        # setup toolbar
        sys.stdout.write("[%s]" % (" " * (toolbar_width)))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
        scores = []
        for e in range(EPISODES):      
            sys.stdout.write("-")
            sys.stdout.flush()
            pi, score = self.play([player1, player2])
            scores.append(score)
            if memory != None:
                memory.commit_stmemory([player1.state, player2.state], [player1.state.value, player2.state.value], pi)

        sys.stdout.write("\n")
        return memory, scores


    
    def play(self, agents):
        best = 0            
        self.env.reset()
        for agent in agents:
            best = max(agent.state.rank, best)
        for agent in agents:
            self.env.add_agent(agent)
            agent.state.hidden = best
            agent.mcts = None
            agent.state.playing = 1
        self.env.setup_game()
        
        while self.env._game.game_stage != GameStage.SHOWDOWN and agents[0].state.playing == 1 and agents[1].state.playing == 1:
            for agent in agents:
                agent.state.gs += 1
            actions = self.env.step(agents)
            a, pi, mcts_val, nn_val = actions[0]
    

        if agents[0].state.playing and not agents[1].state.playing:
            score = 0
        if not agents[0].state.playing and agents[1].state.playing:
            score = 1
        if not agents[0].state.playing and not agents[1].state.playing:
            if agents[0].state.rank > agents[1].state.rank:
                score = -1
            if agents[0].state.rank < agents[1].state.rank:
                score = 1
            if agents[0].state.rank == agents[1].state.rank:
                score = 0
        
        self.memory.commit_ltmemory()
        return pi, score



if __name__ == '__main__':
    logging.basicConfig(level=None)
    Tests()
