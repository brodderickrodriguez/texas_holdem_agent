#!/usr/bin/env python3
# Brodderick Rodriguez
# Auburn University - CSSE
# 19 Mar. 2019

import logging
from agent import *
from agent_mcts import MCTS_Agent
from model import Residual_CNN
from mcts_config import reg, lr, hidden_layers, memory_size, episodes, scoring_threshold
from memory import Memory
import random
import pickle
import numpy as np
from texas_holdem import TexasHoldem, GameStage
import time
import sys
from treys import Evaluator

toolbar_width = episodes



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

import os
import time
class Tests:
    def __init__(self, mode):
        self.mode = mode
        self.current_model = Residual_CNN(reg, lr, (1,3,5), 3, hidden_layers)
        self.best_model    = Residual_CNN(reg, lr, (1,3,5), 3, hidden_layers)
        self.current_agent = MCTS_Agent()
        self.best_agent = MCTS_Agent()
        self.env = Environment()
        self.scores = [0, 0, 0]
        try:
            self.memory = pickle.load('memory.p')
            print('loaded memory!')
        except:
            print('memory load unsuccessful!')
            self.memory = Memory(memory_size)
        try:
            bm_temp = self.best_model.read()
            self.current_model.model.set_weights(bm_temp.get_weights())
            self.best_model.model.set_weights(bm_temp.get_weights())
            print('loaded model!')
        except:
            self.best_model.model.set_weights(self.current_model.model.get_weights())
        self.current_agent.model = self.current_model
        self.best_agent.model    = self.best_model
        os.system('clear')
        if self.mode == 0:
            iter = 0
            while True:
                self.memory, scores = self.play_matches(self.best_agent, self.current_agent, episodes, memory=self.memory)
                self.memory.clear_stmemory()
                print(scores)
                print(len(self.memory.ltmemory))
                if len(self.memory.ltmemory) >= memory_size:
                    self.current_agent.replay(self.memory.ltmemory)

                    if iter % 5 == 0:
                        pickle.dump(self.memory, open('memory.p', "wb"))

                    for s in random.sample(self.memory.ltmemory, min(1000, len(self.memory.ltmemory))):
                        current_value, current_probs = self.current_agent.get_preds(s['state'])
                        best_value, best_probs = self.best_agent.get_preds(s['state'])
                
                    if np.sum(scores) > scoring_threshold:
                        self.best_model.model.set_weights(self.current_model.model.get_weights())
                        self.best_model.write()
                iter += 1
        scores= {"User": 0, "Agent": 0}
        while self.mode == 1:
            pi, score = self.play([User(), self.best_agent])
            if score >= 1:
                scores["User"] += score
            if score <= -1:
                scores["Agent"] -= score
            if score == 0:
                scores["User"] += 0.5
                scores["Agent"] += 0.5
            print('Score:', scores)


    def play_matches(self, player1, player2, EPISODES, memory=None):
        # setup toolbar
        sys.stdout.flush()
        scores = []
        for e in range(EPISODES):      
            sys.stdout.flush()
            sys.stdout.write('-')
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
        
        while self.env._game.game_stage != GameStage.RIVER and agents[0].state.playing == 1 and agents[1].state.playing == 1:
            if self.mode == 1:
                if self.env._game.game_stage == GameStage.INITIAL:
                    print("Here comes the flop!")
                if self.env._game.game_stage == GameStage.FLOP:
                    print("And the turn...")
                if self.env._game.game_stage == GameStage.TURN:
                    print("It all comes down to this, here's the river.")
                if self.env._game.game_stage == GameStage.RIVER:
                    print("Showdown!")
                time.sleep(1)
                
            for agent in agents:
                best = max(agent.state.rank, best)
            for agent in agents:
                agent.state.hidden = best
                agent.state.calc()
            actions = self.env.step(agents)
            for it, action in enumerate(actions):
                if action[0] == 0 or action[0] == '0' or action[0] == 'fold':
                    output = str(agents[it]) + " folded."
                    agents[it].state.playing = 0
                    if self.mode == 1:
                        print(output)
                    break
                if action[0] == 1 or action[0] == '1' or action[0] == 'check':
                    output = str(agents[it]) + " checks."
                    if self.mode == 1:
                        print(output)
                if action[0] == 2 or action[0] == '2' or action[0] == 'bet':
                    output = str(agents[it]) + " bets."
                    if self.mode == 1:
                        print(output)
            a, pi, mcts_val, nn_val = actions[0]
        
        if self.env._game.game_stage == GameStage.RIVER:
            for agent in agents:
                agent.state.playing = 0
    
        for agent in agents:
            agent.state.rank = Evaluator().evaluate(agent.cards, self.env._game.table_cards)
        score = 0
        if agents[0].state.playing and not agents[1].state.playing:
            score = 1
        if not agents[0].state.playing and agents[1].state.playing:
            score = -1
        if not agents[0].state.playing and not agents[1].state.playing:
            if agents[0].state.rank > agents[1].state.rank:
                score = -1
            if agents[0].state.rank < agents[1].state.rank:
                score = 1
            if agents[0].state.rank == agents[1].state.rank:
                score = 0
        if self.mode == 1:
            print("Agent: ", agents[1].state.rank, Card.print_pretty_cards(agents[1].cards))
            print("User: ", agents[0].state.rank, Card.print_pretty_cards(agents[0].cards))
            try:
                Evaluator().hand_summary(self.env._game.table_cards, [agents[0].cards, agents[1].cards])
            except:
                pass
        self.memory.commit_ltmemory()
        return pi, score



if __name__ == '__main__':
    logging.basicConfig(level=None)
    Tests(1)
