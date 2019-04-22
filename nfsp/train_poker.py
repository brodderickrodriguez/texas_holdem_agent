import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import numpy as np
from treys import Evaluator, Deck, Card
from nfsp import NFSPAgent
from env0.env import Environment
from env0.agent import Agent
from env0.agent import AgentAction
from env0.texas_holdem import TexasHoldem
from env0.texas_holdem import GameStage

NUM_PLAYERS = 2

class NFSP_Player(Agent):
    def __init__(self):
        super().__init__()
        self.state = {'rank': [0], 'bet': 0, 'playing': 1}
        self.eval = Evaluator()
        self.win = False

    def act(self, bet_amount, nfsp):
        state = self.getState(bet_amount)
        state = torch.tensor([state], device=device, dtype=torch.float)
        action = nfsp.act(state).item()

        if action == 0:
            self.fold()
        if action == 2:
            self.bet()

        self.agent_action = action
        return action

    def fold(self):
        self.state['playing'] = 0

    def bet(self):
        self.state['bet'] += 1

    def getState(self, bet_amount):
        # state = np.array(self.state["rank"] + [self.state["bet"]] + [self.state["playing"]] + [bet_amount])
        state = np.array(self.state["rank"] + [self.state["bet"]] + [self.state["playing"]] + [bet_amount])
        # state = torch.tensor([state], device=device, dtype=torch.float)
        return state
    
    def update(self, observation):
        if isinstance(observation[0], int):
            self.state['rank'] = self.eval.evaluate(self.cards, observation[1])
        else:
            if 'agent' + str(self.agent_id) in observation:
                self.win = True

    def reset(self):
        self.agent_action = 0
        self.cards = []
        self.state['rank'] = [0]
        self.state['bet'] = 0
        self.state['playing'] = 1
        self.win = False
        

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = Environment()
    
    n_inputs = 4
    n_outputs = 3 # 0 = FOLD, 1 = CHECK, 2 = BET
    nfsp = NFSPAgent(n_inputs, n_outputs, save_progress=True, resume=True)

    agents = [NFSP_Player() for _ in range(NUM_PLAYERS)]

    for agent in agents:
        env.add_agent(agent)

    env.setup_game()

    score_history = deque(maxlen=30)
    ep = 0

    while True:
        env._game.reset()
        for agent in agents:
            agent.reset()

        currPlayer = random.randint(0, NUM_PLAYERS - 1)
        steps = 0

        bet_amount = 1
        bet_limit = 3
        lastActions = [2] * NUM_PLAYERS
        state = agents[currPlayer].getState(bet_amount)
        state = torch.tensor([state], device=device, dtype=torch.float)
        action = 2
        agents[currPlayer].bet()
        next_state = agents[currPlayer].getState(bet_amount)
        reward = [0] * NUM_PLAYERS

        action = torch.tensor([[action]], device=device, dtype=torch.long)
        # nfsp.memorizeRL(state, action, reward[currPlayer], next_state)

        currPlayer = 0
        last_player_memory = (state, action, next_state)
        lastActions = [2] * NUM_PLAYERS
        done = False
        num_actions = 0
        num_steps = 0

        while not done:
            num_actions += 1
            if agents[currPlayer].state['playing'] == 1:
                state = agents[currPlayer].getState(bet_amount)
                state = torch.tensor([state], device=device, dtype=torch.float)
                
                # lastActions[currPlayer] = 0
                # if all(a == 0 for a in lastActions):
                #     reward[currPlayer] = bet_amount
                #     action = 1
                #     done == True
                # else:
                action = agents[currPlayer].act(bet_amount, nfsp)
                lastActions[currPlayer]= action

                if action == 2:
                    if bet_limit == 0:
                        agents[currPlayer].state['bet'] -= 1
                        action = 1
                    else:
                        bet_amount += 1
                        bet_limit -= 1
                    lastActions[currPlayer] = action
                if action == 0:
                    reward[currPlayer] = -agents[currPlayer].state['bet']
                    reward[(currPlayer+1)%2] = bet_amount
                    done = True
                else:
                    if all(a < 2 for a in lastActions):
                        for i in range(NUM_PLAYERS):
                            env.step(agents[i], lastActions[i])
                        lastActions = [2] * NUM_PLAYERS
                        bet_limit = 4
                        num_steps += 1

                    if env._game.game_stage == GameStage.SHOWDOWN:
                        for i in range(NUM_PLAYERS):
                            if agents[i].win:
                                reward[i] = bet_amount
                            else:
                                reward[i] = -agents[i].state['bet']
                        done = True
                                    
                next_state = agents[currPlayer].getState(bet_amount)
                action = torch.tensor([[action]], device=device, dtype=torch.long)

                nfsp.memorizeRL(last_player_memory[0], last_player_memory[1], reward[currPlayer], last_player_memory[2])
                last_player_memory = (state, action, next_state)

                if done:
                    nfsp.memorizeRL(last_player_memory[0], last_player_memory[1], reward[currPlayer], last_player_memory[2])

                currPlayer += 1
                currPlayer %= NUM_PLAYERS

        nfsp.learn()
        ep += 1

        if ep % 2000 == 0:
            print("episode:{0}, RL loss: {1}, SL loss: {2}".format(ep, nfsp.lossRL, nfsp.lossSL))