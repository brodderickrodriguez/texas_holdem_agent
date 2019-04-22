import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import deque
import numpy as np
import copy

# hyper parameters
EPISODES = 50000  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
ETA = .1  # anticipatory parameter
GAMMA = 0.8  # Q-learning discount factor
LR_RL = 0.1  # best action NN optimizer learning rate
LR_SL = 0.005  # average action NN optimizer learning rate
BATCH_SIZE = 128  # Q-learning batch size
UPDATE_PERIOD = 100  # target update period 

RL_MEMORY_SIZE = 50000
SL_MEMORY_SIZE = 200000

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NFSPAgent:
    def __init__(self, n_inputs, n_outputs, resume=False, save_progress=False):
        # Best action policy network
        self.model = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, n_outputs)
        ).to(device)
        # Target network
        self.targetModel = copy.deepcopy(self.model)
        # Average action policy network
        self.avgModel = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, n_outputs)
        ).to(device)

        # Replay memories
        self.memoryRL = deque(maxlen=RL_MEMORY_SIZE)
        self.memorySL = [] # reservoir sampling
        self.memorizeAvgAction = False

        self.n_outputs = n_outputs

        self.optimizer = optim.Adam(self.model.parameters(), LR_RL)
        self.avgOptimizer = optim.Adam(self.avgModel.parameters(), LR_SL)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.steps_done = 0
        self.updates = 0
        self.saves = 0

        self.lossRL = 0
        self.lossSL = 0

        self.save_progress = save_progress

        if resume:
            self.loadAll()
    
    def act(self, state):
        self.steps_done += 1
        if random.random() > ETA:
            return self.avgModel(state).data.max(1)[1].view(1, 1)
        else:
            self.memorizeAvgAction = True
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
            if random.random() > eps_threshold:
                return self.model(state).data.max(1)[1].view(1, 1)
            else:
                return torch.tensor([[random.randrange(self.n_outputs)]], device=device, dtype=torch.long)

    def memorizeRL(self, state, action, reward, next_state):
        self.memoryRL.append((state,
                            action,
                            torch.tensor([reward], device=device, dtype=torch.float),
                            torch.tensor([next_state], device=device, dtype=torch.float)))
        if self.memorizeAvgAction:
                self.memorizeSL(state, action)

    def memorizeSL(self, state, action):
        if len(self.memorySL) == SL_MEMORY_SIZE:
            i = random.randint(0, SL_MEMORY_SIZE-1)
            self.memorySL[i] = (state, action)
        else:
            self.memorySL.append((state,
                                action))
        self.memorizeAvgAction = False

    def updateTarget(self):
        self.targetModel = copy.deepcopy(self.model)
        
        if self.save_progress:
            self.saves += 1
            self.saveAll()
            self.saveAvgModel(self.saves)
    
    def learn(self):

        self.learnAvg()
        """Experience Replay"""
        if len(self.memoryRL) < BATCH_SIZE:
            return
        if self.steps_done % BATCH_SIZE != 0:
            return
        batch = random.sample(self.memoryRL, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)

        current_q = self.model(states).gather(1, actions)
        max_next_q = self.targetModel(next_states).detach().max(1)[0]
        expected_q = rewards + (GAMMA * max_next_q)
        
        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.lossRL = loss

        self.updates += 1
        if self.updates % UPDATE_PERIOD == 0:
            self.updateTarget()

    def learnAvg(self):
        """Average play learning"""
        if len(self.memorySL) < BATCH_SIZE:
            return
        if self.steps_done % BATCH_SIZE != 0:
            return
        batch = random.sample(self.memorySL, BATCH_SIZE)
        states, actions = zip(*batch)

        states = torch.cat(states)
        actions = torch.cat(actions)

        predictions = self.avgModel(states)
        
        loss = self.criterion(predictions, actions.view(-1))
        self.avgOptimizer.zero_grad()
        loss.backward()
        self.avgOptimizer.step()

        self.lossSL = loss

    def saveAvgModel(self, i):
        torch.save(self.avgModel, './TrainedModel/model' + str(i).zfill(5))

    def saveAll(self):
        state = {
            #'memoryRL': self.memoryRL,
            #'memorySL': self.memorySL,
            'model_state_dict': self.model.state_dict(),
            'avgModel_state_dict': self.avgModel.state_dict(),
            'targetModel_state_dict': self.targetModel.state_dict(),
            'model_optimizer': self.optimizer.state_dict(),
            'avgModel_optimizer': self.avgOptimizer.state_dict()
        }
        torch.save(state, './TrainedModel/current_model')

    def loadAll(self):
        state = torch.load('./TrainedModel/current_model')
        #self.memoryRL = state['memoryRL']
        #self.memorySL = state['memorySL']
        self.model.load_state_dict(state['model_state_dict'])
        self.avgModel.load_state_dict(state['avgModel_state_dict'])
        self.targetModel.load_state_dict(state['targetModel_state_dict'])
        self.optimizer.load_state_dict(state['model_optimizer'])
        self.avgOptimizer.load_state_dict(state['avgModel_optimizer'])


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    n_inputs = 4
    n_outputs = 2
    agent = NFSPAgent(n_inputs, n_outputs, save_progress=True)
    score_history = deque(maxlen=30)

    for e in range(1, EPISODES+1):
        state = env.reset()
        steps = 0
        while True:
            #env.render()
            state = torch.tensor([state], device=device, dtype=torch.float)
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action.item())

            # negative reward when attempt ends
            if done:
                reward = -1

            agent.memorizeRL(state, action, reward, next_state)
            agent.learn()

            # print(state)
            # print(action)
            # print(reward)
            # print(next_state)

            state = next_state
            steps += 1

            if done:
                print("episode:{0} score: {1} memory: {2}".format(e, steps, len(agent.memorySL)))
                score_history.append(steps)
                break
        
        if np.mean(score_history) > 190:
            break

    env.close()