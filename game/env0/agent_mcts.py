from texas_holdem import GameStage
from mcts import *
from itertools import islice
from treys import Evaluator, Deck, Card
from mcts_config import *
import numpy as np
import random
from agent import Agent
from state import State
from IPython import display
import pylab as pl
import matplotlib.pyplot as plt

class MCTS_Agent(Agent):

    def __init__(self):
        Agent.__init__(self)
        self.mcts  = None
        self.sims  = sims
        self.cpuct = cpuct
        self.state = State(0, 0, 1, 0)
        self.model = None
        self.eval_ = Evaluator()
        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []

    def eval(self, leaf, value, done, breadcrumbs): 
        if done == 0:
            value, probs = self.get_preds(leaf)
            probs = probs[:3]
            
            for idx, action in enumerate([0, 1, 2]):
                state = leaf.state.take_action(action)[0]
                node = Node(state)
                if state not in self.mcts.tree:
                    self.mcts.add_node(node)
                new_edge = Edge(leaf, node, probs[idx], action)
                leaf.edges.append((action, new_edge))
        return value, breadcrumbs

    def replay(self, memory):
        for _ in range(loops):
            mb = random.sample(memory, min(batch_size, len(memory)))

            training_states = np.array([row['state'].get_bin() for row in mb])
            training_targets = {'value_head' : np.array([row['value'] for row in mb]),
                                'policy_head': np.array([row['AV'] for row in mb])}
            print(training_states.shape, training_targets['value_head'].shape, training_targets['policy_head'].shape)
            fit = self.model.fit(training_states, training_targets, epochs=epochs, verbose=1, validation_split=0, batch_size=32)

            self.train_overall_loss.append(round(fit.history['loss'][epochs - 1],4))
            self.train_value_loss.append(round(fit.history['value_head_loss'][epochs - 1],4)) 
            self.train_policy_loss.append(round(fit.history['policy_head_loss'][epochs - 1],4)) 
        
            
            plt.plot(self.train_overall_loss, 'k')
            plt.plot(self.train_value_loss, 'k:')
            plt.plot(self.train_policy_loss, 'k--')

            plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')

            display.clear_output(wait=True)
            display.display(pl.gcf())
        self.model.print_weights()
    
    def get_preds(self, leaf):
        model_input = self.state.get_model_input()

        preds = self.model.predict(model_input)

        value  = preds[0][0]
        logits = preds[1][0]

        actions = [0, 1, 2] # Fold, Check, Bet

        mask = np.ones(logits.shape, dtype=bool)
        mask[actions] = False
        logits[mask] = -100

        odds = np.exp(logits)
        probs = odds / np.sum(odds)

        return value, probs

    def getAV(self, tau):
        edges = self.mcts.root.edges
        pi = np.zeros(3, dtype=np.integer)
        values = np.zeros(3, dtype=np.float32)
        for action, edge in edges:
            pi[action] = pow(edge.stats['N'], 1/tau)
            values[action] = edge.stats['Q']
        pi = pi / np.sum(pi)
        return pi, values

    def chooseAction(self, pi, values, tau):
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx==1)[0][0] 

        value = values[action]

        return action, value

    def predict(self, input_to_model):
        return self.model.predict(input_to_model)
        
    def sim(self):
        l, v, d, b = self.mcts.move_to_leaf()
        v, b       = self.eval(l, v, d, b)
        self.mcts.back_prop(v, b)

    def update(self, observation, tau):
        if self.mcts is None:
            if observation[0] == GameStage.HAND_COMPLETE:
                print("Whoa! MCTS not initialized")
                return # This shouldn't happen, but if it does, don't)
            self.state.rank    = self.eval_.evaluate(self.cards, observation[1])
            self.state.bet     = 0 
            self.state.playing = 1
            self.state.gs      = 0
            self.mcts = MCTS(Node(self.state), self.cpuct)
        else:
            self.change_root(Node(self.state))
        
        self.state.rank = self.eval_.evaluate(self.cards, observation[1])

        for _ in range(self.sims):
            self.sim()

        pi, values = self.getAV(1)

        action, value = self.chooseAction(pi, values, tau)
    
        nextState = self.state.copy()
        if action == 0:
            nextState.playing = 0
        if action == 2:
            nextState.bet += 1
        nextState.gs += 1
        
        NN_value = -self.get_preds(nextState)[0]

        return action, pi, value, NN_value

    def change_root(self, state):
        self.mcts.add_node(state)
        self.mcts.root = self.mcts.tree[state.id]