from game.env0.texas_holdem import GameStage
from mcts import *
from itertools import islice
from treys import Evaluator, Deck, Card
import mcts_config
import numpy as np
import random
from game.env0.agent import Agent
from state import State

class MCTS_Agent(Agent):

    def __init__(self):
        self.mcts  = None
        self.sims  = mcts_config.sims
        self.cpuct = mcts_config.cpuct
        self.state = State(0, 0, 0, 0)
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
            probs = probs[0, 1, 2]
            
            for idx, action in enumerate([0, 1, 2]):
                if action == 0 or action == 2:
                    if action == 0:
                        leaf.state["playing"] = 0
                    if action == 2:
                        leaf.bet += 1
                state = State(leaf.rank, leaf.bet, leaf.playing, leaf.hidden)
                node = Node(state)
                if state not in self.mcts.tree:
                    self.mcts.add_node(node)
                new_edge = Edge(leaf, node, probs[idx], action)
                leaf.edges.append((action, new_edge))
        return value, breadcrumbs

    def replay(self, memory):
        for _ in range(mcts_config.loops):
            mb = random.sample(memory, min(mcts_config.batch_size, len(memory)))

            training_states = np.array([row['state'].get_model_input() for row in mb])
            training_targets = {'value_head' : np.array([row['value'] for row in mb]),
                                'policy_head': np.array([row['AV'] for row in mb])}
            fit = self.model.fit(training_states, training_targets, epochs=mcts_config.epochs, verbose=1, validation_split=0, batch_size = 32)

            self.train_overall_loss.append(round(fit.history['loss'][mcts_config.epochs - 1],4))
            self.train_value_loss.append(round(fit.history['value_head_loss'][mcts_config.epochs - 1],4)) 
            self.train_policy_loss.append(round(fit.history['policy_head_loss'][mcts_config.epochs - 1],4)) 
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

        pi = pi / (np.sum(pi) * 1.0)
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
                return # This shouldn't happen, but if it does, don't
            self.state.rank    = self.eval_.evaluate(self.cards, observation[1])
            self.state.bet     = 0 
            self.state.playing = 1
            self.mcts = MCTS(self.state, self.cpuct)
        else:
            self.change_root(self.state)
        
        for sim in range(self.sims):
            sim()

        pi, values = self.getAV(1)

        action, value = self.chooseAction(pi, values, tau)
    
        if action == 0:
            nextState = self.state.copy()
            nextState.playing = 0
        if action == 2:
            nextState = self.state.copy()
            nextState.bet += 1
        
        NN_value = -self.get_preds(nextState)[0]

        return action, pi, value, NN_value

    def change_root(self, state):
        self.mcts.root = self.mcts.tree[state]
    