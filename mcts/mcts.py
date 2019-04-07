import numpy as np
import config

class Node:

    def __init__(self, state):
        self.state = state
        self.turn  = state.turn
        self.id    = state.id
        self.edges = []

    def is_leaf(self):
        return len(self.edges) == 0


class Edge:

    def __init__(self, a, b, pred, action):
        self.id     = a.state.id + '>' + b.state.id
        self.a      = a
        self.b      = b
        self.action = action
        self.stats  = {'N': 0, 'W': 0, 'Q': 0, 'P': pred}

    def __getitem__(self, index):
        return self.stats[index]


class MCTS:

    def __init__(self, root, cpuct):
        self.root  = root
        self.tree  = {root.id: root}
        self.cpuct = cpuct

    def __len__(self):
        return len(self.tree)

    def back_fill(self, value, breadcrumbs):
        for edge in breadcrumbs:
            edge['N'] += 1
            edge['W'] += value
            edge['Q'] = edge['W'] / edge['N']

    def move_to_leaf(self):
        breadcrumbs = []
        current     = self.root

        done  = False
        value = False

        while not current.is_leaf():
            maxQU = -2**31

            if current == self.root:
                epsilon = config.epsilon
                nu      = np.random.dirichlet([config.alpha] * len(current.edges))
            else:
                epsilon = 0
                nu = [0] * len(current.edges)
            
            Nb = 0
            for i, (action, edge) in enumerate(current.edges):

                U = self.cpuct * ((1 - epsilon) * edge['P'] + epsilon * nu[i]) * Nb//2 / (1 + edge['N'])
                Q = edge['Q']

                if Q + U > maxQU:
                    maxQU      = Q + U
                    sim_action = action
                    sim_edge   = edge

            _, value, done = current.state.move(sim_action)
            breadcrumbs.append(sim_edge)

        return current, value, done, breadcrumbs