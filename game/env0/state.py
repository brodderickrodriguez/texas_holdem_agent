import numpy as np

class State:

    def __init__(self, rank, bet, playing, hidden):
        self.rank = rank
        self.playing = playing
        self.bet = bet
        self.hidden = hidden
        self.value = 0
        self.gs = 0
    
    def id(self):
        return ''.join(str([self.rank] + [self.bet] + [self.playing] + [self.gs]))

    def get_model_input(self):
        return np.reshape(np.array(list(map(int, format(self.rank, '013b'))) + [self.bet] + [self.playing]), (1,3,5))

    def get_bin(self):
        return np.array(list(map(int, format(self.rank, '013b'))) + [self.bet] + [self.playing]).reshape((1,3,5))

    def take_action(self, action):
        newState = self.copy()
        if newState.playing == 1:
            newState.gs += 1
        if action == 2:
            newState.bet += 1
        if action == 0:
            newState.playing = 0
            newState.value = -1
        if newState.gs >= 4:
            newState.playing = 0
            newState.calc()
        return newState, newState.value, (newState.playing + 1) % 2

    def calc(self):
        # Value function
        if self.gs < 4 and self.playing == 0:
            return -1
        self.value = (-1 if self.rank == self.hidden else 1)
        self.value += (self.value * self.bet)
        return self.value


    def copy(self):
        copy = State(self.rank, self.bet, self.playing, self.hidden)
        copy.gs = self.gs
        return copy

    def is_leaf(self):
        return self.playing == 0