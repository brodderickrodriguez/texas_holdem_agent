import numpy as np

class State:

    def __init__(self, rank, bet, playing, hidden):
        self.rank = rank
        self.playing = playing
        self.bet = bet
        self.hidden = hidden
        self.gs = 0

    def get_model_input(self):
        return [self.rank] + [self.bet] + [self.playing]

    def take_action(self, action):
        self.gs += 1
        if action == 2:
            self.bet += 1
        if action == 0:
            self.playing = 0
        if self.gs >= 4:
            self.playing = 0
            # Value function
            self.value = (-1 if self.rank == self.hidden else 1)
            self.value += (self.value * self.bet)
            return 

    def copy(self):
        copy = State(self.rank, self.bet, self.playing, self.hidden)
        copy.gs = self.gs
        return copy