#!/usr/bin/env python3
# Brodderick Rodriguez - CSSE
# 19 Mar. 2019

from enum import Enum


class PlayerAction(Enum):
    NO_ACTION, FOLD, CHECK = 0, 1, 2


class Player:
    PLAYERID = 0

    def __init__(self):
        self.cards = []
        self.hand_score = 0
        self.player_action = PlayerAction.NO_ACTION
        self.player_id = Player.PLAYERID
        Player.PLAYERID += 1

    def __str__(self): return 'player' + str(self.player_id)

    def __repr__(self): return self.__str__()

    def __lt__(self, other):
        return self.hand_score < other.hand_score
