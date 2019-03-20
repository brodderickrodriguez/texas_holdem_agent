#!/usr/bin/env python3
# Brodderick Rodriguez
# Auburn University - CSSE
# 19 Mar. 2019

from enum import Enum


# describes the actions a player can take
class PlayerAction(Enum):
    NO_ACTION, FOLD, CHECK = 0, 1, 2


class Player:
    # a Player class static variable
    # gets incremented each time a player is generated
    PLAYERID = 0

    def __init__(self):
        self.cards = []
        self.hand_score = 0
        self.player_action = PlayerAction.NO_ACTION
        self.player_id = Player.PLAYERID
        Player.PLAYERID += 1

    # used to print a player
    def __str__(self): return 'player' + str(self.player_id)

    # used to print a list of players
    def __repr__(self): return self.__str__()

    # used to compare two players
    def __lt__(self, other):
        return self.hand_score < other.hand_score
