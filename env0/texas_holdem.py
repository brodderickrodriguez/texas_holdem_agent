#!/usr/bin/env python3
# Brodderick Rodriguez - CSSE
# 19 Mar. 2019

from env0.player import Player
from enum import Enum
from treys import Evaluator, Deck, Card
import logging


class GameStage(Enum):
    INITIAL, FLOP, TURN, RIVER, SHOWDOWN, HAND_COMPLETE = 0, 1, 2, 3, 4, 5

    def increment(self):
        return GameStage(self.value + 1)


class TexasHoldem:
    def __init__(self, players):
        self.deck = Deck()
        self.table_cards = []
        self.players = players
        self.game_stage = GameStage.INITIAL

    def step(self):
        logging.info(self.game_stage)

        if len(self.players) < 2:
            raise RuntimeError('number of players must be at least two')

        if self.game_stage == GameStage.HAND_COMPLETE:
            return self.game_stage

        if self.game_stage == GameStage.INITIAL:
            for player in self.players:
                player.cards = self.deck.draw(n=2)
                logging.info(str(player) + Card.print_pretty_cards(player.cards))

        elif self.game_stage == GameStage.FLOP:
            self.table_cards = self.deck.draw(n=3)

        elif self.game_stage in [GameStage.TURN, GameStage.RIVER]:
            self.table_cards.append(self.deck.draw(n=1))

        elif self.game_stage == GameStage.SHOWDOWN:
            return self.evaluate_hands()

        logging.info(Card.print_pretty_cards(self.table_cards))

        self.game_stage = self.game_stage.increment()
        return self.game_stage

    def evaluate_hands(self):
        evaluator = Evaluator()

        for player in self.players:
            player.hand_score = evaluator.evaluate(player.cards, self.table_cards)
            print(player, player.hand_score)

        self.players.sort(reverse=True)
        winners = [self.players[0]]
        idx = 0

        # check if there are ties for first place
        while idx + 1 < len(self.players) and self.players[idx].hand_score == self.players[idx + 1].hand_score:
            winners.append(self.players[idx + 1])
            idx += 1

        return winners


class Tests:
    def __init__(self):
        pass

    @staticmethod
    def game_stage_test():
        players = [Player() for _ in range(3)]
        game = TexasHoldem(players)

        game.step()  # initial
        game.step()  # flop
        game.step()  # turn
        game.step()  # river
        game.step()  # showdown
        # game.step()  # game over
        # game.step()  # test -- this shouldn't cause a runtime error


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Tests.game_stage_test()
