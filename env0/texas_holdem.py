#!/usr/bin/env python3
# Brodderick Rodriguez
# Auburn University - CSSE
# 19 Mar. 2019

from agent import Agent
from enum import Enum
from treys import Evaluator, Deck, Card
import logging


# describes the current stage of the Texas Hold'em game
class GameStage(Enum):
    INITIAL, FLOP, TURN, RIVER, SHOWDOWN, HAND_COMPLETE = 0, 1, 2, 3, 4, 5

    # simply increments the game stage to the next stage
    def increment(self):
        return GameStage(self.value + 1)


class TexasHoldem:
    def __init__(self, agents):
        # if len(agents) < 2:
        #     raise RuntimeError('number of agents must be at least two')

        # initialize variables and assign their respective values in reset()
        self.deck, self.table_cards, self.agents, self.game_stage = [], [], agents, GameStage.INITIAL
        self.reset()

    def reset(self):
        self.deck = Deck()
        self.table_cards = []
        self.game_stage = GameStage.INITIAL

        logging.info('dealing initial cards')
        for agent in self.agents:
            agent.cards = self.deck.draw(n=2)
            logging.info(str(agent) + Card.print_pretty_cards(agent.cards))

    def step(self):
        # check if hand is complete
        if self.game_stage == GameStage.HAND_COMPLETE:
            return self.game_stage

        # if not, increment the game's stage
        self.game_stage = self.game_stage.increment()

        # log some helpful output
        logging.info(self.game_stage)

        # if game stage is flop, put three cards on the table
        if self.game_stage == GameStage.FLOP:
            self.table_cards = self.deck.draw(n=3)

        # if game stage is turn or river, add another card to the table
        elif self.game_stage in [GameStage.TURN, GameStage.RIVER]:
            self.table_cards.append(self.deck.draw(n=1))

        # if all cards for this hand have been delt, evaluate each agents hands
        # and return a list of winning agents
        elif self.game_stage == GameStage.SHOWDOWN:
            return self.evaluate_hands()

        # if we accidentally call TexasHoldem.step() after the hand is over,
        # this will catch it
        elif self.game_stage == GameStage.HAND_COMPLETE:
            return self.game_stage

        # log some helpful output
        logging.info(Card.print_pretty_cards(self.table_cards))

        # lastly, return the game stage
        return self.game_stage, self.table_cards

    def evaluate_hands(self):
        evaluator = Evaluator()

        # for each agent, evaluate their hand and store its value
        # in agent.hand_score
        for agent in self.agents:
            agent.hand_score = evaluator.evaluate(agent.cards, self.table_cards)
            print(agent, agent.hand_score)

        # sorts agents in descending over respective to their hand_score
        # see Agent.__lt__()
        self.agents.sort(reverse=True)
        winners = [self.agents[0]]
        idx = 0

        # check if there are ties for first place
        while idx + 1 < len(self.agents) and self.agents[idx].hand_score == self.agents[idx + 1].hand_score:
            winners.append(self.agents[idx + 1])
            idx += 1

        return winners


# a test class to make sure everything in this file works
class Tests:
    def __init__(self):
        pass

    @staticmethod
    def game_stage_test():
        agents = [Agent() for _ in range(3)]
        game = TexasHoldem(agents)

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
