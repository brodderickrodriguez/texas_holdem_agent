#!/usr/bin/env python3
# Brodderick Rodriguez
# Auburn University - CSSE
# 19 Mar. 2019

from enum import Enum
from state import State
from treys import Card

# describes the actions an agent can take
class AgentAction(Enum):
    NO_ACTION, FOLD, CHECK = 0, 1, 2


class Agent:
    # an Agent class static variable
    # gets incremented each time an Agent is generated
    AGENTID = 0

    def __init__(self):
        self.env = None
        self.cards = []
        self.hand_score = 0
        self.agent_action = AgentAction.NO_ACTION
        self.agent_id = Agent.AGENTID
        Agent.AGENTID += 1

    # used to print an agent
    def __str__(self): return 'Agent ' + str(self.agent_id)

    # used to print a list of agent
    def __repr__(self): return self.__str__()

    # used to compare two agents
    def __lt__(self, other):
        return self.hand_score < other.hand_score

    # called when the environment changes
    def update(self, observation):
        print('got update', observation)
        print(observation)

class User(Agent):
    # an Agent class static variable
    # gets incremented each time an Agent is generated
    AGENTID = 0

    def __init__(self):
        Agent.__init__(self)
        self.state = State(0,0,1,0)

    # used to print an agent
    def __str__(self): return 'User'

    # used to print a list of agent
    def __repr__(self): return self.__str__()

    # used to compare two agents
    def __lt__(self, other):
        return self.hand_score < other.hand_score

    # called when the environment changes
    def update(self, observation, _):
        print("Your cards are: ", Card.print_pretty_cards(self.cards))
        print("The table has cards: ", Card.print_pretty_cards(observation[1]))
        print("Choose an action: [0 = fold, 1 = check, 2 = bet]")
        while True:
            action = str(input())
            if action in ['0','1','2'] or action.lower() in ['fold', 'check', 'bet']:
                if action == '0' or action == 'fold':
                    self.state.playing == 0
                return action, None, None, None


