#!/usr/bin/env python3
# Brodderick Rodriguez
# Auburn University - CSSE
# 19 Mar. 2019

from enum import Enum


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
    def __str__(self): return 'agent' + str(self.agent_id)

    # used to print a list of agent
    def __repr__(self): return self.__str__()

    # used to compare two agents
    def __lt__(self, other):
        return self.hand_score < other.hand_score

    # called when the environment changes
    def update(self, observation):
        print('got update', observation)


