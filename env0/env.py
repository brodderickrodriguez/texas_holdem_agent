#!/usr/bin/env python3
# Brodderick Rodriguez
# Auburn University - CSSE
# 19 Mar. 2019

import logging
from TexasHoldem_ReinforcementLearning.env0.agent import Agent
from TexasHoldem_ReinforcementLearning.env0.texas_holdem import TexasHoldem


class Environment:
    def __init__(self):
        self._agents, self._agent_actions, self._game = [], {}, None
        self._called_env_step = False

    def add_agent(self, agent):
        agent.env = self
        self._agents.append(agent)

    def setup_game(self):
        self._game = TexasHoldem(self._agents)
        self._agent_actions = {}

    def step(self, agent, action):
        if self._game is None:
            raise RuntimeError('game has not been created. Must call Environment.setup_game() first.')

        self._agent_actions[agent] = action

        # if all players have committed actions, perform an env update and inform agents
        if len(self._agent_actions) == len(self._agents):
            observation = self._game.step(self._agent_actions)
            [agent.update(observation) for agent in self._agents]
            self._agent_actions = {}

        return


class Tests:
    def __init__(self):

        env = Environment()

        agent1 = Agent()
        agent2 = Agent()

        env.add_agent(agent1)
        env.add_agent(agent2)

        env.setup_game()

        # before flop

        env.step(agent1, 'test action')
        env.step(agent2, 'test action')

        # after flop
        # before turn

        env.step(agent1, 'test action')
        env.step(agent2, 'test action')

        # after flop
        # before river

        env.step(agent1, 'test action')
        env.step(agent2, 'test action')

        # after river

        env.step(agent1, 'test action')
        env.step(agent2, 'test action')




if __name__ == '__main__':
    logging.basicConfig(level=None)
    Tests()
