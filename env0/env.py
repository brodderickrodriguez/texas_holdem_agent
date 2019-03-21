#!/usr/bin/env python3
# Brodderick Rodriguez
# Auburn University - CSSE
# 19 Mar. 2019

import logging
from multiprocessing.pool import ThreadPool
from TexasHoldem_ReinforcementLearning.env0.player import Player
from TexasHoldem_ReinforcementLearning.env0.texas_holdem import TexasHoldem


class Agent:
    def __init__(self):
        self.env = None
        self.player = Player()

    def begin(self):
        if self.env is None:
            raise RuntimeError('you must add this agent to an environment first')

        observation = self.env.step(self, "test action")
        print(self.player.player_id, observation)

        observation = self.env.step(self, "test action")
        print(self.player.player_id, observation)

        return observation


class Environment:
    def __init__(self):
        self._agents, self._agent_actions, self._game = [], None, None
        self._called_env_step = False
        self._game_observations = []

    def add_agent(self, agent):
        agent.env = self
        self._agents.append(agent)

    def setup_game(self):
        self._game = TexasHoldem([a.player for a in self._agents])
        self._agent_actions = {i: {} for i in range(6)}

    def step(self, agent, action):
        if self._game is None:
            raise RuntimeError('game has not been created. Must call Environment.setup_game() first.')

        self._agent_actions[self._game.game_stage.value][agent] = action

        # if not all players have committed actions, wait
        while len(self._agent_actions[self._game.game_stage.value]) < len(self._agents):
            continue

        # if we have not yet called game.step() for this game stage, do so
        if not self._called_env_step:
            self._called_env_step = True
            self._game_observations.append(self._game.step(None))
            self._called_env_step = False

        # TODO: issue here!
        # need to wait until the game stages match before returning something
        # is also doesnt work all the time :(
        # while len(self._game_observations) == 0:
        #     continue

        return self._game_observations[-1:]


class Tests:
    def __init__(self):

        thread_pool = ThreadPool(processes=2)
        env = Environment()

        agent1 = Agent()
        agent2 = Agent()

        env.add_agent(agent1)
        env.add_agent(agent2)

        env.setup_game()

        # do some multi threading
        t1 = thread_pool.apply_async(agent1.begin)
        t2 = thread_pool.apply_async(agent2.begin)
        t1.get()
        t2.get()


if __name__ == '__main__':
    logging.basicConfig(level=None)
    Tests()
