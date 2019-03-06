#!/usr/local/bin/python3
# Brodderick Rodriguez
# Auburn University - CSSE
# 06 Mar. 2019

import gym
import sys
sys.path.append('/Users/bcr/Dropbox/Projects/CODE/Python/TexasHoldem_RL')
import HoldemEnvironment.holdem
import HoldemEnvironment.holdem.holdem as holdem


def play_out_hand(env, n_seats):
  # reset environment, gather relevant observations
  (player_states, (community_infos, community_cards)) = env.reset()
  (player_infos, player_hands) = zip(*player_states)

  # display the table, cards and all
  env.render(mode='human')

  terminal = False
  while not terminal:
    # play safe actions, check when noone else has raised, call when raised.
    actions = holdem.safe_actions(community_infos, n_seats=n_seats)
    (player_states, (community_infos, community_cards)), rews, terminal, info = env.step(actions)
    env.render(mode='human')

# env = gym.make('TexasHoldem-v1') # holdem.TexasHoldemEnv(2)
env = holdem.TexasHoldemEnv(2, max_limit=1e2, debug=False)

# start with 2 players
env.add_player(0, stack=2000) # add a player to seat 0 with 2000 "chips"
env.add_player(1, stack=2000) # add another player to seat 1 with 2000 "chips"

# play out a hand
play_out_hand(env, env.n_seats)


