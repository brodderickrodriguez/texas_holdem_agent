import numpy as np
from collections import deque

import mcts_config as config

class Memory:
	def __init__(self, memory_size):
		self.memory_size = config.memory_size
		self.ltmemory = deque(maxlen=config.memory_size)
		self.stmemory = deque(maxlen=config.memory_size)

	def commit_stmemory(self, identities, state, actionValues):
		for r in identities(state, actionValues):
			self.stmemory.append({
				'state': r[0], 
				'AV': r[1], 
				'playerTurn': r[0].playerTurn
				})

	def commit_ltmemory(self):
		for i in self.stmemory:
			self.ltmemory.append(i)
		self.clear_stmemory()

	def clear_stmemory(self):
		self.stmemory = deque(maxlen=config.memory_size)