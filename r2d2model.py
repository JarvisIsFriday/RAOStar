#!/usr/bin/env python 

# author: Yun Chang 
# email: yunchang@mit.edu 
# r2d2 model as simple model to test rao star 

import numpy as np 

class R2D2Model(object): 
	# noted there are 7 blocks total, A through G 
	# G is the goal, F is fire 
	def __init__(self, icy_blocks, DetermObs=True):
		# icy blocks are defined blocks that are icy 
		self.icy_blocks = icy_blocks
		self.icy_move_forward_prob = 0.8 
		self.DetermObs = DetermObs 
		# if observation deterministic or not. If determinstic, once move, 
		# no excatly where it is. If not, P(o_k+1|s_k+1) = 0.6 for the cell 
		# it actually is in and 0.1 for the neighborinf cells 
		if not self.DetermObs:
			self.obsProb = 0.6
		# environment will be represented as a 3 x 3 grid, with (2,0) and  (2,2) blocked 
		# note python indexing
		self.env = np.zeros([3,3])
		self.env[2,0] = 1
		self.env[2,2] = 1
		self.fire = (2,1) # fire located at self.env[2,1]: terminal 
		self.goal = (1,2) # goal position 
		self.optimization = 'minimize' # want to minimize the steps to goal 
		self.actual_state = (1,0)

	def state_valid(self, state):
		try: 
			return self.env[state[0], state[1]] == 0
		except IndexError:
			return False

	def actions(self, state):
		validActions = []
		for act in [(1,0), (0,1), (-1,0), (0,-1)]:
			newx = state[0] + act[0]
			newy = state[1] + act[1]
			if self.state_valid((newx, newy)):
				validActions.append(act)
		return validActions

	def is_terminal(self, state):
		return state == self.fire

	def state_transitions(self, state, action):
		if state in self.icy_blocks:
			intended_new_state = (state[0]+action[0], state[1]+action[1])
			self.actual_state = intended_new_state
			newstates = [(intended_new_state,self.icy_move_forward_prob)]
			for slip in [-1, 1]:
				slipped = [(action[i]+slip)%2 for i in range(2)]
				slipped_state = (state[0]+slipped[0], state[1]+slipped[1])
				if self.state_valid(slipped_state):
					newstates.append((slipped_state,(1-self.icy_move_forward_prob)/2))
			return newstates
		newstates = [((state[0]+action[0], state[1]+action[1]),1.0)]
		self.actual_state = (state[0]+action[0], state[1]+action[1])
		return newstates

	def observations(self, state):
		if self.DetermObs:
			return [(self.actual_state, 1.0)]
		else: # robot only knows if it is on icy or non icy block 
			if self.actual_state in self.icy_blocks:
				return [(self.icy_blocks[i], 1/len(self.icy_blocks)) for i in range(len(self.icy_blocks))]
			else: 
				prob = 1/(6-len(self.icy_blocks))
				dist = []
				for i in range(3):
					for j in range(3):
						if self.env[i,j] == 0 and (i,j) not in self.icy_blocks:
							dist.append((i,j),prob)
				return dist 

	def state_risk(self, state):
		if state == self.fire:
			return 1.0
		return 0.0

	def heuristic(self, state): 
		# square of euclidean distance as heuristic 
		return sum([(self.goal[i] - state[i])**2 for i in range(2)])

	def execution_risk_heuristic(self, state):
		# sqaure of euclidean distance to fire as heuristic 
		return sum([(self.fire[i] - state[i])**2 for i in range(2)])

