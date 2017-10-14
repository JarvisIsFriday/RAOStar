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
		self.env = np.zeros([3, 3])
		self.env[2, 0] = 1
		self.env[2, 2] = 1
		self.fire = (2, 1)  # fire located at self.env[2,1]: terminal
		self.goal = (1, 2)  # goal position
		self.optimization = 'minimize'  # want to minimize the steps to goal
		self.action_map = {
			"right": 5,
			"left": 6,
			"up": 7,
			"down": 8
		}

	def print_model(self):
		height, width = self.env.shape
		print(" ")
		print("    ** Model environment **")
		for j in range(height):
			# print("row: " + str(j))
			row_str = "   "
			for i in range(width):
				if self.goal == (j, i):
					row_str += " [goal] "
				elif self.fire == (j, i):
					row_str += " [fire] "
				elif self.env[j][i]:
					row_str += " [----] "
				else:
					row_str += " [" + str(j) + ", " + str(i) + "] "
			print(row_str)

	def print_policy(self, policy):
		height, width = self.env.shape
		policy_map = np.zeros([height, width])

		for key in policy:
			coords = key.split(":")[0].split("(")[1].split(")")[0]
			col = int(coords.split(",")[0])
			row = int(coords.split(",")[1])
			action_string = policy[key]
			for action_name in self.action_map:
				if action_name in action_string:
					policy_map[col][row] = self.action_map[action_name]
					break
		print(" ")
		print("         ** Policy **")
		for j in range(height):
			# print("row: " + str(j))
			row_str = "   "
			for i in range(width):
				if self.goal == (j, i):
					row_str += " [goal] "
				elif self.fire == (j, i):
					row_str += " [fire] "
				elif self.env[j][i]:
					row_str += " [----] "
				else:
					if policy_map[j][i] == 5:
						row_str += " [ -> ] "
					if policy_map[j][i] == 6:
						row_str += " [ <- ] "
					if policy_map[j][i] == 7:
						row_str += " [ ^^ ] "
					if policy_map[j][i] == 8:
						row_str += " [ vv ] "
					if policy_map[j][i] == 0:
						row_str += " [    ] "
			print(row_str)

	def state_valid(self, state):  # check if a particular state is valid
		if state[0] < 0 or state[1] < 0:
			return False
		try:
			return self.env[state[0], state[1]] == 0
		except IndexError:
			return False

	def actions(self, state):
		validActions = []
		for act in [(1, 0, "down"), (0, 1, "right"), (-1, 0, "up"), (0, -1, "left")]:
			newx = state[0] + act[0]
			newy = state[1] + act[1]
			if self.state_valid((newx, newy)):
				validActions.append(act)
		if state == self.goal:
			return []
		return validActions

	def is_terminal(self, state):
		return state == self.goal  # not sure what this does #mdeyo: is this comment from Yun?

	def state_transitions(self, state, action):
		newstates = []
		intended_new_state = (state[0] + action[0], state[1] + action[1])
		if not self.state_valid(intended_new_state):
			return newstates
		if state in self.icy_blocks:
			newstates.append((intended_new_state, self.icy_move_forward_prob))
			for slip in [-1, 1]:
				slipped = [slip if action[i] ==0 else action[i] for i in range(2)]
				slipped_state = (state[0] + slipped[0], state[1] + slipped[1])
				if self.state_valid(slipped_state):
					newstates.append(
						(slipped_state, (1 - self.icy_move_forward_prob) / 2))
		else:
			newstates.append((intended_new_state, 1.0))
		return newstates

	def observations(self, state):
		if self.DetermObs:
			return [(state, 1.0)]
		else:  # robot only knows if it is on icy or non icy block
			if state in self.icy_blocks:
				return [(self.icy_blocks[i], 1 / len(self.icy_blocks)) for i in range(len(self.icy_blocks))]
			else:
				prob = 1 / (6 - len(self.icy_blocks))
				dist = []
				for i in range(3):
					for j in range(3):
						if self.env[i, j] == 0 and (i, j) not in self.icy_blocks:
							dist.append((i, j), prob)
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

