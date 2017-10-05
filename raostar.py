#!/usr/bin/env python

# author: Yun Chang 
# email: yunchang@mit.edu
# re-implementation of Pedro Santana's RAO* algorithm 
# RAO*: a risk-aware planner for POMDP's 
# Forward, heuristic planner for partially-observable 
# chance-constrained domains 

import operator 
import numpy as np 
import time 

from raostarhypergraph import RAOStarGraphNode, RAOStarGraphOperator, RAOStarHyperGraph
from belief import BeliefState,avg_func,bound_prob
from belief import predict_belief,update_belief,compute_observation_distribution,is_terminal_belief

class RAOStar(object): 
	# find optimal policy and/or tree representing partially 
	# observable domains 

	def __init__(self, model, node_name='id', cc=0.0, cc_type='overall',
		terminal_prob=1.0, randomization=0.0, halt_on_violation=False):

		self.model = model
		self.cc = cc
		self.cc_type = cc_type 
		# type of chance constraint could be "overall": execution risk 
		# bound at the root (constraints overall execution); "everywhere":
		# bounds the execution risk at every policy node. 
		self.terminal_prob = terminal_prob
		self.graph = None # graph construction 
		self.opennodes = set() # set of nodes to be explored 
		self.policy_ancestors = {} # ancestors set for updating policy 
		self.halt_on_violation = halt_on_violation 
		# whether constraint violations is terminal 

		# execution risk cap 
		if self.cc_type.lower() in ['overall', 'o']:
			self.er_cap = 1.0 
		elif self.cc_type.lower() in ['everywhere', 'e']:
			self.er_cap = self.cc
		else: 
			raise TypeError('Choose either overall or everywhere for constraint type')

		# choose the comparison function depending on the type of search 
		if model.optimization == 'maximize':
			self.is_better = operator.gt
			self.is_worse = operator.lt
			self.initial_Q = -np.inf
			self.select_best = lambda n_list: max(n_list, key=lambda node: node.value)
		elif model.optimization == 'minimize':
			self.is_better = operator.lt
			self.is_worse = operator.gt
			self.initial_Q = np.inf
			self.select_best = lambda n_list: min(n_list, key=lambda node: node.value)
		else: 
			raise TypeError('unable to recognize optimization')

		self.start_time = 0.0

		# functions from model (Model function aliases)
		self.A = model.actions 
		self.T = model.state_transitions
		self.O = model.observations
		self.h = model.heuristic
		self.r = model.state_risk 
		self.e = model.execution_risk_heuristic
		self.term = model.is_terminal
		self.node_name = node_name

	def search(self, b0, time_limit=np.inf, iter_limit=np.inf):
		self.start_time = time.time()
		self.init_search(b0)
		count = 0 
		root = self.graph.root
		# initial objective at the root, which is the best possible
		# (can only degrade with admissible heuristic) 
		prev_root_val = np.inf if (self.model.optimization == 'maximize') else -np.inf
		interrupted = False
		while len(self.opennodes) > 0 and (count <= iter_limit) and (time.time()-self.start_time <= time_limit):
			count += 1
			expanded_nodes = self.expand_best_partial_solution()
			self.update_values_and_best_actions(expanded_nodes)
			# best actions aka policy 

			# Updates the mapping of ancestors on the best policy graph and the list of open nodes 
			self.update_policy_open_nodes()

			root_value = root.value
			# root node changed from its best value 
			if not np.isclose(root_value, prev_root_value):
				# if the heuristic is admissible, the root value can only degrade 
				if self.is_better(root_value, prev_root_value):
					print('Warning: root value improved. Check admissibility')
				else:
					prev_root_value = root_value
		policy = self.extract_policy()
		return policy, self.graph

	def init_search(self, b0):
		# initializes the search fields (initialize graph with start node)
		self.graph = RAOStarHyperGraph()
		start_node = RAOStarGraphNode(value=None, state=BeliefState(b0)) # if self.node_name == 'id'
		if self.node_name == 'state':
			start_node = RAOStarGraphNode(name=str(b0), value=None, state=BeliefState(b0))
		self.set_new_node(start_node, 0, self.cc)
		self.graph.add_node(start_node)
		self.graph.root = start_node 
		self.update_policy_open_nodes()

	def set_new_node(self, node, depth, er_bound):
		# sets the fields of a terminal node 
		b = node.state.belief 
		# Depth of a node is its dist to the root 
		node.depth = depth 
		# Probability of violating constraints in a belief state. (never change) 
		node.risk = bound_prob(avg_func(b, self.r))
		if is_terminal_belief(b, self.term, self.terminal_prob):
			self.set_terminal_node(node)
		else: 
			# the value of a node is the average of the heuristic only when it's 
			# first created. After that, the value is given as a function of its children
			node.value = avg_func(b, self.h)
			node.terminal = False # new node is non terminal 
			node.best_action = None # no action associated yet 
			node.exec_risk_bound = bound_prob(er_bound) # execution risk bound 
			node.exec_risk = node.risk # avg heuristic estimate of execution risk at node 

	def set_terminal_node(self, node): 
		# set fields of a terminal node 
		b = node.state.belief 
		node.terminal = True
		node.value = avg_func(b, self.h)
		node.exec_risk = node.risk
		node.best_action = None
		self.graph.remove_all_edges(node) # terminal node has no edges 

	def update_policy_open_nodes(self):
		# traverse graph starting at root along marked actions, recording ancestors
		# and open nodes 
		queue = deque([self.graph.root]) # starts at root and expands nodes with policy and 
		# add nodes with no policy yet to opennodes 
		# policy ancestors={}
		self.opennodes = set() 
		while len(queue) > 0:
			node = queue.popleft() 
			if node.best_action != None: # node already has a best action 
				children = self.graph.hyperedge_successors(node, node.best_action)
				queue.extend(children)
			else: # no best action has been assigned yet 
				if not node.terminal:
					self.opennodes.add(node)

	def _get_all_actions(self, belief, A):
		if len(belief)>0:
			all_node_actions = []
			action_ids=set() #Uses str(a) as ID

			for particle_key, particle_tuple in belief.items():
				new_actions = [a for a in A(particle_tuple[0]) if not str(a) in action_ids]
				all_node_actions.extend(new_actions) # add action and make sure no overlap 
				action_ids.update([str(a) for a in new_actions]) 
			return all_node_actions
		else:
			return []

	def expand_best_partial_solution(self):
		# expands a node in the graph currently contained in the best 
		# partial solution. Add new nodes and edges on the graph 
		nodes_to_expand = self.opennodes
		self.opennodes = None 
		for node in nodes_to_expand:
			belief = node.state.belief # belief state associated to the node 
			parent_risk = node.risk # execution risk for current node 
			parent_bound = node.exec_risk_bound  # er bound for current node 
			parent_depth = node.depth # dist of parent to root 

			# if the current node is guaranteed to violate constraints and a violation
			# is set to halt process: make node terminal 
			if self.halt_on_violation and np.isclose(parent_risk, 1.0):
				all_node_actions = []
			else: 
				all_node_actions = self.get_all_actions(belief, self.A)
			action_added = False # flag if a new action has been added 
			if len(all_node_actions) > 0:
				added_count = 0
				for act in all_node_actions:
					# expand the node into the child nodes generated by action 
					child_obj_list, prob_list, prob_safe_list, new_child_idxs, pretty_obs_list = \
																self.obtain_child_objs_and_probs(belief, \
																	self.T, self.O, self.r, act)
					# initializes the new child nodes 
					for c_idx in new_child_idxs:
						self.set_new_node(child_obj_list[c_idx],parent_depth+1, 0.0)

					# if parent bound Delta is ~ 1.0, the child nodes are guaranteed to have 
					# their risk bound equal to 1 
					if (not np.isclose(parent_bound, 1.0)):
						# computes execution risk bounds for the child nodes
						er_bounds, er_bound_infeasible = self.compute_exec_risk_bounds(parent_bound,\
																parent_risk, child_obj_list, prob_safe_list) 
					else: 
						er_bounds = [1.0]*len(child_obj_list)
						er_bound_infeasible = False

					# Only creates new operator if all er bounds a non-negative 
					if not er_bound_infeasible:
						# updates the values of the execution risk for all children 
						# that will be added to the graph 
						for idx, child in enumerate(child_obj_list):
							child.exec_risk_bound = er_bounds[idx]
						# average instantaneous value (cost or reward) 
						avg_op_value = avg_func(belief, self.V, act)
						act_obj = RAOStarGraphOperator(name=str(act), op_value=avg_op_value, \
											properties={'prob':prob_list, 'prob_safe':prob_safe_list, \
											'obs':pretty_obs_list}) # an "Action" object crerated 
						# add edge (Action) to graph 
						self.graph.add_hyperedge(parent_obj=node, child_obj_list=child_obj_list, op_obj=act_obj)
						action_added = True 
						added_count += 1 
			if not action_added:
				self.set_terminal_node(node)
		return nodes_to_expand # returns the list of node either added actions to or marked terminal 

	def update_values_and_best_actions(self, expanded_nodes):
		# updates the Q values on nodes on the graph and the current best policy 
		# for each expanded node at a time 
		for exp_idx, exp_node in enumerate(expanded_nodes):
			Z = self.build_ancestor_list(exp_node)
			# updates the best action at the node 
			for node in Z:
				# all actions available at that node 
				all_action_operators = [] if node.terminal else self.graph.all_node_operators(node)
				# get all actions (edges) of node from graph 
				# risk at the node's belief state (does no depend on the action taken)
				risk = node.risk 
				# current *admissible* (optimistic) estimate of the node's Q value 
				current_Q = node.value
				# execution risk bound. the execution risk cap depends on type of chance 
				# constraint being imposed 
				er_bound = min([node.exec_risk_bound, self.er_cap])

				best_action_idx = -1 
				best_Q = self.initial_Q # -inf or inf based on optimization 
				best_D = -1 # depth 
				exec_risk_for_best = -1.0 

				# Estimates value and risk of the current node for each possible action 
				for act_idx, act in enumerate(all_action_operators):
					probs = act.properties['prob']
					prob_safe = act.properties['prob_safe']
					children = self.graph.children(node, act)

					# estimate Q of taking this action from current node. Composed of 
					# current reward and the average reward of its children 
					Q = act.op_value + np.sum([p*child.value for (p, child) in zip(probs, children)])
					#Average child depth 
					D = 1 + np.sum([p*child.depth for (p, child) in zip(probs, children)])

					# compute an estimate of the er of taking this action from current node. 
					# composed of the current risk and the avg execution risk of its children 
					exec_risk = risk + (1.0 - risk)*np.sum([p*child.exec_risk for (p,child) in zip(probs_safe, children)])

					# if execution risk bound has been violated or if Q value for this action is worse 
					# than current best, we should definitely no select itr. 
					if (exec_risk > er_bound) or self.is_worse(Q, best_Q):
						select_action = False 
					# if risk bound respected and Q value is equal or better 
					else:
						if np.isclose(Q, best_Q) and self.enforce_dfs:
							# select the action if current node is deepter
							select_action = D > best_D
						else:
							select_action = True 
					# Test if the risk bound for the current node has been violated 
					if select_action:
						# Updates the execution risk bounds for the children 
						child_er_bounds, cc_infeasible = self.compute_exec_risk_bounds(er_bound, risk, children, probs_safe)
						if not cc_infeasible: # if chance constraint has not been violated 
							for idx, child in enumerate(children):
								child.exec_risk_bound = child_er_bounds[idx] 

							# Updates the best action at node 
							best_Q = Q
							best_action_idx = act_idx
							best_D = D 
							exec_risk_for_best = exec_risk

				# Test if some action has been selected 
				if best_action_idx >= 0:
					if (not np.isclsoe(best_Q, current_Q)) and self.is_better(best_Q, current_Q):
						print('WARNING: node Q value improved, which might indicate inadmissibility.')

					# updates optimal value est, execution tisk est, and mark best action 
					node.value = best_Q
					node.exec_risk = exec_risk_for_best
					node.best_action = all_action_operators[best_action_idx]
				else: # no action was selected, so this node is terminal 
					if not node.terminal: 
						self.set_terminal_node(node) 

	def compute_exec_risk_bounds(self, parent_bound, parent_risk, child_list, prob_safe_list, is_terminal_node=False):
		# computes the execution risk bounds for each sibling in a list of children of a node 
		exec_risk_bounds = [0.0]*len(child_list)
		# If the parent bound is almost one, the risk of the children are guaranteed to be feasible 
		if np.isclose(parent_bound, 1.0):
			exec_risk_bound = [1.0]*len(child_list)
			infeasible = False
		else:
			# if parent bound isn't one, but risk is almost one, or if parent already violates the risk bound 
			# don't try to propagate, since children guaranteed to violate 
			if np.isclose(parent_risk, 1.0) or (parent_risk > parent_bound):
				infeasible = True 
			# Only if the parent bound and the parent risk are below 1, and the parent risk is below the parent bound,
			# then try to propagate risk 
			else: 
				infeasible = False 
				# risk "consumed" by parent node 
				parent_term = (parent_bound - parent_risk)/(1.0 - parent_risk)

				for idx_child, child in enumerate(child_list):
					# Risk consumed by the siblings of the current node 
					sibling_term = np.sum([p*c.exec_risk for (p,c) in zip(prob_safe_list, child_list) if (c!= child)])
					
					# exec risk bound, which caps ar 1.0 
					exec_risk_bound = min([(parent_term-sibling_term)/prob_safe_list[idx_child], 1.0])
					# A negative bound means that the chance constraint is guaranteed
					# to be violated. The same is true if the admissible estimate
					# of the execution risk for a child node violates its upper bound.
					if exec_risk_bound >= 0.0:
						if child.exec_risk<=exec_risk_bound or np.isclose(child.exec_risk,exec_risk_bound):
							exec_risk_bounds[idx_child] = exec_risk_bound
						else:
							infeasible = True
							break
					else:
						infeasible = True
						break
		return exec_risk_bounds, infeasible

	def build_ancestor_list(self, expanded_node):
		# create set Z that contains the expanded node and all of its ancestors in the explicit graph 
		# along marked action arcs (ancestors nodes from best policy) 
		Z = []
		queue = deque([expanded_node])
		while len(queue) > 0:
			node = queue.popleft()
			Z.append(node)
			for parent in self.graph.all_node_ancestors(node):
				if not parent.terminal and parent.best_action != None:
					if node in self.graph.hyperedge_successors(parent, parent.best_action):
						queue.append(parent)
		return Z 

	def extract_policy(self):
		# extract policy mapping nodes to actions 
		queue = dequeue([self.graph.root]) # from root 
		policy = {}
		while len(queue) > 0:
			node = queue.popleft()
			if node.best_action != None:
				policy[node] = node.best_action
				children = self.graph.hyperedge_successors(node, node.best_action)
				queue.extend(children)
		return policy 

	def choose_node(self):
		# chooses an element from open list to be expanded 
		if len(self.opennodes) > 1:
			# selects best node to expand 
			node = self.select_best(self.opennodes) # select best node 
			self.opennodes.remove(node)
		else: # if there is only one, use that one
			node = self.opennodes.pop()
		return node 

	def obtain_child_objs_and_probs(self, belief, T, O, r, act):
		# predicts new particles using current belief and state transition 
		pred_belief, pred_belief_safe = predict_belief(belief, T, r, act)
		# Given the predicted belief, computes the probability distribution of potential observations. 
		# Each observations corresponds to a new node on the hypergraph, whose edge is annotated by the 
		# prob of that particular observation 
		obs_distribution, obs_distribution_safe, state_to_obs = compute_observation_distribution(pred_belief, pred_belief_safe, O)

		# for each observation, computes corresponding updated belief 
		child_obj_list=[]; prob_list=[]; prob_safe_list=[]; new_child_idxs=[]
		pretty_obs_list=[]; count=0
		for str_obs, (obs, obs_prob) in obs_distribution.items():
			# Performs belief state update 
			child_blf_state = update_belief(pred_belief, state_to_obs, obs)

			candidate_child_obj = RAOStarGraphNode(child_blf_state)
 
			if self.graph.has_node(candidate_child_obj): # if node already present
				child_obj = self.graph.nodes[candidate_child_obj]
			else: 
				# the node initiated 
				child_obj = candidate_child_obj
				new_child_idxs.append(count)
			child_obj_list.append(child_obj)
			prob_list.append(obs_prob)

			if str_obs in obs_distribution_safe: 
				obs_safe_tuple = obs_distribution_safe[str_obs]
				prob_safe_list.append(obs_safe_tuple[1])
			else: 
				prob_safe_list.append(0.0)
			pretty_obs_list.append(self.model)
			count += 1
		return child_obj_list, prob_list, prob_safe_list, new_child_idxs, pretty_obs_list

