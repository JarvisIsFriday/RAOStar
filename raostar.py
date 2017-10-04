#!/usr/bin/env python
# Author: Yun Chang 
# Email: yunchang@mit.edu 

# framework for rao* search 

from tree import Tree, Action, State

def policy_terminal(policy):
	if [] in policy.values():
		return False
	return True

def RAOstar(initial_belief, chance_constraint, scenario):
	# initial belief is a state
	# scenario is a setup of the environment: the 
	G = Tree() # Tree (or graph) just a way to 
	# systematically explore 
	G.add_state(initial_belief) # start with initial belief 
	P = {initial_belief:[]} # policy 
	# iterate until policy non-terminal 
	while not policy_terminal(P):
		n, G = expandPolicy(G, P)
		P = updatePolicy(n, G, P)
	return P 

def select_best(treegraph, optimization="max"):
	# select the best leaf to expand
	# optimization can be "min" or "max"\
	if optimization == 'max':
		return max(treegraph.open_states, key=lambda n: n.value)
	else:
		return min(treegraph.open_states, key=lambda n: n.value)

def estimateQER(belief, action, scenario):
	# calculate Q*: heuristics and er: execution risk 
	# step 1: calculate probability of system in constrain violating path 
	rb = 0 
	for s in belief.states:
		cv = 0
		if not scenario.pass_constraint(s[0]):
			cv = 1
		rb += cv*s[1]
	# step 2: calculate safe prior belief 
	# note (1 - rb) is just a normalization constant 
	safe_prior = {}
    for state in belief.states:
    	if scenario.pass_constraint(state[0]):
	        probstate = scenario.transferFunction(state)
	        for s in probstate:
	            if s[0] not in prior.keys():
	                safe_prior[s[0]] = s[1]/(1-rb)
	            else:
	                safe_prior[s[0]] += s[1]/(1-rb)
	# step 3: calculate probability or safety
	for s_ in safe_prior.keys():
            safe_prior[s_]*= observeFunction(s_)
    P_sa = sum(prior.values())
    # calculate execution risk 


def expandPolicy(treegraph, policy, scenario):
	n = select_best(treegraph)
	for a in scenario.get_action(n):
		T = scenario.transferFunction(n,a)
		O = scenario.observeFunction(n)
		R = scenario.rewardFunction(n)
		ch = n.expandNode(T, O)
		Q_star, er = estimateQER(n, a, scenario)
		
		


if __name__ == '__main__':
	b_0 = State({'cold':1.0}, None)
	cc = 0.1
	RAOstar(b_0, cc)

