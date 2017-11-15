#!/usr/bin/env python

# author: Yun Chang
# email: yunchang@mit.edu
# rao* on r2d2 model

from utils import import_models
import_models()
from r2d2model import R2D2Model
from raostar import RAOStar
from ashkan_icaps_model import *

# from iterative_raostar import *

ice_blocks = [(1, 0), (1, 1)]
model = R2D2Model(ice_blocks)
model2 = Ashkan_ICAPS_Model()
# algo = RAOStar(model, cc=0.08)
algo = RAOStar(model2, cc=0.9, ashkan_continuous=True)

b_init = {(1, 0, 0): 1.0}
b0 = ContinuousBeliefState(1, 1)
# P, G = algo.search(b_init)
P, G = algo.search(b0)

# print(P)

# Remove all the states from policy that did not have a best action
P_notNone = {}
for i in P:
    if P[i] != 'None':
        P_notNone[i] = P[i]

P = P_notNone

# for key, value in P.items():
# print("{0:.2f}".format(float(value['state'].mean_b[0][0])),
#       "{0:.2f}".format(float(value['state'].mean_b[1][0])), value['action'])


def best_action_children(G, P, print_out=False):
    children_names = []
    children = []
    action = G.root.best_action
    prob_outcome = 0
    if print_out:
        print('# Outcomes from best action:')
    for i, child in enumerate(G.hyperedge_successors(G.root, G.root.best_action)):
        prob_outcome = action.properties['prob'][i]
        children_names.append((child.name, prob_outcome))
        children.append(child)
        if print_out:
            print('##   state: ' + child.name + ' prob: ' + str(prob_outcome) +
                  ' exec_risk_bound: ' + str(child.exec_risk_bound))
    return children


def next_child(G, state):
    next_child = None
    most_probable = 0
    for i, child in enumerate(G.hyperedge_successors(state, state.best_action)):
        prob_outcome = state.best_action.properties['prob'][i]
        if prob_outcome > most_probable:
            next_child = child
            most_probable = prob_outcome
    print(" action ", state.best_action.name)
    print("next state: ", "{0:.2f}".format(float(
        next_child.state.mean_b[0][0])), "{0:.2f}".format(float(next_child.state.mean_b[1][0])))
    # next_child.state.sigma_b)
    return next_child


print("Next best action is: " + G.root.best_action.name)
print(best_action_children(G, P, True))
# model.print_model()
# model.print_policy(P)

s1 = next_child(G, G.root)
s2 = next_child(G, s1)
s3 = next_child(G, s2)
s4 = next_child(G, s3)
s5 = next_child(G, s4)

# for n in G.nodes.values():
# n.print_node()
