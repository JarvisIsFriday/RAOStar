#!/usr/bin/env python

# author: Yun Chang
# email: yunchang@mit.edu
# rao* on quadcopter with guest model

from utils import import_models
import_models()

from quad_model import QuadModel
from raostar import RAOStar


# note the boundary of the world (ex anything with row column 0 and the upper bound)
# is the wall
model = QuadModel((7, 7), (5, 5))
algo = RAOStar(model, cc=0.1)

b_init = {((1, 1, 90), (3, 1, 90)): 1.0}

P, G = algo.search(b_init)

# get the policies that does not give none
P_notNone = {}
for i in P:
    if P[i] != 'None':
        P_notNone[i] = P[i]

print(P_notNone)

# print out the policy for each state of guest 

