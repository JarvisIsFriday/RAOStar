#!/usr/bin/env python

# author: Yun Chang
# email: yunchang@mit.edu
# rao* on r2d2 model

from utils import import_models
import_models()
from r2d2model import R2D2Model
from raostar import RAOStar
import graph_to_json

chance_constraint = 0.08 

ice_blocks = [(1, 0), (1, 1)]
model = R2D2Model(ice_blocks)
algo = RAOStar(model, cc=chance_constraint)

b_init = {(1, 0, 0): 1.0}

P, G = algo.search(b_init)
# print(P)

model.print_model()
model.print_policy(P)

# for n in G.nodes.values():
# 	n.print_node()

graph_to_json.graph_to_json(G, chance_constraint, "r2d2_raos.json")