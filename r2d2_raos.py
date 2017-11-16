#!/usr/bin/env python

# author: Yun Chang
# email: yunchang@mit.edu
# rao* on r2d2 model

from utils import import_models
import_models()
from r2d2model import R2D2Model
from raostar import RAOStar
from iterative_raostar import *

ice_blocks = [(1, 0), (1, 1)]
model = R2D2Model(ice_blocks)
algo = RAOStar(model, cc=0.08)

b_init = {(1, 0, 0): 1.0}
P, G = algo.search(b_init)

P = clean_policy(P)

model.print_model()
model.print_policy(P)


most_likely_policy(G, model)
