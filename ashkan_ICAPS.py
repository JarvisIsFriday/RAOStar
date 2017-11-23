from utils import import_models
import_models()
from raostar import RAOStar
from ashkan_icaps_model import *
from iterative_raostar import *
from icaps_sim import *
import numpy as np


model = Ashkan_ICAPS_Model("70% risk")
algo = RAOStar(model, cc=0.8, debugging=False, ashkan_continuous=True)

b0 = ContinuousBeliefState(1, 1)
P, G = algo.search(b0)

most_likely_policy(G, model)
Sim = Simulator(10, 10, G, P, model, grid_size=50)

for i in range(11):
    for j in range(11):
        risk = static_obs_risk_coords(
            np.matrix([i, j]), np.matrix([[0.2, 0], [0, 0.2]]))
        Sim.draw_risk_colors(i, j, risk)
        # print(i, j, risk)

Sim.start_sim()
