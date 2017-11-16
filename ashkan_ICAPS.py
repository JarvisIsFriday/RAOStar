from utils import import_models
import_models()
from raostar import RAOStar
from ashkan_icaps_model import *
from iterative_raostar import *

model = Ashkan_ICAPS_Model()
algo = RAOStar(model, cc=0.99, debugging=True, ashkan_continuous=True)

b0 = ContinuousBeliefState(0, 0)
P, G = algo.search(b0)

most_likely_policy(G, model)
