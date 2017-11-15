from utils import import_models
import_models()
from r2d2model import R2D2Model
from ashkan_icaps_model import *
from raostar import RAOStar
from belief import *

###############
#
###############
model = Ashkan_ICAPS_Model()
algo = RAOStar(model, cc=0.09, ashkan_continuous=True)
b_init = ContinuousBeliefState()
P, G = algo.search(b_init)

model.print_model()
model.print_policy(P)
