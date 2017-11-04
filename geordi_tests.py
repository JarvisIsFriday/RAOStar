#!/usr/bin/env python

# author: Matt Deyo
# email: mdeyo@mit.edu
# simple vehicle models based on MERS Toyota work for rao star

from utils import import_models
import_models()

from vehicle_model import *

ego_vehicle = VehicleModel("ego", {'x': 0, 'y': 0, 'v': 0, 'theta': 0})

move_forward = ActionModel("forward")
ego_vehicle.add_action(move_forward)

print(ego_vehicle)
print(ego_vehicle.action_list)
