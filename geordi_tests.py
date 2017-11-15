#!/usr/bin/env python

# author: Matt Deyo
# email: mdeyo@mit.edu
# simple vehicle models based on MERS Toyota work for rao star

from utils import import_models
import_models()
from vehicle_model import *
from geordi_road_model import *

road_model = highway_2_lanes_offramp_ex()
print(road_model)
plot_road_model(road_model)

ego_vehicle = VehicleModel(
    "ego", {'road': 'one', 'x': 0, 'y': 0, 'v': 0, 'theta': 0})

agent_vehicle1 = VehicleModel(
    "agent1", {'road': 'two', 'x': 0, 'y': 0, 'v': 0, 'theta': 0})

move_forward = ActionModel("forward")
ego_vehicle.add_action(move_forward)

move_forward_agent = ActionModel("forward")
agent_vehicle1.add_action(move_forward_agent)

# geordi_model = GeordiModel()
geordi_model = GeordiModel([ego_vehicle, agent_vehicle1])
# geordi_model.add_vehicle_model(ego_vehicle)
# geordi_model.add_vehicle_model(agent_vehicle1)
