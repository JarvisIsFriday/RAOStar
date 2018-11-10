#!/usr/bin/env python

# author: Cyrus Huang
# email: huangxin@mit.edu
# simple vehicle left turn scenario used for ICAPS19-intention 

from utils import import_models
import_models()
from intention_vehicle_model import *
from geordi_road_model import *
from raostar import RAOStar
from pprint import pprint
import ast

def prettyprint(policy):
	for keys, values in policy.items():
		state, probability, depth = keys
		best_action = values

		node_info = {}
		node_info['state'] = state
		node_info['probability'] = probability
		node_info['depth'] = depth
		node_info['the_best_action'] = best_action

		# print(ast.literal_eval(state))

		pprint(node_info)

# assume an open road
road_model = intersection_left_turn_ex()

# add an ego vehicle that can choose from stop (wait) and turn
ego_vehicle = VehicleModel('Ego', VehicleState(
    state={'x': 88, 'y': 180}), isControllable=True)
ego_vehicle.add_action(stop_action(ego=True))
ego_vehicle.add_action(turn_left_action(ego=True))

# add an agent vehicle that can go forward or slow down
agent1_vehicle = VehicleModel('Agent1', VehicleState(
    state={'x': 92, 'y': 240}))
agent1_vehicle.add_action(agent_forward_action())
agent1_vehicle.add_action(agent_slow_down_action())

# geordi_model = GeordiModel()
geordi_model = GeordiModel(
    [ego_vehicle, agent1_vehicle], road_model)
print(geordi_model.road_model)
actions = geordi_model.get_available_actions(geordi_model.current_state)
print(actions)
print(agent1_vehicle.action_list[0].precondition_check(agent1_vehicle.name,
                                                       geordi_model.current_state, geordi_model))

new_states = geordi_model.state_transitions(
    geordi_model.current_state, actions[0])

print('new_states', new_states)

algo = RAOStar(geordi_model, cc=0.1, debugging=False, cc_type='o', fixed_horizon = 3)

b_init = {geordi_model.current_state: 1.0}
P, G = algo.search(b_init)

# print(P)
prettyprint(P)

# permutations = calculate_permutations(
#     geordi_model, geordi_model.current_state, agent2_vehicle.action_list, 5)

# print(len(permutations))
# print(permutations)

# for p in permutations:
# print(p['actions'])

# geordi_model.add_vehicle_model(ego_vehicle)
# geordi_model.add_vehicle_model(agent_vehicle1)