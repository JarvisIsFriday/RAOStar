#!/usr/bin/env python

# author: Matt Deyo
# email: mdeyo@mit.edu
# simple vehicle models based on MERS Toyota work for rao star

import numpy as np


class VehicleModel(object):
    '''First Python version of Geordi vehicle model, this one for intersections.

    Attributes:
        name (str): Name to id each vehicle.
        current_state (dictionary): Maps variables to values for current state
            of the vehicle. Starts with initial state and is used during execution.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    '''

    def __init__(self, name, initial_state, model_action_list=[], isControllable=False, speed_limits=[0, 10], DetermObs=True):
        self.name = name
        self.current_state = initial_state
        self.speed_limits = speed_limits
        self.action_list = model_action_list

    def add_action(self, action_model):
        for action in self.action_list:
            if action.name == action_model.name:
                print('VehicleModel: ' + self.name +
                      ' already has action named: ' + action.name)
                return False
        action_model.agent_name = self.name
        self.action_list.append(action_model)
        return True

    def __repr__(self):
        return "VehicleModel: " + self.name + " " + str(self.current_state)


class ActionModel(object):
    """Model for each vehicle action with preconditions and effects functions.

    Attributes:
        name (str): Name to id each vehicle.
        current_state (dictionary): Maps variables to values for current state
            of the vehicle. Starts with initial state and is used during execution.
        attr2 (:obj:`int`, optional): Description of `attr2`.
    """

    def __init__(self, name, precondition_check=lambda x: False, effect_function=lambda x: x, action_cost=1):
        self.name = name
        self.precondition_check = precondition_check
        self.effect_function = effect_function
        self.cost = action_cost
        self.agent_name = "unassigned"

    def __repr__(self):
        return "ActionModel: " + self.name

    '''
    def state_valid(self, state):  # check if a particular state is valid
        if state[0] < 0 or state[1] < 0:
            return False
        try:
            return self.env[state[0], state[1]] == 0
        except IndexError:
            return False

    def actions(self, state):
        validActions = []
        for act in [(1, 0, "down"), (0, 1, "right"), (-1, 0, "up"), (0, -1, "left")]:
            newx = state[0] + act[0]
            newy = state[1] + act[1]
            if self.state_valid((newx, newy)):
                validActions.append(act)
        if state == self.goal:
            return []
        return validActions

    def is_terminal(self, state):
        return state == self.goal  # not sure what this does #mdeyo: is this comment from Yun?

    def state_transitions(self, state, action):
        newstates = []
        intended_new_state = (state[0] + action[0], state[1] + action[1])
        if not self.state_valid(intended_new_state):
            return newstates
        if state in self.icy_blocks:
            newstates.append((intended_new_state, self.icy_move_forward_prob))
            for slip in [-1, 1]:
                slipped = [(action[i] + slip) % 2 * slip for i in range(2)]
                slipped_state = (state[0] + slipped[0], state[1] + slipped[1])
                if self.state_valid(slipped_state):
                    newstates.append(
                        (slipped_state, (1 - self.icy_move_forward_prob) / 2))
        else:
            newstates.append((intended_new_state, 1.0))
        return newstates

    def observations(self, state):
        if self.DetermObs:
            return [(state, 1.0)]
        else:  # robot only knows if it is on icy or non icy block
            if state in self.icy_blocks:
                return [(self.icy_blocks[i], 1 / len(self.icy_blocks)) for i in range(len(self.icy_blocks))]
            else:
                prob = 1 / (6 - len(self.icy_blocks))
                dist = []
                for i in range(3):
                    for j in range(3):
                        if self.env[i, j] == 0 and (i, j) not in self.icy_blocks:
                            dist.append((i, j), prob)
                return dist

    def state_risk(self, state):
        for fire in self.fires:
            if state == fire:
                return 1.0
        return 0.0

    def costs(self, action):
        if action[2] == "up":
            return 1  # bias up action
        else:
            return 1

    def values(self, state, action):
        # return value (heuristic + cost)
        return self.costs(action) + self.heuristic(state)

    def heuristic(self, state):
        # square of euclidean distance as heuristic
        return sum([(self.goal[i] - state[i])**2 for i in range(2)])

    def execution_risk_heuristic(self, state):
        # sqaure of euclidean distance to fire as heuristic
        return 0
        # return sum([(self.fire[i] - state[i])**2 for i in range(2)])

    def print_model(self):
        height, width = self.env.shape
        print(" ")
        print("    ** Model environment **")
        for j in range(height):
            # print("row: " + str(j))
            row_str = "   "
            for i in range(width):
                if self.env[j][i]:
                    row_str += " [-----] "
                    continue
                row_str += " [" + str(j) + "," + str(i)
                if self.goal == (j, i):
                    row_str += " g] "
                elif self.state_risk((j, i)):
                    row_str += " f] "
                elif (j, i) in self.icy_blocks_lookup:
                    row_str += " i] "
                else:
                    row_str += "  ] "
            print(row_str)

    def print_policy(self, policy):
        height, width = self.env.shape
        policy_map = np.zeros([height, width])

        for key in policy:
            coords = key.split(":")[0].split("(")[1].split(")")[0]
            col = int(coords.split(",")[0])
            row = int(coords.split(",")[1])
            action_string = policy[key]
            for action_name in self.action_map:
                if action_name in action_string:
                    policy_map[col][row] = self.action_map[action_name]
                    break
        print(" ")
        print("         ** Policy **")
        for j in range(height):
            # print("row: " + str(j))
            row_str = "   "
            for i in range(width):
                if self.goal == (j, i):
                    row_str += " [goal] "
                elif self.state_risk((j, i)):
                    row_str += " [fire] "
                elif self.env[j][i]:
                    row_str += " [----] "
                else:
                    if policy_map[j][i] == 5:
                        row_str += " [ -> ] "
                    if policy_map[j][i] == 6:
                        row_str += " [ <- ] "
                    if policy_map[j][i] == 7:
                        row_str += " [ ^^ ] "
                    if policy_map[j][i] == 8:
                        row_str += " [ vv ] "
                    if policy_map[j][i] == 0:
                        row_str += " [    ] "
            print(row_str)
    '''
