#!/usr/bin/env python

# continuous dynamics and observation model to be used with RAO*
# Matt Deyo 2017


""" environment example
10x10 continuous space
_____________
|			|
|			|
|			|
|			|
|___________|
"""


import numpy as np
from scipy.stats import norm
from continuous_belief import *


class Ashkan_ICAPS_Model(object):
    def __init__(self, name="unnamed model"):
        self.vel = 2
        self.name = name

        # environment size (x,y)
        # model walls as risks
        # goal specify direction as well as coordinate (x,y,thet)
        self.goal_area = [8, 8]
        self.max_coords = [10, 10]
        # self.goal_area = [5, 5]

        # now trying goal is just hitting x coord
        self.goal_x = 5
        self.optimization = "minimize"
        # note state include time (depth)

    def actions(self, state):
        '''
        Available ego actions not dependent on state.
        Not limiting the ego to stay without bounds.
        '''
        diag_factor = 1 / np.sqrt(2)
        return [['DOWN', 0, -self.vel], ['DOWN-RIGHT', self.vel * diag_factor, -self.vel * diag_factor], ['RIGHT', self.vel, 0], ['UP-RIGHT', self.vel * diag_factor, self.vel * diag_factor], ['UP', 0, self.vel], ['UP-LEFT', -self.vel * diag_factor, self.vel * diag_factor], ['LEFT', -self.vel, 0], ['DOWN-LEFT', -self.vel * diag_factor, -self.vel * diag_factor]]

    def is_terminal(self, state):
        # print('is_terminal', state.state_print())
        x, y = state.mean_b[0], state.mean_b[1]
        # return(x > self.goal_area[0] and x < self.max_coords[0] and y >
        # self.goal_area[1] and y < self.max_coords[1])
        result = (x > self.goal_area[0] and y > self.goal_area[1])
        # if result:
        # print('found goal!')
        return result
        # return(state.mean_b[0] > self.goal_x)

    def state_transitions(self, state, action):
        '''The uncontollable agent obstacle can move left [-1,0] or right [1,0]
        each time step with equal probability'''
        new_states = []
        control_input = [[action[1]], [action[2]]]
        new_states.append(
            [cont_belief_update(state, control_input), 1.0])
        # print(new_states[0][0].mean_b)
        # print(new_states)
        return new_states

    def observations(self, state):
        return [(state, 1.0)]  # assume observations is deterministic

    def state_risk(self, state):
        risk = static_obs_risk(state)
        # print('state_risk:' + str(state.state_print()) + " {0:.2f}".format(risk))
        return risk

    def costs(self, action):
        '''
        Treating all ego actions as uniform cost, not trying to guide behavior
        '''
        return 2

    def values(self, state, action):
        # return value (heuristic + cost)
        if self.is_terminal(state):
            return 0

        return self.costs(action)
        # return self.costs(action) + self.heuristic(state)

    def heuristic(self, state):
        # square of euclidean distance as heuristic
        if isinstance(state, ContinuousBeliefState):
            state = state
        else:
            state = state.state
        if self.is_terminal(state):
            return 0
        # print('h state', state)
        # distance_to_goal_corner = np.sqrt((state.mean_b[0] - self.goal_x)**2)
        distance_to_goal_corner = np.sqrt(
            (state.mean_b[0] - (self.goal_area[0] + 1))**2 + (state.mean_b[1] - (self.goal_area[1] + 1))**2)
        return distance_to_goal_corner
        # return np.sqrt(sum([(self.goal[i] - state[0][i])**2 for i in
        # range(2)]))

    def execution_risk_heuristic(self, state):
        return 0  # don't have a good heuristic for this yet


a1 = 2
bb1 = -2
a2 = 1
bb2 = 2
a3 = -4
bb3 = 27
a4 = float(-1 / 3.0)
bb4 = 5


def one_line_risk(a1, b1, m1, std1):
    # print('a1', a1)
    # print('b1', b1)
    # print('m1', m1)
    # print('p1', p1)
    new_array = np.matrix([[-a1, 1]])
    mo2 = float(new_array * m1.T - b1)
    po2 = float(new_array * std1 * new_array.T)
    # print('mo2', mo2)
    # print('po2', po2)
    return 1 - norm.cdf(0, mo2, po2)


def static_obs_risk_coords(xy, std):
    risks = [1 - one_line_risk(a1, bb1, xy, std),
             1 - one_line_risk(a2, bb2, xy, std),
             1 - one_line_risk(a3, bb3, xy, std),
             one_line_risk(a4, bb4, xy, std)]
    return min(risks)


def static_obs_risk(belief_state):
    m1 = belief_state.mean_b[0:2].T
    var1 = belief_state.sigma_b[0:2, 0:2]
    std1 = np.sqrt(belief_state.sigma_b[0:2, 0:2])
    # print('m1', m1)
    # print('var1', var1)
    # print('std1', std1)
    risks = [1 - one_line_risk(a1, bb1, m1, std1),
             1 - one_line_risk(a2, bb2, m1, std1),
             1 - one_line_risk(a3, bb3, m1, std1),
             one_line_risk(a4, bb4, m1, std1)]
    return min(risks)
