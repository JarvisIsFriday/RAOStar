#!/usr/bin/env python
# referenced Pedro Santana's original belief.py for his package of rao*
# slimmed it down and simplify

# author: Yun Chang
# yunchang@mit.edu

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import norm
from matplotlib.collections import LineCollection

sigma_w = 0.07
sigma_v = 0.05
sigma_b0 = 0.01


deltaT = 0.5
m = 10
b = 50

# Dynamics
n = 4
Ax = np.matrix([[1, deltaT], [0, 1 - deltaT * b / m]])
Bx = np.matrix([[0], [deltaT / m]])
Ay = np.matrix([[1, deltaT], [0, 1 - deltaT * b / m]])
By = np.matrix([[0], [deltaT / m]])

A = np.matrix([[Ax[0, 0], 0, Ax[0, 1], 0],
               [0,  Ay[0, 0], 0, Ay[0, 1]],
               [0, Ax[1, 0], Ax[1, 1], 0],
               [0, 0, Ay[1, 0], Ay[1, 1]]])

B = np.matrix([[0, 0], [0, 0], [Bx[1, 0], 0], [0, By[1, 0]]])

C = np.eye(n)
Bw = np.eye(n)
Dv = np.eye(n)

K = np.matrix([[0, 0, 29.8815, 0],
               [0, 0, 0, 29.8815]])

Ac = A + B * K
K0 = np.diag([(1 - Ac[2, 2]) / B[2, 0], (1 - Ac[3, 3]) / B[3, 1]])
print('k0', K0)

r = [[1], [1]]

print('\nA:\n', A)
print('\nB:\n', B)

a1 = 2
bb1 = -2
a2 = 1
bb2 = 2
a3 = -4
bb3 = 27
a4 = -1 / 3
bb4 = 5


class ContinuousBeliefState(object):
    """
    Class representing a continuous belief state.
    """

    def __init__(self, x=0, y=0, v_x=0, v_y=0, decimals=5):
        self.n = 4
        self.nn = self.n * 2
        self.mean_b = np.zeros([self.nn, 1])
        self.mean_b[0] = x
        self.mean_b[1] = y
        self.mean_b[2] = v_x
        self.mean_b[3] = v_y
        self.belief = self

        self.sigma_b = np.zeros([self.nn, self.nn])
        for i in range(self.n):
            self.sigma_b[i][i] = sigma_b0
        # print("\nmean_b0:\n", self.mean_b)
        # print("\nsigma_b0:\n", self.sigma_b)
        self.current_P = sigma_b0 * np.eye(n)

    def __len__(self):
        return 7

    def copy(self):
        '''Return a copy of the this belief_state as a new object'''
        new_belief_state = ContinuousBeliefState()
        new_belief_state.mean_b = self.mean_b
        new_belief_state.sigma_b = self.sigma_b
        new_belief_state.current_P = self.current_P
        return new_belief_state


def cont_belief_update(belief_state, control_input):
    time_steps = 2

    # P = sigma_b0 * np.eye(n)
    # print(P)

    noise_covariance = np.diag(
        [sigma_w, sigma_w, sigma_w, sigma_w, sigma_v, sigma_v, sigma_v, sigma_v])

    current_belief = belief_state.copy()
    new_m_b = None
    new_sigma_b = None
    new_P = None

    for k in range(time_steps):

        m_b = current_belief.mean_b
        sigma_b = current_belief.sigma_b
        P = current_belief.current_P

        W = np.random.normal(0, sigma_w)
        V = np.random.normal(0, sigma_v)

        term_in_L = (A * P * A.T + Bw * sigma_w * np.eye(n) * Bw.T)
        L = term_in_L * C.T * (C * term_in_L * C.T +
                               Dv * sigma_v * np.eye(n) * Dv.T).I
        new_P = (np.eye(n) - L * C) * term_in_L
        # print('L', L)
        # print('P', P)

        # print(np.matrix([[A, B * K], 0]))
        # print(np.matrix([A, B * K]))
        # print([A, B * K])

        topAA = np.concatenate((A, B * K), 1)
        bottomAA = np.concatenate((L * C * A, A + B * K - L * C * A), 1)

        AA = np.concatenate((topAA, bottomAA), 0)
        # print(AA)

        BBu = np.concatenate((B, B), 0)
        # print(BBu)
        BBtop = np.concatenate((Bw, np.zeros([n, n])), 1)
        BBbottom = np.concatenate((L * C * Bw, L * Dv), 1)
        BB = np.concatenate((BBtop, BBbottom), 0)
        # print('BB', BB)

        new_m_b = AA * m_b + BBu * K0 * control_input
        new_sigma_b = AA * sigma_b * AA.T + BB * noise_covariance * BB.T

        current_belief.mean_b = new_m_b
        current_belief.sigma_b = new_sigma_b
        current_belief.current_P = new_P

        # print('m_b at ', k, ' is:\n', new_m_b)
        # print('sigma_b at ', k, ' is:\n', new_sigma_b)

    new_belief_state = ContinuousBeliefState()
    new_belief_state.mean_b = new_m_b
    new_belief_state.sigma_b = new_sigma_b
    new_belief_state.current_P = new_P

    return new_belief_state


def plot_belief_state(axis, belief_state, color=(1, 0, 0, 0.5)):
    new_ellipse = Ellipse(xy=(belief_state.mean_b[0], belief_state.mean_b[1]),
                          width=2 * np.sqrt(belief_state.sigma_b[0, 0]), height=2 * np.sqrt(belief_state.sigma_b[1, 1]), angle=0)
    new_ellipse.set_facecolor(color)
    axis.add_artist(new_ellipse)


def dynamic_obs_risk(ego_belief_state, obs_belief_state):
    ego_m = ego_belief_state.mean_b[0:2]
    ego_s = ego_belief_state.sigma_b[0:2, 0:2]
    obs_m = obs_belief_state.mean_b[0:2]
    obs_s = obs_belief_state.sigma_b[0:2, 0:2]
    x_mean, y_mean = ego_m - obs_m
    [[x_std, _], [_, y_std]] = np.sqrt(ego_s + obs_s)
    collision_box = 1
    x_risk = norm.cdf(collision_box, x_mean, x_std) - \
        norm.cdf(-collision_box, x_mean, x_std)
    y_risk = norm.cdf(collision_box, y_mean, y_std) - \
        norm.cdf(-collision_box, y_mean, y_std)
    return min(x_risk, y_risk)


class BeliefState(object):
    """
    Class representing a discrete belief state.
    """

    def __init__(self, belief_dict, decimals=5):
        self.belief = belief_dict

    @property
    def belief(self):
        """Dictionary representing the belief state"""
        return self._belief_dict

    @belief.setter
    def belief(self, new_belief):
        if isinstance(new_belief, dict):
            self._belief_dict = new_belief
        else:
            raise TypeError(
                'Belief states should be given in dictionary form.')

    @property
    def particle_prob_tuples(self):
        """List of particles and their associated probabilities"""
        return list(self.belief.values())

    @property
    def particles(self):
        """List of particles in the belief state."""
        return [p_tup[0] for p_tup in self.particle_prob_tuples]

    @property
    def probabilities(self):
        """List of probabilities in the belief state."""
        return [p_tup[1] for p_tup in self.particle_prob_tuples]

    @property
    def entropy(self):
        """Entropy associated with the belief state."""
        return np.sum([(-p * np.log2(p) if not np.isclose(p, 0.0) else 0.0) for p in self.probabilities])


def avg_func(belief, func, *args):
    """Averages the output of a function over a belief state."""
    # basically the expected value of the function - YC
    avg_value = 0.0
    for state, prob in belief.items():
        # Applies function on a state, with any number of supplied arguments
        # after that, and averages with the probability.
        avg_value += func(state, *args) * prob
    return avg_value


def bound_prob(prob):
    """Ensures that a probability value is within [0.0,1.0]"""
    return min(1.0, max(0.0, prob))


def blf_indicator(op_type, belief, ind_func, *args):
    """Applies an indicator function to a belief, with the option of
    stopping at the first True or False."""
    if op_type == 'count':  # How many times it returns true
        count = 0
        for state, prob in belief.items():
            count += ind_func(state, *args)
        return count
    elif op_type == 'prob':  # Probability of being true
        prob = 0.0
        for state, prob in belief.items():
            prob += ind_func(state, *args) * prob
        return prob
    elif op_type == 'has_true':  # Contains true
        for state, prob in belief.items():
            if ind_func(state, *args):
                return True
        return False
    elif op_type == 'has_false':  # Contains false
        for state, prob in belief.items():
            if not ind_func(state, *args):
                return True
        return False


def is_terminal_belief(blf_state, term_fun, terminal_prob):
    """Determines if a given belief state corresponds to a terminal node,
    according to the stopping criterion."""
    # Evaluates the terminal indicator function over the whole belief state,
    # returning True only is the belief has no nonterminal state.
    if terminal_prob == 1.0:
        return not blf_indicator('has_false', blf_state, term_fun)
    # Flags a state as being terminal if a large percentage of its particles
    # are terminal states.
    else:
        return blf_indicator('prob', blf_state, term_fun) >= terminal_prob


def predict_belief(belief, T, r, act):
    """
    Propagates a belief state forward according to the state transition
    model. Also computes the *safe predicted belief*, i.e., the predicted
    belief coming from particles in non-constraint-violating paths.
    """
    pred_belief = {}
    pred_belief_safe = {}
    sum_safe = 0.0
    # For every particle of the current belief
    for particle_state, particle_prob in belief.items():

        if np.isclose(r(particle_state), 0.0):  # Safe belief state
            safe_state = True
            sum_safe += particle_prob
        else:
            safe_state = False

        # For every possible next state (with some probability)
        for next_state, trans_prob in T(particle_state, act):
            # Probability for the next state
            next_prob = particle_prob * trans_prob

            # Ensures that impossible transitions do not 'pollute' the belief
            # with 0 probability particles.
            if next_prob > 0.0:
                if next_state in pred_belief:
                    pred_belief[next_state] += next_prob
                else:
                    pred_belief[next_state] = next_prob

                if safe_state:  # Safe belief state
                    if next_state in pred_belief_safe:
                        pred_belief_safe[next_state] += next_prob
                    else:
                        pred_belief_safe[next_state] = next_prob

    if sum_safe > 0.0:  # Not all particles are on violating paths
        # Normalizes the safe predicted belief
        for next_state, b_tuple in pred_belief_safe.items():
            pred_belief_safe[next_state] /= sum_safe

    return pred_belief, pred_belief_safe


def compute_observation_distribution(pred_belief, pred_belief_safe, O):
    """Computes the probability of getting an observation, given some
    predicted belief state."""
    obs_distribution = {}  # Prob. distrib. of observations
    obs_distribution_safe = {}  # Prob. distrib. of 'safe' observations
    state_to_obs = {}  # Mapping from state to possible observations (used
    # later as likelihood function in belief state update)

    beliefs = [pred_belief, pred_belief_safe]
    distribs = [obs_distribution, obs_distribution_safe]
    sum_probs = [0.0, 0.0]

    # Iterates over the different belief and corresponding distributions
    for i, (belief, distrib) in enumerate(zip(beliefs, distribs)):

        # For every particle in the current predicted belief
        for particle_state, particle_prob in belief.items():

            # Ensures that 0 probability particles do not 'pollute' the
            # likelihood function.
            if particle_prob > 0.0:
                if i == 0:  # i == 0 the belief, i == 1 the safe belief
                    state_to_obs[particle_state] = []

                # For every possible observation (with some probability)
                for obs, obs_prob in O(particle_state):

                    # Ensures that impossible observations do not 'pollute' the
                    # likelihood with 0 probability observations.
                    if obs_prob > 0.0:
                        if i == 0:
                            state_to_obs[particle_state].append(
                                [obs, obs_prob])

                        if obs not in distrib:
                            distrib[obs] = obs_prob * particle_prob
                        else:
                            distrib[obs] += obs_prob * particle_prob

                        # Accumulates the probabilities
                        sum_probs[i] += obs_prob * particle_prob

    return obs_distribution, obs_distribution_safe, state_to_obs


def update_belief(pred_belief, state_to_obs, obs):
    """Performs belief state update."""

    post_belief = copy_belief(pred_belief)  # Does not copy the state objects.

    # For every particle in the current belief
    prob_sum = 0.0
    zero_prob_states = []
    for state, prob in post_belief.items():
        # Checks if obs is a possible observation (nonzero likelihood)
        found_obs = False
        for possible_obs, obs_prob in state_to_obs[state]:
            if possible_obs == obs:  # obs is a possible observation
                prob *= obs_prob  # Likelihood
                prob_sum += prob  # Prob. sum
                found_obs = True
                break
        # If obs was not found, that particle is removed (zero probability)
        if not found_obs:
            zero_prob_states.append(state)
    # Removes zero probability particles
    for state in zero_prob_states:
        del post_belief[state]

    # Normalizes the probabilities
    if prob_sum > 0.0:
        for state in post_belief:
            post_belief[state] /= prob_sum
    return post_belief


def copy_belief(belief):
    """
    Copies the necessary elements that compose a belief state
    """
    return {k: v for k, v in belief.items()}
