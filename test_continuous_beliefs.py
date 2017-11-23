
from belief import *

a1 = 2
bb1 = -2
a2 = 1
bb2 = 2
a3 = -4
bb3 = 27
a4 = -1 / 3
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

b0 = ContinuousBeliefState(0, 0, 0, 0, 0)
obs0 = ContinuousBeliefState(1, 0, 0, 0, 0)
obs1 = ContinuousBeliefState(1.5, 0, 0, 0, 0)

print('risk0:', dynamic_obs_risk(b0, obs0))
print('risk1:', dynamic_obs_risk(b0, obs1))

# raise ValueError()

# % % <0
# R1 = 1 - risk(a1, b1, m1, p1)
# print(1 - one_line_risk(a1, b1, m1, p1))


# values computed from obstacle vertices
x1 = 3
y1 = 4
x2 = 4
y2 = 6

x = np.matrix([[x1, x2]])
y = np.matrix([[y1, y2]])
print('x', x)
print('y', y)
c1 = np.matrix([[1, x[0, 0]], [1, x[0, 1]]])
c = np.linalg.solve(c1, y.T)
a1 = c[1]
b1 = c[0]
# print(1 - one_line_risk(a1, b1, m1, p1))

# values computed from obstacle vertices
x1 = 4
y1 = 6
x2 = 3
y2 = 4

x = np.matrix([[x1, x2]])
y = np.matrix([[y1, y2]])
print('x', x)
print('y', y)
c1 = np.matrix([[1, x[0, 0]], [1, x[0, 1]]])
c = np.linalg.solve(c1, y.T)
a1 = c[1]
b1 = c[0]
# print(1 - one_line_risk(a1, b1, m1, p1))

b0 = ContinuousBeliefState(0, 0, 0, 0, 0)
b1 = cont_belief_update(b0, [[1], [1]])
b2 = cont_belief_update(b1, [[1], [1]])
b3 = cont_belief_update(b2, [[1], [1]])
b4 = cont_belief_update(b3, [[0.5], [0]])
b5 = cont_belief_update(b4, [[0.1], [0]])
b6 = cont_belief_update(b4, [[0.5], [0]])
b7 = cont_belief_update(b4, [[1.2], [0]])

# print('b0 risk', static_obs_risk(b0))
# print('b1 risk', static_obs_risk(b1))
# print('b2 risk', static_obs_risk(b2))
# print('b3 risk', static_obs_risk(b3))
# print('b4 risk', static_obs_risk(b4))
# print('b5 risk', static_obs_risk(b5))
# print('b6 risk', static_obs_risk(b6))
print('b7 risk', static_obs_risk(b7))


print(b1.mean_b)
print(b0.mean_b)

# fig, ax = plt.subplots()  # note we must use plt.subplots, not plt.subplot
fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')

# (or if you have an existing figure)
# fig = plt.gcf()
# ax = fig.gca()
plot_belief_state(ax, b0)
plot_belief_state(ax, b1)
plot_belief_state(ax, b2)
plot_belief_state(ax, b3)
plot_belief_state(ax, b4)
plot_belief_state(ax, b5, (0, 1, 0, 0.5))
plot_belief_state(ax, b6)
plot_belief_state(ax, b7, (0, 0, 1, 0.5))
# print('b0', b0.mean_b, b0.sigma_b)
# print('b1', b1.mean_b, b1.sigma_b)
# print('b5', b5.mean_b, b5.sigma_b)
# print('b6', b6.mean_b, b6.sigma_b)
print('b7', b7.mean_b, b7.sigma_b)

ax.set_xlim(-1, 7)
ax.set_ylim(-1, 7)

x1 = 3
y1 = 4
x2 = 4
y2 = 6
x3 = 5
y3 = 7
x4 = 6
y4 = 3
lines = LineCollection([[(3, 4), (4, 6)], [(4, 6), (5, 7)], [
    (5, 7), (6, 4)], [(6, 4), (3, 4)]])
ax.add_collection(lines)

# ax.add_artist(circle1)
# ax.add_artist(circle2)
# ax.add_artist(circle3)
plt.show()

# fig.savefig('plotcircles.png')
