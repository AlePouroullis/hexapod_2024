import numpy as np
import matplotlib.pyplot as plt

from hexapod.simulator import Simulator
from NEATController import Controller, stationary
from kinematic import tripod_gait
import matplotlib
import neat
import pickle

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

leg_n = 5
period = 0.5
duty_factor = 0.5

# radius, offset, step_height, phase, duty_cycle
tripod_gait = np.array([[0.15, 0, 0.06, 0, duty_factor],]*6)

fig, ax = plt.subplots()
fig.set_size_inches(w=4.7747, h=3.5)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     r'C:\Users\micha\PycharmProjects\Honours Project\NEATHex\config-feedforward')

with open(r"C:\Users\micha\PycharmProjects\Honours Project\NEATOutput\bestGenomes\NEATGenome1.pkl", 'rb') as f:
    winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

controller = Controller(tripod_gait, body_height=0.15, velocity=0.5, crab_angle=-np.pi / 6, ann=winner_net,
                        printangles=True)

simulator = Simulator(controller, follow=True, visualiser=False, collision_fatal=False, failed_legs=[0])
x = 0
joint_angles = np.empty((0))
while x < (240 * 5) - 1:
    simulator.step()
    current_array = simulator.controller.joint_angles(x / 240.0)
    joint_angles = np.append(joint_angles, current_array, axis=0)
    x += 1

    current_array = simulator.controller.joint_angles(x / 240.0)
x, y, z = controller.positions[(leg_n-1)*3:leg_n*3,:] * 1000
dx, dy, dz = controller.velocities[(leg_n-1)*3:leg_n*3,:] * 1000

shift = -1
dx = np.roll(dx, shift)
dy = np.roll(dy, shift)
dz = np.roll(dz, shift)

mid = int(period*240*duty_factor)
ax.scatter(y[mid:], z[mid:], label="Swing phase", marker='.')
ax.scatter(y[:mid], z[:mid], label="Support phase", marker='.')

# show velocity arrows
step = 12
q = ax.quiver(y[::step], z[::step], dy[::step], dz[::step], units="xy", angles="xy", pivot='tail')
# ax.quiverkey(q, X=0.3, Y=0.85, U=500, label="0.5 m/s foot velocity", labelpos='E', coordinates='figure')

ax.set_title("Foot Trajectory")
ax.set_xlabel('Y (mm)')
ax.set_ylabel('Z (mm)')

ax.set_ylim([-150,-40])
# ax.set_xlim([-150,30])

plt.legend()

fig.tight_layout()

plt.savefig('foot_traj_plot.pdf')

# plt.show()
