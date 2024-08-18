import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from NEATController import Controller
from hexapod.simulator import Simulator
from kinematic import tripod_gait, stationary
import matplotlib
import pickle
import neat

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

leg_n = 3

fig = plt.figure()
fig.set_size_inches(w=4.7747, h=3.5)

ax = fig.add_subplot(111, projection='3d')

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


joint_angles = controller.angles[(leg_n-1)*3:leg_n*3,:] # leg 3
print(joint_angles)
x, y, z = controller.forward_kinematics(joint_angles) * 1000

ax.scatter(x[:60], y[:60], z[:60], label="Swing phase")
ax.scatter(x[60:], y[60:], z[60:], label="Support phase")

ax.set_title("Foot trajectory")
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')

# ax.set_xlim([-300,300])
# ax.set_ylim([-300,300])
# ax.set_zlim([-300,300])

plt.legend()

fig.tight_layout()

plt.savefig('NEAT Foot Trajectory.pdf')

plt.show()
