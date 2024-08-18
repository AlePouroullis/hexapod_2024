import math
import sys
import neat
import numpy as np

""" The NEAT controller class. Responsible for managing the ANN queries at each time step
"""

stationary = [0.18, 0, 0, 0, 0] * 6


class Controller:

    def __init__(self, params=stationary, crab_angle=0.0, body_height=0.14, period=1.0, velocity=0.1, dt=1 / 240, ann = None, printangles = False, activations = 2):
        # link lengths
        self.count = 0
        self.activations = activations
        self.ann = ann
        self.l_1 = 0.05317
        self.l_2 = 0.10188
        self.l_3 = 0.14735

        self.dt = dt
        self.period = period
        self.velocity = velocity
        self.crab_angle = crab_angle
        self.body_height = body_height
        self.printangles = printangles

        self.array_dim = int(np.around(period / dt))

        self.positions = np.empty((0, self.array_dim))
        self.velocities = np.empty((0, self.array_dim))

        self.angles = np.empty((0, self.array_dim))
        self.speeds = np.empty((0, self.array_dim))

        params = np.array(params).reshape(6, 5)

        initial_angle = [0.0, 0.08994219, -1.48775765] * 6
        self.min = 1000
        self.max = -1000
        self.current_angle = initial_angle

    # Method to query artificial neural network to get the angles for the next time step.
    def joint_angles(self, t):

        # Feed Sin and Cos waves into the network
        sinewave = math.sin(t * 2 * math.pi / 3 * 16) * 2*math.pi
        coswave = math.cos(t * 2 * math.pi / 3 * 16) * 2*math.pi
        input_angles = np.append(self.current_angle, sinewave)
        input_angles = np.append(input_angles, coswave)
        # Activate Network
        current_angles = self.ann.activate(input_angles)
        # Scale outputs to the allowed angular range
        for i in range(len(current_angles)):
            if i % 3 == 0:
                current_angles[i] = (current_angles[i] * 0.91 * 2) - 0.91
            elif i % 3 == 1:
                current_angles[i] = (((current_angles[i] -0)*(0.64+0.2))/(1-0))-0.2
            else:
                current_angles[i] = (((current_angles[i] -0)*(-1.4+2.11))/(1-0))-2.11

        self.current_angle = current_angles

        return current_angles


    def IMU_feedback(self, measured_attitude):
        return


# reshapes a 32 length array of floats range 0.0 - 1.0 into the range expected by the controller
def reshape(x):
    x = np.array(x)
    # get body height and velocity
    height = x[0] * 0.2
    velocity = x[1] * 0.5
    leg_params = x[2:].reshape((6, 5))
    # radius, offset, step_height, phase, duty_cycle
    param_min = np.array([0.0, -1.745, 0.01, 0.0, 0.0])
    param_max = np.array([0.3, 1.745, 0.2, 1.0, 1.0])
    # scale and shifted params into the ranges expected by controller
    leg_params = leg_params * (param_max - param_min) + param_min

    return height, velocity, leg_params


if __name__ == '__main__':
    import time

    # Radius, offset, step, phase, duty_cycle
    leg_params = [
        0.1, 0, 0.1, 0.0, 0.5,  # leg 0
        0.1, 0, 0.1, 0.5, 0.5,  # leg 1
        0.1, 0, 0.1, 0.0, 0.5,  # leg 2
        0.1, 0, 0.1, 0.5, 0.5,  # leg 3
        0.1, 0, 0.1, 0.0, 0.5,  # leg 4
        0.1, 0, 0.1, 0.5, 0.5]  # leg 5

    start = time.perf_counter()
    ctrl = Controller(leg_params)
    end = time.perf_counter()

    print((end - start) * 1000)

