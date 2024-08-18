"""
Run Hexapod Simulator Script

This script runs the hexapod simulator with a specified controller.
It allows for visualization and testing of individual controllers.

Usage:
python run_simulator.py <controller_file> [--duration DURATION] [--visualize]
"""

import argparse
import numpy as np
import neat
import pickle
from hexapod.simulator import Simulator
from hexapod.controllers.NEATController import Controller, reshape, stationary

def load_controller(filename):
    """
    Load a controller from a file.

    Args:
        filename (str): Path to the controller file.

    Returns:
        neat.nn.FeedForwardNetwork: Loaded neural network controller.
    """
    try:
        with open(filename, 'rb') as f:
            genome = pickle.load(f)
        
        # Load NEAT configuration
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             'NEATHex/config-feedforward')
        
        return neat.nn.FeedForwardNetwork.create(genome, config)
    except Exception as e:
        print(f"Error loading controller: {e}")
        return None

def run_simulation(controller, duration, visualize):
    """
    Run the hexapod simulation with the given controller.

    Args:
        controller (neat.nn.FeedForwardNetwork): Neural network controller.
        duration (float): Duration of the simulation in seconds.
        visualize (bool): Whether to visualize the simulation.

    Returns:
        float: Distance traveled by the hexapod.
    """
    leg_params = np.array(stationary).reshape(6, 5)
    hex_controller = Controller(leg_params, body_height=0.15, velocity=0.5, period=1.0, crab_angle=-np.pi/6, ann=controller)
    simulator = Simulator(controller=hex_controller, visualiser=visualize, collision_fatal=True)

    for t in np.arange(0, duration, step=simulator.dt):
        try:
            simulator.step()
        except RuntimeError as collision:
            print(f"Collision detected at time {t}")
            break

    distance = simulator.base_pos()[0]
    simulator.terminate()
    return distance

def main(args):
    controller = load_controller(args.controller_file)
    if controller is None:
        return

    distance = run_simulation(controller, args.duration, args.visualize)
    print(f"Simulation completed. Distance traveled: {distance:.2f} units")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hexapod Simulator with specified controller")
    parser.add_argument("controller_file", type=str, help="Path to the controller file")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration of the simulation in seconds")
    parser.add_argument("--visualize", action="store_true", help="Visualize the simulation")

    args = parser.parse_args()
    main(args)