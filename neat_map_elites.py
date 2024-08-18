"""
NEAT Map-Elites Script for Hexapod Robot Gait Evolution

This script implements a NEAT (NeuroEvolution of Augmenting Topologies) algorithm
combined with the MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) approach
to evolve gaits for a hexapod robot.

Usage:
python script_name.py <map_size> <run_num> [--checkpoint CHECKPOINT] [--archive_load_file ARCHIVE_LOAD_FILE] [--start_index START_INDEX]
"""

import sys
import os
import pickle
import argparse
import numpy as np
import neat
import pymap_elites.map_elites.cvt as cvt_map_elites
import pymap_elites.map_elites.common as cm
from hexapod.controllers.NEATController import Controller, reshape, stationary
from hexapod.simulator import Simulator
from neat.reporting import ReporterSet

# Global configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'NEATHex/config-feedforward')

def evaluate_gait(x, duration=5):
    net = neat.nn.FeedForwardNetwork.create(x, config)
    leg_params = np.array(stationary).reshape(6, 5)
    try:
        controller = Controller(leg_params, body_height=0.15, velocity=0.5, period=1.0, crab_angle=-np.pi / 6, ann=net)
    except:
        return 0, np.zeros(6)
    simulator = Simulator(controller=controller, visualiser=False, collision_fatal=True)
    contact_sequence = np.full((6, 0), False)
    for t in np.arange(0, duration, step=simulator.dt):
        try:
            simulator.step()
        except RuntimeError as collision:
            return 0, np.zeros(6)
        contact_sequence = np.append(contact_sequence, simulator.supporting_legs().reshape(-1, 1), axis=1)
    fitness = simulator.base_pos()[0]  # distance travelled along x axis
    descriptor = np.nan_to_num(np.sum(contact_sequence, axis=1) / np.size(contact_sequence, axis=1), 
                               nan=0.0, posinf=0.0, neginf=0.0)
    simulator.terminate()
    x.fitness = fitness
    return fitness, descriptor

def load_genomes(num=200):
    reporters = ReporterSet()
    stagnation = config.stagnation_type(config.stagnation_config, reporters)
    reproduction = config.reproduction_type(config.reproduction_config, reporters, stagnation)
    genomes = reproduction.create_new(config.genome_type, config.genome_config, num)
    return list(genomes.values())

def load_checkpoint(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

def setup_output_directories(run_num, map_size):
    base_dir = f"mapElitesOutput/NEAT/{run_num}_{map_size}"
    archive_dir = f"{base_dir}archive"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    return base_dir, archive_dir

def main(args):
    base_dir, archive_dir = setup_output_directories(args.run_num, args.map_size)

    params = {
        "cvt_samples": 1000000,
        "batch_size": 2390,
        "random_init": 0.01,
        "random_init_batch": 2390,
        "dump_period": 1e6,
        "parallel": True,
        "cvt_use_cache": True,
        "min": 0,
        "max": 1,
    }

    if args.checkpoint:
        genomes = load_checkpoint(args.checkpoint)
        archive_load_file = args.archive_load_file
        start_index = args.start_index
    else:
        genomes = load_genomes(int(args.map_size * 0.01))
        archive_load_file = None
        start_index = 0

    log_file = open(f'{base_dir}/log.dat', 'a' if args.checkpoint else 'w')
    archive_file = f'{archive_dir}/archive'

    try:
        archive = cvt_map_elites.compute(
            6, genomes, evaluate_gait, n_niches=args.map_size, max_evals=10e6,
            log_file=log_file, archive_file=archive_file,
            archive_load_file=archive_load_file, params=params, start_index=start_index,
            variation_operator=cm.neatMutation
        )
    except Exception as e:
        print(f"Error during Map-Elites computation: {e}")
    finally:
        log_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NEAT Map-Elites Script for Hexapod Robot Gait Evolution")
    parser.add_argument('map_size', type=int, help="Size of the map to be tested")
    parser.add_argument('run_num', type=str, help="Run/map number")
    parser.add_argument('--checkpoint', type=str, help="Path to checkpoint file")
    parser.add_argument('--archive_load_file', type=str, help="Path to archive load file")
    parser.add_argument('--start_index', type=int, default=0, help="Starting index for computation")

    args = parser.parse_args()
    main(args)