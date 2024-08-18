"""
NEAT Map-Elites Script for Hexapod Robot Gait Evolution

This script implements a NEAT (NeuroEvolution of Augmenting Topologies) algorithm
combined with the MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) approach
to evolve gaits for a hexapod robot.

The script uses the following main components:
1. NEAT for evolving neural network controllers
2. MAP-Elites for maintaining a diverse archive of high-performing solutions
3. A hexapod robot simulator for evaluating gaits

Key features:
- Customizable map size and run number
- Checkpoint and archive loading capabilities
- Parallel computation support
- Robust error handling and logging

Dependencies:
- neat-python
- numpy
- pymap_elites
- Custom hexapod simulator and controller modules

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

def load_config(config_path='NEATHex/config-feedforward'):
    """
    Load the NEAT configuration from a file.

    Args:
        config_path (str): Path to the NEAT configuration file.

    Returns:
        neat.Config: Loaded NEAT configuration object.

    Raises:
        SystemExit: If there's an error loading the configuration.
    """
    try:
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_path)
    except Exception as e:
        print(f"Error loading NEAT config: {e}")
        sys.exit(1)

def evaluate_gait(x, duration=5):
    """
    Evaluate the fitness of a given genome by simulating the hexapod gait.

    Args:
        x (neat.genome.DefaultGenome): The genome to evaluate.
        duration (float): Duration of the simulation in seconds.

    Returns:
        tuple: A tuple containing the fitness (float) and descriptor (numpy.ndarray).

    Note:
        This function uses a global 'config' variable for the NEAT configuration.
    """
    try:
        # Create neural network from genome
        net = neat.nn.FeedForwardNetwork.create(x, config)
        leg_params = np.array(stationary).reshape(6, 5)
        
        # Set up controller and simulator
        controller = Controller(leg_params, body_height=0.15, velocity=0.5, period=1.0, crab_angle=-np.pi / 6, ann=net)
        simulator = Simulator(controller=controller, visualiser=False, collision_fatal=True)
        
        # Run simulation
        contact_sequence = np.full((6, 0), False)
        for t in np.arange(0, duration, step=simulator.dt):
            try:
                simulator.step()
                contact_sequence = np.append(contact_sequence, simulator.supporting_legs().reshape(-1, 1), axis=1)
            except RuntimeError:
                return 0, np.zeros(6)
        
        # Calculate fitness and descriptor
        fitness = simulator.base_pos()[0]  # Distance traveled along x-axis
        descriptor = np.nan_to_num(np.sum(contact_sequence, axis=1) / np.size(contact_sequence, axis=1),
                                   nan=0.0, posinf=0.0, neginf=0.0)
        x.fitness = fitness
        return fitness, descriptor
    except Exception as e:
        print(f"Error in evaluate_gait: {e}")
        return 0, np.zeros(6)
    finally:
        if 'simulator' in locals():
            simulator.terminate()

def load_genomes(config, num=200):
    """
    Load initial genomes for the NEAT algorithm.

    Args:
        config (neat.Config): NEAT configuration object.
        num (int): Number of genomes to create.

    Returns:
        list: List of created genomes.

    Raises:
        SystemExit: If there's an error loading the genomes.
    """
    try:
        reporters = ReporterSet()
        stagnation = config.stagnation_type(config.stagnation_config, reporters)
        reproduction = config.reproduction_type(config.reproduction_config, reporters, stagnation)
        genomes = reproduction.create_new(config.genome_type, config.genome_config, num)
        return list(genomes.values())
    except Exception as e:
        print(f"Error loading genomes: {e}")
        sys.exit(1)

def load_checkpoint(filename):
    """
    Load a checkpoint file containing saved genomes.

    Args:
        filename (str): Path to the checkpoint file.

    Returns:
        list: List of loaded genomes.

    Raises:
        SystemExit: If there's an error loading the checkpoint.
    """
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

def setup_output_directories(run_num, map_size):
    """
    Set up output directories for storing results.

    Args:
        run_num (str): Run/map number.
        map_size (int): Size of the map.

    Returns:
        tuple: A tuple containing paths to the base directory and archive directory.
    """
    base_dir = f"mapElitesOutput/NEAT/{run_num}_{map_size}"
    archive_dir = f"{base_dir}archive"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    return base_dir, archive_dir

def main(args):
    """
    Main function to run the NEAT Map-Elites algorithm.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    global config
    config = load_config()

    base_dir, archive_dir = setup_output_directories(args.run_num, args.map_size)

    # MAP-Elites parameters
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

    # Load genomes from checkpoint or create new ones
    if args.checkpoint:
        genomes = load_checkpoint(args.checkpoint)
        archive_load_file = args.archive_load_file
        start_index = args.start_index
    else:
        genomes = load_genomes(config, int(args.map_size * 0.01))
        archive_load_file = None
        start_index = 0

    # Set up logging
    log_file = open(f'{base_dir}/log.dat', 'a' if args.checkpoint else 'w')
    archive_file = f'{archive_dir}/archive'

    try:
        # Run MAP-Elites algorithm
        cvt_map_elites.compute(
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