# Hexapod Robot Gait Evolution Project

This project implements a NEAT (NeuroEvolution of Augmenting Topologies) algorithm combined with the MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) approach to evolve gaits for a hexapod robot. It includes scripts for adaptation and MAP-Elites-based evolution.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setting Up the Virtual Environment](#setting-up-the-virtual-environment)
3. [Project Structure](#project-structure)
4. [Adaptation Script](#adaptation-script)
5. [MAP-Elites Script](#map-elites-script)
6. [Running the Scripts](#running-the-scripts)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- virtualenv or venv (for creating virtual environments)

## Setting Up the Virtual Environment

1. Clone the repository:
   ```
   git clone https://github.com/your-username/hexapod-gait-evolution.git
   cd hexapod-gait-evolution
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

```
hexapod-gait-evolution/
├── adaptation_script.py
├── map_elites_script.py
├── hexapod/
│   ├── controllers/
│   │   └── NEATController.py
│   └── simulator.py
├── NEATHex/
│   └── config-feedforward
├── requirements.txt
└── README.md
```

## Adaptation Script

The adaptation script (`adaptation_script.py`) implements Map-Based Bayesian Optimisation (MBOA) for hexapod robot control. It uses a pre-generated map of behaviors and their expected performances to guide the optimization process, aiming to find the best controller for the robot.

Key features:
- Loads centroids, map, and genomes from files
- Implements Upper Confidence Bound (UCB) acquisition function
- Performs MBOA algorithm
- Uses Gaussian Process regression for performance prediction

## MAP-Elites Script

The MAP-Elites script (`map_elites_script.py`) combines NEAT with the MAP-Elites approach to evolve diverse and high-performing gaits for the hexapod robot.

Key features:
- Customizable map size and run number
- Checkpoint and archive loading capabilities
- Parallel computation support
- Robust error handling and logging

## Running the Scripts

### Adaptation Script

To run the adaptation script:

```
python adaptation_script.py
```

Note: You may need to modify the script to specify the correct file paths and evaluation function.

### MAP-Elites Script

To run the MAP-Elites script:

```
python map_elites_script.py <map_size> <run_num> [--checkpoint CHECKPOINT] [--archive_load_file ARCHIVE_LOAD_FILE] [--start_index START_INDEX]
```

Arguments:
- `map_size`: Size of the map to be tested
- `run_num`: Run/map number
- `--checkpoint`: (Optional) Path to checkpoint file
- `--archive_load_file`: (Optional) Path to archive load file
- `--start_index`: (Optional) Starting index for computation (default: 0)

Example:
```
python map_elites_script.py 1000 1 --checkpoint checkpoints/checkpoint_1000.pkl
```

## Troubleshooting

If you encounter any issues:

1. Ensure your virtual environment is activated.
2. Verify that all required packages are installed correctly.
3. Check that file paths in the scripts are correct for your system.
4. If using GPUs, ensure CUDA is properly set up (if applicable).

For more detailed error messages, you can modify the scripts to include additional logging.

If problems persist, please open an issue on the GitHub repository with a detailed description of the error and your system configuration.