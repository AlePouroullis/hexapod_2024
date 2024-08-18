"""
Map-Based Bayesian Optimisation (MBOA) for Hexapod Robot Control

This script implements Map-Based Bayesian Optimisation experiments for hexapod robot control.
It uses a pre-generated map of behaviors and their expected performances to guide the
optimization process, aiming to find the best controller for the robot.

The code is adapted from https://github.com/chrismailer/mailer_gecco_2021

Main components:
1. Data loading functions for centroids, map, and genomes
2. Upper Confidence Bound (UCB) acquisition function
3. MBOA algorithm implementation
4. Utility functions for Gaussian Process regression

Dependencies:
- numpy
- GPy
- hexapod (custom module for robot simulation)

Usage:
Run this script as the main program to execute the MBOA algorithm.
Modify the parameters in the __main__ section as needed.
"""

from copy import copy
import numpy as np
import GPy
from hexapod.controllers.NEATController import Controller, reshape, stationary
from hexapod.simulator import Simulator

def load_centroids(filename: str) -> np.ndarray:
    """
    Load CVT voronoi centroids from a file.

    Args:
        filename (str): Path to the file containing centroids.

    Returns:
        np.ndarray: Array of centroids.
    """
    return np.loadtxt(filename)

def load_map(filename: str, genomes_filename: str, dim: int = 6) -> tuple:
    """
    Load the generated map, including fitness, descriptors, and genomes.

    Args:
        filename (str): Path to the map file.
        genomes_filename (str): Path to the genomes file.
        dim (int): Dimension of the descriptor space.

    Returns:
        tuple: (fitness array, descriptor array, genomes array)
    """
    data = np.loadtxt(filename)
    fit = data[:, 0]
    desc = data[:, 1:dim + 1]
    genomes = np.load(genomes_filename, allow_pickle=True)
    return fit, desc, genomes

def UCB(mu_map: np.ndarray, kappa: float, sigma_map: np.ndarray) -> int:
    """
    Upper Confidence Bound acquisition function for Bayesian optimization.

    Args:
        mu_map (np.ndarray): Mean predictions.
        kappa (float): Exploration-exploitation trade-off parameter.
        sigma_map (np.ndarray): Standard deviation predictions.

    Returns:
        int: Index of the point with the highest UCB value.
    """
    GP = mu_map + kappa * sigma_map
    return np.argmax(GP)

def MBOA(map_filename: str, genomes_filename: str, centroids_filename: str, eval_func, 
         max_iter: int, rho: float = 0.4, print_output: bool = True) -> tuple:
    """
    Map-Based Bayesian Optimisation algorithm.

    Args:
        map_filename (str): Path to the map file.
        genomes_filename (str): Path to the genomes file.
        centroids_filename (str): Path to the centroids file.
        eval_func (callable): Function to evaluate real performance.
        max_iter (int): Maximum number of iterations.
        rho (float): Length scale for the GP kernel.
        print_output (bool): Whether to print progress information.

    Returns:
        tuple: (number of iterations, best index, best performance, updated map)
    """
    alpha = 0.90
    kappa = 0.05
    variance_noise_square = 0.001
    dim_x = 6

    num_it = 0
    real_perfs, tested_indexes = [-1], []
    X, Y = np.empty((0, dim_x)), np.empty((0, 1))

    centroids = load_centroids(centroids_filename)
    fits, descs, ctrls = load_map(map_filename, genomes_filename, centroids.shape[1])

    n_fits, n_descs, n_ctrls = np.array(fits), np.array(descs), np.array(ctrls)
    n_fits_real = copy(n_fits)
    fits_saved = copy(n_fits)

    started = False

    while (max(real_perfs) < alpha * max(n_fits_real)) and (num_it <= max_iter):
        if started:
            kernel = GPy.kern.Matern52(dim_x, lengthscale=rho, ARD=False) + GPy.kern.White(dim_x, np.sqrt(variance_noise_square))
            m = GPy.models.GPRegression(X, Y, kernel)
            means, variances = m.predict(n_descs)
            n_fits_real = means.flatten() + fits_saved
            index_to_test = UCB(n_fits_real, kappa, variances.flatten())
        else:
            index_to_test = np.argmax(n_fits)
            started = True
            real_perfs = []

        if print_output:
            print(f"Expected perf: {n_fits_real[index_to_test]}")

        if index_to_test in tested_indexes:
            if print_output:
                print("Behaviour already tested")
            break

        ctrl_to_test = n_ctrls[index_to_test]
        tested_indexes.append(index_to_test)

        real_perf = eval_func(ctrl_to_test)
        if print_output:
            print(f"Real perf: {real_perf}")

        num_it += 1

        X = np.vstack((X, n_descs[index_to_test]))
        Y = np.vstack((Y, np.array(real_perf) - fits_saved[index_to_test]))

        real_perfs.append(real_perf)

        new_map = np.loadtxt(map_filename)
        new_map[:, 0] = n_fits_real

    best_index = tested_indexes[np.argmax(real_perfs)]
    best_perf = max(real_perfs)

    return num_it, best_index, best_perf, new_map

if __name__ == "__main__":
    print("Running Map-Based Bayesian Optimisation")
    
    # Example usage (replace with actual file paths and evaluation function)
    map_file = "path/to/map.dat"
    genomes_file = "path/to/genomes.npy"
    centroids_file = "path/to/centroids.dat"
    
    def example_eval_func(ctrl):
        # Replace this with actual evaluation logic
        return np.random.rand()
    
    iterations, best_idx, best_performance, updated_map = MBOA(
        map_file, genomes_file, centroids_file, example_eval_func, 
        max_iter=100, rho=0.4, print_output=True
    )
    
    print(f"Optimization completed in {iterations} iterations")
    print(f"Best index: {best_idx}, Best performance: {best_performance}")
    
    # Optionally save the updated map
    # np.savetxt("path/to/updated_map.dat", updated_map)