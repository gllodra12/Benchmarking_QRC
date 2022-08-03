from argparse import ArgumentParser
from collections import deque
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq

from benchmarking_qrc.hamiltonian import (
    get_coefficients,
    get_evolution_op,
    quadratic_op,
    quadratic_spin_op,
)
from benchmarking_qrc.measurements import get_features, observables
from benchmarking_qrc.reservoir import cptp_map, initial_state


REPOSITORY_ROOT = Path.cwd().parent.parent
STORE_DATA = REPOSITORY_ROOT / "data" / "memory_capacity"


def main(
    n_particles,
    dimensions,
    coefficients,
    operator,
    delay,
    obs_form,
    wash_time,
    train_time,
    test_time,
    excited_state,
    dt,
    degree=1,
):
    """It measures if the reservoir is able to remember the previous inputs with a certain delay. This ability
    to memorize previous outputs is called the memory capacity and it is bounded between 1 (remembers
    all previous inputs with delay d) and 0 (it doesn't remember anything).

    Using supervised learning we train a reservoir and obtain the best weights to perform the given task.
        Example: Given a the following sequence:
            input_sequence = [0.2, 0.3, 0.4, 0.5]
        if the delay=1 the target data would be:
            target_sequence = [0.1, 0.2, 0.3, 0.4]

    Once we have finish training we check if the model has overfitted the data by looking if the memory
    capacity in the test data (unseen data) is similar to the training data (seen data).

    Args:
        n_particles (int): Number of particles in the system
        dimensions (int): Number of particles on the same site (dim_boson=n, dim_fermion=2).
        coefficients (tuple(list, list)): Coefficients for the onsite elements (J_ii) and coupling elements (J_ij)
        operator (str): Type of particles in the reservoir (boson, fermion, spin)
        delay (int): Number of times we shift the target data respect to the input data.
        obs_form (str): Choose an observable to measure the reservoir. There are two observables "ii" (a^_i a_i) and "ij" (a^_i a_j)
        wash_time (int): Number of iterations to erase the reservoir initial conditions
        train_time (int): Number of iterations to train the reservoir.
        test_time (int): Number of iterations to validate if there is overfitting.
        excited_state (int): The input into the reservoir is linear combination of the ground state (|0>) an excited state (|n>)
        dt (int): Period of time the system is evolving.
        degree (int, optional): Degree assign to the input data. Linear task (d=1), non-linear task (d!=1). Defaults to 1.

    Returns:
        inputs (np.array): Input data
        (train_targets, test_targets) (np.array): Array with the targets
        (train_predictions, test_predictions) (np.array): Array with the predictions by the model
        data (np.array): Dataset computed after generating all the reservoirs and observables.
        opt_weights (np.array): Best weights for this dataset and these targets.
        training_mc (float): Memory capacity on the training data.
        test_mc (float): Memory capacity on the test data.
    """
    # Define reservoir initial state
    initial_reservoir = initial_state(n_particles, dimensions)

    # Define Hamiltonian
    if operator == "boson" or operator == "fermion":
        is_bosonic = True if operator == "boson" else False
        hamiltonian = quadratic_op(n_particles, is_bosonic, dimensions, coefficients)
    elif operator == "spin":
        hamiltonian = quadratic_spin_op(n_particles, coefficients)

    # Get evolution operator
    evolution_op = get_evolution_op(hamiltonian, dt)

    # The random sequence is determined with the seed of the function get_coefficients
    signals = np.random.uniform(low=0, high=1, size=wash_time)
    train_signals = np.random.uniform(low=0, high=1, size=train_time + delay)
    test_signals = np.random.uniform(low=0, high=1, size=test_time + delay)

    # Washing step: Forget initial conditions. Take the last reservoir once the washing time have been completed
    washed_reservoir = deque(
        cptp_map(signals, initial_reservoir, evolution_op, dimensions, excited_state), maxlen=1
    ).pop()

    # Prepare data
    train_inputs = get_inputs(train_signals, delay)
    test_inputs = get_inputs(test_signals, delay)
    train_targets = get_targets(train_signals, delay, degree)
    test_targets = get_targets(test_signals, delay, degree)
    inputs = np.concatenate((train_inputs, test_inputs))

    # Generate reservoirs
    reservoirs = cptp_map(inputs, washed_reservoir, evolution_op, dimensions, excited_state)

    # Get train and test dataset
    obs = observables(operator, n_particles, dimensions, obs_form)
    data, dm_snapshots = get_features(reservoirs, obs)
    train_data = data[:train_time, :]
    test_data = data[train_time:, :]

    # Obtain best weights (training process) and make predictions
    opt_weights, _, _, _ = lstsq(train_data, train_targets)
    train_predictions = np.matmul(train_data, opt_weights)
    test_predictions = np.matmul(test_data, opt_weights)

    # Evaluate model using the memory capacity
    train_mc = memory_capacity(train_predictions, train_targets)
    test_mc = memory_capacity(test_predictions, test_targets)

    return opt_weights, dm_snapshots, train_mc, test_mc


def get_targets(inputs, delay, degree):
    if degree == 1:
        return inputs[:-delay]
    else:
        return inputs[:-delay] ** degree


def get_inputs(signals, delay):
    return signals[delay:]


def memory_capacity(predictions, targets):
    """Measures if our system is able to remember previous injected signals.
    By definition the memory capacity is:
        MC = cov^2(x, y)/(var(x) * var(y))^2
    where cov is the covariance function and var is the variance.

    As you may have notice MC is just the correlation (a.k.a Pearson’s correlation coefficient)
    to the power of 2.
        corr = cov(x, y)/(var(x) * var(y))

    Args:
        predictions (np.array): Array of outputs from our system
        targets (np.array): Array with the values we want to predict.
            In the case we want to predict the initial signal with a delay t.

    Returns:
        Float: Value between 0 (low correlation) and 1 (great correlation)
    """
    return (np.corrcoef(predictions, targets)[0, 1]) ** 2


def save_linear_task(data, config, dataname, seed):
    path = STORE_DATA / config / dataname
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    filename = str(seed[0]) + ".npy" if len(seed) == 1 else f"range_{seed[0]}_{seed[1]}"
    path = path / filename
    np.save(str(path), data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("n_particles", help="Number of particles in your reservoir", type=int)
    parser.add_argument(
        "dimensions",
        help="Fermions have hilbert dimension 2, while bosons can have infinite dimension, so you must set a positive integer",
        type=int,
    )
    parser.add_argument(
        "operator",
        help="Choose which particles form the reservoirs: fermions, bosons or spins",
        type=str,
        choices=["fermion", "boson", "spin"],
    )
    parser.add_argument("delay", help="Number of times we shift the target data respect to the input data.", type=int)
    parser.add_argument(
        "obs_form",
        help="Choose an observable to measure the reservoir (a^_i a_i --> ii, a^_i a_j --> ij)",
        type=str,
        choices=["ii", "ij"],
    )
    parser.add_argument("wash_time", help="Number of iterations to erase the reservoir initial", type=int)
    parser.add_argument("train_time", help="Number of iterations to train the reservoir", type=int)
    parser.add_argument("test_time", help="Number of iterations to validate the train time", type=int)
    parser.add_argument("dt", help="Period of ime the system is evolving", type=float)
    parser.add_argument(
        "-s", "--seed", help="seed 1 100 generates, seed 3 generates a unique seed", type=int, nargs="+"
    )
    parser.add_argument(
        "-e",
        "--excited_state",
        help="The input into the reservoir is linear combination of the ground state (|0>) and excited states (|n>)",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-d", "--degree", help="Degree assign to the input data. Linear task (d=1), non-linear task (d!=1)", type=int
    )
    parser.add_argument(
        "-r",
        "--coef_range",
        help="Set all coefficient values J_ij at random, set min and max value",
        type=float,
        nargs="+",
    )
    parser.add_argument("-o", "--coef_onsite", help="Coefficient values on site (J_ii)", type=float, nargs="+")
    parser.add_argument("-c", "--coef_coupling", help="Coefficient values coupling site (J_ij)", type=float, nargs="+")
    args = parser.parse_args()

    if len(args.seed) > 2:
        raise ValueError("seed value must have length 1 or 2")
    seeds = range(args.seed[0], args.seed[1]) if len(args.seed) == 2 else args.seed

    all_train_mc, all_test_mc = list(), list()
    for seed in seeds:
        # Define the network structure (J_ij)
        coefficients = get_coefficients(args.n_particles, args.coef_range, args.coef_onsite, args.coef_coupling, seed)

        # Train Qreservoir to remember past inputs
        opt_weights, dm_snapshots, train_mc, test_mc = main(
            args.n_particles,
            args.dimensions,
            coefficients,
            args.operator,
            args.delay,
            args.obs_form,
            args.wash_time,
            args.train_time,
            args.test_time,
            args.excited_state,
            args.dt,
            degree=args.degree,
        )
        all_train_mc.append(train_mc)
        all_test_mc.append(test_mc)

    # Saving results
    config = "_".join(
        map(
            str,
            (
                args.n_particles,
                args.dimensions,
                args.operator,
                args.delay,
                args.obs_form,
                args.wash_time,
                args.train_time,
                args.test_time,
                args.excited_state,
                args.dt,
                args.degree,
            ),
        )
    )

    if args.coef_range:
        config = config + f"_uniform_({args.coef_range[0]}, {args.coef_range[1]})"
    if args.coef_onsite and args.coef_coupling:
        config = config + f"_custom_onsite={args.coef_onsite}_coupling={args.coef_coupling}"

    save_linear_task(all_train_mc, config, "train", args.seed)
    save_linear_task(all_test_mc, config, "test", args.seed)
