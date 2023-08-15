from typing import List, Union

import numpy as np
import random
from time import time
import math
from dotmap import DotMap

from src.utils.utils import random_start, one_hot_to_dec


def unbalance_sa_solve(
    qubo: np.ndarray,
    n_iter: int = 1,
    n_temp_iter: int = 1000,
    temp: float = 20,
    warm_start: Union[List, np.ndarray] = None,
) -> DotMap:
    """
    Custom simulated annealing solver for the unbalance problem
    Mutation only in valid solution space

    :param qubo: qubo to solve
    :param n_iter: number of runs
    :param n_temp_iter: number of mutations
    :param temp: starting temperature
    :param warm_start: warm start solution vector
    :return: solution with samples, energies and times
    """
    samples = []
    energies = []
    n_blades = int(math.sqrt(len(qubo)))
    indices = list(range(0, n_blades))

    # evaluate energy of vector based on qubo
    f = lambda x: (x.T @ qubo @ x)
    start_time = time()

    for _ in range(n_iter):

        # define start vector
        if warm_start is None:
            x, curr_dec = random_start(n_blades)
        else:
            x = np.array(warm_start)
            curr_dec = one_hot_to_dec(warm_start, n_blades)

        # evaluate start vector
        curr, curr_eval = x, f(x)
        best, best_eval = curr, curr_eval

        for i in range(n_temp_iter):

            # randomly switch two blades
            candidate = np.copy(curr)
            candidate_dec = curr_dec.copy()
            swap_pos = random.sample(indices, 2)

            candidate[swap_pos[0] * n_blades + candidate_dec[swap_pos[0]]] = 0
            candidate[swap_pos[0] * n_blades + candidate_dec[swap_pos[1]]] = 1

            candidate[swap_pos[1] * n_blades + candidate_dec[swap_pos[1]]] = 0
            candidate[swap_pos[1] * n_blades + candidate_dec[swap_pos[0]]] = 1

            candidate_dec[swap_pos[0]], candidate_dec[swap_pos[1]] = (
                candidate_dec[swap_pos[1]],
                candidate_dec[swap_pos[0]],
            )

            # evaluate new vector
            candidate_eval = f(candidate)

            # keep best vector
            if candidate_eval <= best_eval:
                best, best_eval = candidate, candidate_eval

            # update temperature according to Metropolis–Hastings algorithm
            diff = candidate_eval - curr_eval
            t = temp / float(i + 1)

            metropolis_eval = math.exp(-diff / t)

            # base new mutations on new vector
            if diff <= 0 or random.random() < metropolis_eval:
                curr, curr_eval, curr_dec = candidate, candidate_eval, candidate_dec

        samples.append(best.tolist())
        energies.append(best_eval)

    runtime_sa = time() - start_time
    solution = DotMap(
        solver="unbalance_sa",
        samples=samples,
        energies=energies,
        runtime=DotMap(total=runtime_sa, qpu=None),
    )

    return solution
