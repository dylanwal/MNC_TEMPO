import logging

import numpy as np
from scipy.integrate import solve_ivp

from particle_swarm import particle_swarm_optimization
from constants import get_const
import expt_data.data as expt_data
from ode_function import odefunc


def objective(
        k: np.ndarray,
        y0: list[np.ndarray],
        data: list[expt_data.Data],
        const: list[np.ndarray],
        group_duplicate_datasets: list[list[int]]
) -> float:
    """ Average duplicated data sets"""
    error = 0
    for indexes in group_duplicate_datasets:
        d = data[indexes[0]]
        y0_ = y0[indexes[0]]
        c_ = const[indexes[0]]
        try:
            sol = solve_ivp(odefunc, d.t_span, y0_, args=(k, c_), t_eval=d.time, method='Radau')

        # many of the k values result inf.; so just penalize the optimizer to avoid these
        except ValueError:
            error += 1
            continue
        if sol.success is False:
            # print(sol.message)
            error += 1
            continue

        # calculate error for each set and average
        err = 0
        for i in indexes:
            d = data[i]
            err += np.sum((sol.y[0] - d.TEMPO_p) ** 2 + (sol.y[1] - d.TEMPO) ** 2 + (sol.y[2] - d.TEMPOH2_p) ** 2)

        error += err / len(indexes)

    return error


def main():
    # Load experimental data
    y0_ = []
    const_ = []
    for d in expt_data.ALL_DATA:
        y0 = np.array([
            d.TEMPO_p[0] if d.TEMPO_p[0] != 0 else 0.0000001,  # TEMPO_p
            d.TEMPO[0] if d.TEMPO[0] != 0 else 0.0000001,  # TEMPO
            d.TEMPOH2_p[0] if d.TEMPOH2_p[0] != 0 else 0.0000001,  # TEMPOH2
        ])
        y0_.append(y0)
        const_.append(get_const(d.H, d.O2, d.Fe))

    # Particle Swarm optimization
    bounds = np.array((
        (8e0, 1.2e5),  # k_dis
        (0.5, 1),  # E_Fe
        (1e-9, 2e-2),  # k_ORR
        (1e7, 1e13),  # k_THOR
        (1e-3, 1e4),  # k_O2
        (1e-3, 1e4),  # k_neg_O2
    ))
    best_k = particle_swarm_optimization(
        objective,
        4096*4,
        bounds.flatten(),
        args=(y0_, expt_data.ALL_DATA, const_, expt_data.get_group_duplicate_datasets(expt_data.ALL_DATA)),
        max_iter=100,
        logging_level=logging.DEBUG,
        c1=0.7,
        c2=0.2,
        multiprocess=12,
        pruning_distance=1
    )
    print(f"Optimized k: {','.join([str(i) for i in best_k])}")



def compute_single_objective():
    # Load experimental data
    y0_ = []
    const_ = []
    for d in expt_data.ALL_DATA:
        y0 = np.array([
            d.TEMPO_p[0] if d.TEMPO_p[0] != 0 else 0.0000001,  # TEMPO_p
            d.TEMPO[0] if d.TEMPO[0] != 0 else 0.0000001,  # TEMPO
            d.TEMPOH2_p[0] if d.TEMPOH2_p[0] != 0 else 0.0000001,  # TEMPOH2
        ])
        y0_.append(y0)
        const_.append(get_const(d.H, d.O2, d.Fe))

    k = np.array([971.346830197743,0.5802317286184792,8.158086014865025e-05,10000000000.0,71.71640322492323, 63.23939458995973])
    obj = objective(k, y0_, expt_data.ALL_DATA, const_, expt_data.get_group_duplicate_datasets(expt_data.ALL_DATA))
    print(obj)


if __name__ == '__main__':
    main()
    # compute_single_objective()
