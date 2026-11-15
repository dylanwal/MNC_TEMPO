"""
This is the main solver to fit the data
"""

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
    data_set = expt_data.ALL_DATA_CUT
    for d in data_set:
        # 0.0000001 is used to avoid numerical issues of dividing by zero.
        y0 = np.array([
            d.TEMPO_p[0] if d.TEMPO_p[0] != 0 else 0.0000001,  # TEMPO_p
            d.TEMPO[0] if d.TEMPO[0] != 0 else 0.0000001,  # TEMPO
            d.TEMPOH2_p[0] if d.TEMPOH2_p[0] != 0 else 0.0000001,  # TEMPOH2
        ])
        y0_.append(y0)
        const_.append(get_const(d.H, d.O2, d.Fe))

    # Particle Swarm optimization
    bounds = np.array((
        (5e2, 1e3),  # k_dis
        (0.7, 0.8),  # E_Fe
        (5e-9, 1e-5),  # k_ORR
        (1e4, 5e6),  # k_THOR
        (1e-4, 1e-0),  # k_O2
        (1e-4, 1e-0),  # k_neg_O2
    ))
    best_k = particle_swarm_optimization(
        objective,
        2048*3,
        bounds.flatten(),
        args=(y0_, data_set, const_, expt_data.get_group_duplicate_datasets(data_set)),
        max_iter=100,
        logging_level=logging.DEBUG,
        w=0.2,
        c1=0.2,
        c2=0.1,
        multiprocess=12,
        pruning_distance=0.2
    )
    print(f"Optimized k: {','.join([str(i) for i in best_k])}")

# quick check solution
def compute_single_objective():
    # Load experimental data
    y0_ = []
    const_ = []
    data_set = expt_data.ALL_DATA_CUT
    for d in data_set:
        # 0.0000001 is used to avoid numerical issues of dividing by zero.
        y0 = np.array([
            d.TEMPO_p[0] if d.TEMPO_p[0] != 0 else 0.0000001,  # TEMPO_p
            d.TEMPO[0] if d.TEMPO[0] != 0 else 0.0000001,  # TEMPO
            d.TEMPOH2_p[0] if d.TEMPOH2_p[0] != 0 else 0.0000001,  # TEMPOH2
        ])
        y0_.append(y0)
        const_.append(get_const(d.H, d.O2, d.Fe))

    k = np.array([61.60915375092519,0.8756606010064557,1.2638850261453523e-09,74054.30509766466,0.0008738623595637976,1.5177594298879394e-07])

    bounds = np.array([(10**(np.log10(i)-3), 10**(np.log10(i)+3)) for i in k])
    from scipy.optimize import minimize
    res = minimize(objective, x0=k,
                   args=(y0_, data_set, const_, expt_data.get_group_duplicate_datasets(data_set)),
                   method='Nelder-Mead', bounds=bounds,
                   )
    print(res)
    best_k = res.x

    print(f"Optimized k: {','.join([str(i) for i in best_k])}")

    obj = objective(k, y0_, data_set, const_, expt_data.get_group_duplicate_datasets(data_set))
    print(obj)


if __name__ == '__main__':
    main()
    # compute_single_objective()

"""
best solutions:
k = np.array([61.60915375092519,0.8756606010064557,1.2638850261453523e-09,74054.30509766466,0.0008738623595637976,1.5177594298879394e-07])  # 9.781117112762855e-05
"""