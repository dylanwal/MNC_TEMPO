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
        y0 = np.array([
            d.TEMPO_p[0] if d.TEMPO_p[0] != 0 else 0.0000001,  # TEMPO_p
            d.TEMPO[0] if d.TEMPO[0] != 0 else 0.0000001,  # TEMPO
            d.TEMPOH2_p[0] if d.TEMPOH2_p[0] != 0 else 0.0000001,  # TEMPOH2
        ])
        y0_.append(y0)
        const_.append(get_const(d.H, d.O2, d.Fe))

    # Particle Swarm optimization
    bounds = np.array((
        (1e1, 1e3),  # k_dis
        (0.8, 0.9),  # E_Fe
        (5e-11, 1e-7),  # k_ORR
        (1e4, 1e6),  # k_THOR
        (1e-5, 1e-2),  # k_O2
        (1e-5, 1e-2),  # k_neg_O2
    ))
    best_k = particle_swarm_optimization(
        objective,
        2048*2,
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


def compute_single_objective():
    # Load experimental data
    y0_ = []
    const_ = []
    data_set = expt_data.ALL_DATA_CUT
    for d in data_set:
        y0 = np.array([
            d.TEMPO_p[0] if d.TEMPO_p[0] != 0 else 0.0000001,  # TEMPO_p
            d.TEMPO[0] if d.TEMPO[0] != 0 else 0.0000001,  # TEMPO
            d.TEMPOH2_p[0] if d.TEMPOH2_p[0] != 0 else 0.0000001,  # TEMPOH2
        ])
        y0_.append(y0)
        const_.append(get_const(d.H, d.O2, d.Fe))

    k = np.array([971.346830197743,0.5802317286184792,8.158086014865025e-05,10000000000.0,71.71640322492323, 63.23939458995973]) # start at zero
    # k = np.array([7.74104752207566, 0.862891173493384, 8.85116656134E-09, 58183.0599974819, 0.000962922800758801, 0.00022852474646638]) # excel
    k = np.array([52.91344677312251,0.7014163982435674,8.80918581071844e-09,68262.87681890542,0.32405367754300635,0.0001])
    obj = objective(k, y0_, data_set, const_, expt_data.get_group_duplicate_datasets(data_set))
    print(obj)


if __name__ == '__main__':
    main()
    # compute_single_objective()

"""
34 | 2.983063e-04 | 577 | [7.15655131e+04 5.00000000e-01 2.00000000e-02 2.26389598e+12
 1.65840383e+03 7.71845393e+03] | 90
29 | 2.636776e-04 | 4056 | [269.7759001037147,0.6,1e-07,26298193.292025592,26.32339541331292,0.18040851998315197] | 36
0.000115
99 | 1.135158e-04 | 6540 | [5.29134468e+01 7.01416398e-01 8.80918581e-09 6.82628768e+04
 3.24053678e-01 1.00000000e-04] | 5953
64 | 1.093471e-04 | 2289 | [7.24263617e+01 6.83645795e-01 4.80225199e-09 6.21926671e+04
 6.49854712e-01 1.00000000e-05] | 6834
 
 
 26 | 1.282962e-04 | 92 | [6.31837882e+01 8.92337973e-01 4.29224476e-09 1.69630144e+05
 6.71898641e-04 3.91925481e-04] | 3376
"""