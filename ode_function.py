
import numpy as np
from numba import njit


@njit
def odefunc(t: np.ndarray, y: np.ndarray, k: np.ndarray, const: np.ndarray) -> np.ndarray:
    F = const[0]
    E_ORR = const[1]
    R = const[2]
    T = const[3]
    Fe_tot = const[4]
    H = const[5]
    O2 = const[6]
    K1 = const[7]
    K2 = const[8]
    E1 = const[9]
    E2 = const[10]

    k_dis = k[0]
    E_Fe = k[1]
    k_ORR = k[2]
    k_THOR = k[3]
    k_O2 = k[4]
    k_neg_O2 = k[5]

    TEMPO_p = y[0]
    TEMPO = y[1]
    TEMPOH2_p = y[2]

    TEMPOH = TEMPOH2_p * K2 / (K2 + H)

    P = k_ORR * np.exp(-.5 * F * (E2 - E_ORR) / (R * T))
    Q = np.exp(.5 * F * (E2 - E1) / (R * T))

    # eq 11 of manuscript
    FeO2 = k_O2 * O2 * Fe_tot / (
            (k_neg_O2 + np.sqrt(TEMPO / TEMPO_p) * P + k_THOR * TEMPOH) *
            (1 + H ** -1 * TEMPO_p / TEMPO * np.exp(F * (E2 - E_Fe) / (R * T))) +
            k_O2 * O2
    )

    j_orr = -4 * FeO2 * P * np.sqrt(TEMPO / TEMPO_p)  # no F

    # eq 8 of manuscript
    r_TEMPO_THOR = k_THOR * TEMPOH * FeO2

    r_TEMPOH2_p = -Fe_tot * k_dis * (
            (TEMPOH * np.sqrt(TEMPO_p / TEMPO) * Q) - (H * TEMPO * np.sqrt(TEMPO / TEMPO_p) / (K1 * Q))
    ) - r_TEMPO_THOR
    r_TEMPO_P = r_TEMPOH2_p + r_TEMPO_THOR - j_orr
    r_TEMPO = j_orr - r_TEMPO_THOR - 2 * r_TEMPOH2_p

    return np.array([r_TEMPO_P, r_TEMPO, r_TEMPOH2_p])
