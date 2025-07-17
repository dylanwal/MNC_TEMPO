
import numpy as np


def get_const(H: float, O2: float, Fe_tot: float) -> np.ndarray:
    F = 9.64853 * 10 ** 4  # C⋅mol−1 or (A * s)/mol
    E_ORR = 1.23  # V (kg m**2/(s**3 A))
    R = 8.314  # kg * m**2/(s**2 K mol)
    T = 298  # 30 + 273.15  # K
    E1 = 0.62  # V
    E2 = 0.745  # V
    K1 = 1
    K2 = 4.57088e-8
    return np.array([F, E_ORR, R, T, Fe_tot, H, O2, K1, K2, E1, E2])
