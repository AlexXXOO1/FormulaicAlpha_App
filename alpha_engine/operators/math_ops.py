from __future__ import annotations

import numpy as np
import pandas as pd


def signed_power(x: pd.Series | np.ndarray, power: float) -> np.ndarray:
    arr = np.asarray(x, dtype="float64")
    return np.sign(arr) * (np.abs(arr) ** power)
