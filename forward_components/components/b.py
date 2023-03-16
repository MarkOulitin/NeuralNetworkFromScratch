import numpy as np
from typing import Tuple

def linear_forward(A: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, dict[str, np.ndarray]]:
    output = np.matmul(A, W) + b
    linear_cache = {
        "A": A,
        "W": W,
        "b": b,
    }
    return output, linear_cache