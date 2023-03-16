import numpy as np
from typing import Tuple

def relu(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.maximum(Z, 0), Z