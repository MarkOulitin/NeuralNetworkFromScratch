import numpy as np
from typing import List
def initialize_parameters(layer_dims: List[int]) -> List[np.ndarray]:
    Ws: List[np.ndarray] = []
    Bs: List[np.ndarray] = []
    for input, output in zip(layer_dims, layer_dims[1:]):
        Ws.append(np.random.randn(input, output))
        Bs.append(np.random.randn(output))
    return Ws + Bs