import numpy as np
from typing import Tuple
from c import softmax
from d import relu
from b import linear_forward
def linear_activation_forward(A_prev, W, B, activation: str) -> Tuple[np.ndarray, dict]:
    activation_func = None
    if activation == "softmax":
        activation_func = softmax
    elif activation == "relu":
        activation_func = relu
    else:
        raise Exception('got unexpected value for "activation" argument - must be softmax or relu')
    output, cache = linear_forward(A=A_prev, W=W, b=B)
    output, activation_cache = activation_func(output)
    cache['activation'] = activation_cache
    return output, cache