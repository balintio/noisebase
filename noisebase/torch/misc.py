"""
noisebase.torch.misc
--------------------

also available under noisebase.torch
"""

import torch
import numpy as np

def tensor_like(like, data):
    return torch.tensor(data, dtype=like.dtype, device=like.device)

def lrange(start, stop, step=1):
    return list(range(start, stop, step))

class Shuffler():
    """Numpy rng convenience
    """
    def __init__(self, seed):
        rng = np.random.default_rng(seed = seed)
        self.seed = rng.integers(2**31)
    
    def shuffle(self, seed, sequence):
        rng = np.random.default_rng(seed = self.seed + seed)
        rng.shuffle(sequence)
    
    def split(self, seed, sequence, split, shuffled = True, smallest = 0):
        if shuffled:
            self.shuffle(seed, sequence)
        idx = max(round(len(sequence) * split), smallest)
        return sequence[:idx], sequence[idx:]
    
    def integers(self, seed, *args):
        rng = np.random.default_rng(seed = self.seed + seed)
        return rng.integers(*args)

    def derive(self, seed):
        return Shuffler(self.seed + seed)