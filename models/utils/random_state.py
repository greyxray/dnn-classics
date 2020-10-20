import random
import numpy as np
import torch


class RandomState(object):

    def __init__(self, seed=137):

        self.SEED = seed

    def fix(self, source=['random', 'np', 'torch'], seed=None):

        if seed is not None:
            self.SEED = seed

        if 'random' in source:
            random.seed(self.SEED)

        if 'np' in source:
            np.random.seed(self.SEED)

        if 'torch' in source:
            torch.manual_seed(self.SEED)
            torch.cuda.manual_seed(self.SEED)
            torch.backends.cudnn.deterministic = True
