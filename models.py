import numpy as np

class RandomModel:

    def predict(*args):
        # return np.random.randint(0, 64, (2,), dtype=np.int32), 0
        return np.random.randint(0, 32 * 32, (1,), dtype=np.int32)[0], 0
