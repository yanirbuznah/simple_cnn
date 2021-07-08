import time
from enum import Enum

import numpy as np


class ActivationFunction(object):
    def __init__(self, f, d):
        self.f = f
        self.d = d


ActivationFunction.ReLU = ActivationFunction(lambda a: np.maximum(0, a), lambda a: (a > 0).astype(int))
ActivationFunction.Sigmoid = ActivationFunction(lambda a: 1/(1+np.exp(-a)), lambda a: a * (1-a))


class AdaptiveLearningRateMode(Enum):
    PREDEFINED_DICT = 1
    FORMULA = 2


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            pass
            # if (te - ts) * 1000 > 10:
            #     print('%r  %2.2f ms' % \
            #           (method.__qualname__, (te - ts) * 1000))
        return result
    return timed
