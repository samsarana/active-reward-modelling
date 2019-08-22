import numpy as np

class RunningStat:
    """
    https://github.com/joschu/modular_rl/blob/master/modular_rl/running_stat.py
    http://www.johndcook.com/blog/standard_deviation/
    """
    def __init__(self, shape=()):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM)/self._n
            self._S[...] = self._S + (x - oldM)*(x - self._M)
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M.item() # sam modified this to return python number, to be consistent with var()
    @property
    def var(self):
        return self._S/(self._n - 1) if self._n > 1 else np.square(self._M)
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape


class TrueRewardRunningStat(RunningStat): # TODO make a subclass like this for reward model case
    def __init__(self, shape=()):
        super().__init__()

    def push_clip_pairs(self, clip_data):
        _, rews, _ = clip_data
        for rew in rews.reshape(-1):
            self.push(rew)