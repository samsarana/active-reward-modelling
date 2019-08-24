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
    def __init__(self, shape=()): # TODO pretty sure this function is redundant
        super().__init__()

    def push_clip_pairs(self, clip_data):
        _, rews, _ = clip_data
        for rew in rews.reshape(-1):
            self.push(rew)


class LinearSchedule(object):
    """Copy-pasted from:
       https://github.com/openai/baselines/blob/master/baselines/common/schedules.py
    """
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class ExpSchedule(object):
    def __init__(self, decay_rate, final_p, initial_p=1.0):
        """Exponential decay every time .step() is called
        Parameters
        ----------
        decay_rate: float
            exponential decay rate
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.epsilon = initial_p
        self.final_p = final_p
        self.decay_rate = decay_rate

    def value(self, t=None):
        """Takes `t`, a timestep, just to have identical
           signature to LinearSchedule.value
        """ 
        return self.epsilon

    def step(self):
        if self.epsilon > self.final_p:
            self.epsilon *= self.decay_rate