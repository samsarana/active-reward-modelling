import numpy as np
from collections import deque
import gym
from gym import spaces

def preprocess_atari_env(env):
    """Does the same preprocessing as in
       Minh, Christiano, Ibarz etc.
       TODO still need to implement replacing 
       score with a constant black background.
       Strangely, this doesn't seem to be implemented in
       github.com/machine-intelligence/rl-teacher-atari...
    """
    env = gym.wrappers.AtariPreprocessing(env, noop_max=90)
    env = FrameStack(env, k=4)
    return env


class FrameStack(gym.Wrapper):
    """This code copied from: tinyurl.com/y5779rke
       Source: github.com/machine-intelligence/rl-teacher-atari
       ----------
       Why is this necessary on top of gym.wrappers.AtariPreprocessing?
       Strangely, it seems that gym.wrappers.AtariPreprocessing implements
       (by default) the two "most common" preprocessing steps mentioned in
       pp.3-4 of Machado et al., (2017) but not frame stacking.
       Also, whilst they claim to have "followed the guidelines" that Machado
       set, and indeed they do so for Episode termination (see pp.5-7) contra
       Minh et al. who terminate epusdode when life is lost, they don't follow
       the recommendation in Injecting stochasticity.
       Machado: sticky actions (as in -v0 of Atari games)
       gym.wrappers.AtariPreprocessing: noop_max=30 (which might be in addition
       to using -v0 of the Atari game with sticky actions with eta=0.25)
       Gym default: random frame skips, Brockman et al., 2016 (unless you specify
       Deterministic or NoFrameSkip)
    """
    def __init__(self, env, k):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        assert len(shp) == 2 or shp[2] == 1  # can only stack 1-channel frames
        self.observation_space = spaces.Box(low=0, high=255, shape=(k, shp[0], shp[1])) # CHANGED FROM (shp[0], shp[1], k)

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._observation()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        if len(self.frames[0].shape) == 2:
            return np.stack(self.frames, axis=0) # CHANGED FROM AXIS=2
        else:
            return np.concatenate(self.frames, axis=0) # CHANGED FROM AXIS=2. SEEMS BAD!!! work rationally, from the bottom up...


class DiscreteToBox(gym.Wrapper):
    """Takes env with
       env.observation_space = Box
       and wraps it to have observation_space Discrete
       with shape (1,)
    """
    def __init__(self, env):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        n = env.observation_space.n
        self.observation_space = spaces.Box(low=0, high=n-1, shape=(1,))

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        return np.array([ob])

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        return np.array([ob]), reward, done, info