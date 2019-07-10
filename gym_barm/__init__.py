from gym.envs.registration import register

register(
    id='CartPoleContinuous-v0',
    entry_point='gym_barm.envs:Continuous_CartPoleEnv',
    max_episode_steps=float('inf'),
    reward_threshold=195.0,
)