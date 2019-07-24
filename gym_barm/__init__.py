from gym.envs.registration import register

register(
    id='CartPole_Cont-v0',
    entry_point='gym_barm.envs:Continuous_CartPoleEnv',
    max_episode_steps=float('inf'),
    reward_threshold=195.0,
)

register(
    id='CartPole_Enriched-v0',
    entry_point='gym_barm.envs:EnrichedCartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='CartPole_EnrichedCont-v0',
    entry_point='gym_barm.envs:EnrichedContinuousCartPoleEnv',
    max_episode_steps=float('inf'),
    reward_threshold=195.0,
)