from gym.envs.registration import register

register(
    id='CartPoleContinual-v0',
    entry_point='gym_barm.envs:Continuous_CartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='MountainCarContinual-v0',
    entry_point='gym_barm.envs:MountainCarContinualEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='MountainCarContinualEnriched-v0',
    entry_point='gym_barm.envs:MountainCarContinualEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='AcrobotHard-v1',
    entry_point='gym_barm.envs:AcrobotEnv',
    reward_threshold=-90.0,
    max_episode_steps=500,
)

register(
    id='AcrobotScrambled-v1',
    entry_point='gym_barm.envs:AcrobotScrambledEnv',
    reward_threshold=-100.0,
    max_episode_steps=500,
)

register(
    id='CartPole_Cont-v0',
    entry_point='gym_barm.envs:Continuous_CartPoleEnv',
    max_episode_steps=200,
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
    max_episode_steps=200,
    reward_threshold=195.0,
)