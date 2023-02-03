from gymnasium.envs.registration import register

register(
    id='WaterPouringEnv-v0',
    entry_point='water_pouring.envs:PouringEnv',
    nondeterministic = True
)