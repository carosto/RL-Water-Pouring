from gymnasium.envs.registration import register

register(
    id='WaterPouringEnvBase-v0',
    entry_point='water_pouring.envs:PouringEnvBase',
    nondeterministic = True
)