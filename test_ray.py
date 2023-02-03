from ray.rllib.algorithms.td3 import TD3Config
from ray.tune.logger import pretty_print

import gymnasium as gym

import ray

from water_pouring.envs.pouring_env import PouringEnv

"""algo = {
    TD3Config()
    .rollourts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env="WaterPouringEnv-v0")
    .build()
}

for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i%5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")"""


ray.init()

env_name = "WaterPouringEnv-v0"
algo = TD3Config().environment(env=PouringEnv, env_config={'use_gui':False}).framework("torch").build()
env = gym.make(env_name)
print('test 1')
episode_reward = 0
done = False
obs = env.reset()
for i in range(10):
    print(f'test {i}')
    action = algo.compute_single_action(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
print(episode_reward)