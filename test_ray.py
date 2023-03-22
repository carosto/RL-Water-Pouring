from ray.rllib.algorithms.td3 import TD3Config
from ray.tune.logger import pretty_print

import ray.rllib.agents.ppo as ppo

from ray.tune.registry import register_env

import gymnasium as gym

import ray

from tqdm import trange

from water_pouring.envs.pouring_env_rotating_wrapper import PouringEnvRotatingWrapper

import sys

def env_creator(env_config):
    env = gym.make("WaterPouringEnvBase-v0")
    wrapped_env = PouringEnvRotatingWrapper(env)
    return wrapped_env


ray.init()

env_name = "WaterPouringEnvBase-v0"

register_env(env_name, env_creator)

config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 0
trainer = ppo.PPOTrainer(config=config, env="WaterPouringEnvBase-v0")

for i in trange(10):
    result = trainer.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = trainer.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")
"""config = TD3Config().rollouts(num_rollout_workers = 0).environment(env=env_name, env_config={'use_gui' : False})
algo = config.build()

for i in trange(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")"""

"""print('test 1')
episode_reward = 0
done = False
obs = env.reset()
for i in range(10):
    print(f'test {i}')
    action = algo.compute_single_action(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
print(episode_reward)"""