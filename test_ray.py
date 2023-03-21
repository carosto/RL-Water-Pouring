from ray.rllib.algorithms.td3 import TD3Config
from ray.tune.logger import pretty_print

from ray.tune.registry import register_env

import gym

import ray

from water_pouring.envs.pouring_env_rotating_wrapper import PouringEnvRotatingWrapper


def env_creator(env_config):
    env = gym.make("WaterPouringEnvBase-v0")
    wrapped_env = PouringEnvRotatingWrapper(env)
    return wrapped_env#PouringEnvRotating(use_gui=env_config['use_gui'])


ray.init()

env_name = "WaterPouringEnvBase-v0"

register_env(env_name, env_creator)

#algo = TD3Config().environment(env=PouringEnvRotating, env_config={'use_gui':False}).framework("torch").build()
algo = TD3Config().environment(env=env_name, env_config={'use_gui':False}).framework("torch").build()
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