from water_pouring.envs.pouring_env_x_rotation_wrapper import XRotationWrapper
import gymnasium as gym
import numpy as np
import sys

env_kwargs = {
        "spill_punish" : 0.1,
        "hit_reward": 0.01,
        "jerk_punish": 0.1,
        "particle_explosion_punish": 0,
        "max_timesteps": 500,
        "scene_file": "scene_test_rotated.json",
        "output_directory": "test"
    }

env = gym.make("WaterPouringEnvBase-v0", **env_kwargs)
wrapped_env = XRotationWrapper(env)
#env = PouringEnvBase(use_gui=False)
obs = wrapped_env.reset()

print(wrapped_env.cup_position)
print(wrapped_env.jug_start_position)

print("The initial observation is {}".format(obs))
sum_reward = 0
step = 0
while True:
    print('Step: ', step)
    # Take a random action
    if step < 10:
        action = np.array([0])
    else:
        action = np.array([0.1])#wrapped_env.action_space.sample()

    print('Action: ', action)
    """
    print('Jug: ', wrapped_env.simulation.n_particles_jug)
    print('Cup: ', wrapped_env.simulation.n_particles_cup)
    print("Pouring: ", wrapped_env.simulation.n_particles_pouring)
    print('Spilled: ', wrapped_env.simulation.n_particles_spilled)
    print('Sum: ', wrapped_env.simulation.n_particles_jug + wrapped_env.simulation.n_particles_cup + wrapped_env.simulation.n_particles_pouring + wrapped_env.simulation.n_particles_spilled)"""
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    print('Reward: ', reward)
    #print("The new observation is {}".format(obs))
    sum_reward += reward
    if terminated or truncated:
        print('Done')
        break
    step += 1
print('Jug: ', wrapped_env.simulation.n_particles_jug)
print('Cup: ', wrapped_env.simulation.n_particles_cup)
print('Sum reward: ', sum_reward)
wrapped_env.close()