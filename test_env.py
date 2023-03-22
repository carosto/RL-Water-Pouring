from water_pouring.envs.pouring_env_rotating_wrapper import PouringEnvRotatingWrapper
import gymnasium as gym


env = gym.make("WaterPouringEnvBase-v0")
wrapped_env = PouringEnvRotatingWrapper(env)
#env = PouringEnvBase(use_gui=False)
obs = wrapped_env.reset()

print("The initial observation is {}".format(obs))

while True:
    # Take a random action
    action = wrapped_env.action_space.sample()
    print('Action: ', action)
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    print('Observation space shape: ', wrapped_env.observation_space.shape)
    print('Reward: ', reward)
    print('Observation sample: ', wrapped_env.observation_space.sample())
    print("The new observation is {}".format(obs))
    
    if terminated == True:
        break

wrapped_env.close()