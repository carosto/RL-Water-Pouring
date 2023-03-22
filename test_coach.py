from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.environments.gym_environment import gymnasium as gymVectorEnvironment
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import SimpleSchedule
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters


# define the environment parameters
env_params = GymVectorEnvironment()
env_params.level = 'water_pouring.envs:PouringEnvRotating'
env_params.additional_simulator_parameters = {'use_gui' : False}

# Clipped PPO
agent_params = ClippedPPOAgentParameters()
agent_params.network_wrappers['main'].input_embedders_parameters = {
    'state': InputEmbedderParameters(scheme=[]),
    'desired_goal': InputEmbedderParameters(scheme=[])
}

graph_manager = BasicRLGraphManager(
    agent_params=agent_params,
    env_params=env_params,
    schedule_params=SimpleSchedule()
)

graph_manager.improve()