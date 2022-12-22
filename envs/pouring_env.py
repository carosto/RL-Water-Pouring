import gymnasium as gym
from gymnasium import spaces

import pysplishsplash as sph
import pysplishsplash.Utilities.SceneLoaderStructs as Scenes

class PouringEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, arg1, arg2, ...):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = #TODO spaces.Discrete(N_DISCRETE_ACTIONS)
    # Example for using image as input:
    self.observation_space = #TODO spaces.Box(low=0, high=255, shape= (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

  def step(self, action):
    # Execute one time step within the environment
    ...
  def reset(self, use_gui=False):
    # Reset the state of the environment to an initial state
    base = sph.Exec.SimulatorBase()
    base.init(
            useGui=use_gui, sceneFile=os.path.abspath("scene_test.json"),
            stopAt=stopAt, #originally 20
            outputDir= output_dir + self.fname
        )

    # Create an imgui simulator
    gui = sph.GUI.Simulator_GUI_imgui(base)
    base.setGui(gui)
    # TODO check how to use callback method
    base.setTimeStepCB(self.time_step_callback)

    # Get the scene and add objects
    scene = sph.Exec.SceneConfiguration.getCurrent().getScene()

    # Add the cup
    cup_pos = [0,0,0]
    cup_file = "wine_glass.obj"
    rots = R.from_quat(self.cup_data[100,3:])
    r = rots.as_matrix()
    scene.boundaryModels.append(
            Scenes.BoundaryData(
                meshFile=f"ObjectFiles/{cup_file}",
                translation=self.cup_data[100,:3],
                scale=[1, 1, 1],
                color=[0.5, 0.5, 0.5, 1.0],
                isAnimated=True,
                rotation=r,
                isWall=False,
                mapInvert=False,
                mapResolution=[25, 25, 25],
            )
        )
    


  def render(self, mode='human', close=False):
    # Render the environment to the screen
    ...