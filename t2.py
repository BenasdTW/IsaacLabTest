
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.robots import Robot
from omni.isaac.core.tasks import BaseTask
# from omni.isaac.core.utils.torch.rotations import quat_to_euler_angles

class IsaacSimRLTask(BaseTask):
    def __init__(self, name="SimpleTask", env=None):
        super().__init__(name=name)
        self._env = env

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        # Create a simple scene with a robot and a goal
        create_prim("/World/Floor", "Xform")
        self._robot = Robot("/World/Robot")
        scene.add(self._robot)
        self._goal_position = np.array([1.0, 0.0, 0.0])  # Example goal

    def get_robot(self):
        return self._robot

    def get_goal(self):
        return self._goal_position


class IsaacSimEnv(gym.Env):
    def __init__(self):
        super(IsaacSimEnv, self).__init__()

        # Initialize Isaac Sim
        self._world = World(stage_units_in_meters=1.0)
        self._task = IsaacSimRLTask(env=self)
        self._world.add_task(self._task)
        self._world.reset()

        # Define observation and action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)  # Example
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # Example

    def reset(self):
        # Reset Isaac Sim environment
        self._world.reset()
        self._task.reset()
        robot_position = self._task.get_robot().get_base_position()
        goal_position = self._task.get_goal()
        return self._get_observation(robot_position, goal_position)

    def step(self, action):
        # Apply action
        self._task.get_robot().apply_action(action)

        # Step simulation
        self._world.step(render=False)

        # Compute new observation
        robot_position = self._task.get_robot().get_base_position()
        goal_position = self._task.get_goal()
        observation = self._get_observation(robot_position, goal_position)

        # Compute reward
        distance_to_goal = np.linalg.norm(goal_position - robot_position)
        reward = -distance_to_goal  # Reward is negative distance to the goal

        # Check if episode is done
        done = distance_to_goal < 0.1  # Example success condition

        return observation, reward, done, {}

    def render(self, mode="human"):
        # Render simulation
        self._world.render()

    def close(self):
        # Cleanup Isaac Sim
        self._world.close()

    def _get_observation(self, robot_position, goal_position):
        # Create an observation vector (e.g., robot position and distance to goal)
        distance_to_goal = goal_position - robot_position
        return np.concatenate([robot_position, distance_to_goal])


if __name__ == "__main__":
    # Create environment
    env = IsaacSimEnv()

    # Train PPO agent
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Evaluate trained model
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            print("Goal reached!")
            break

    env.close()
