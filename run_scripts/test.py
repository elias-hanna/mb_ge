import mb_ge
import gym

# from pybotics.robot import Robot
# from pybotics.predefined_models import ur10

# robot = Robot.from_parameters(ur10())

import numpy as np
np.set_printoptions(suppress=True)

joints = np.deg2rad([5,5,5,5,5,5])
# pose = robot.fk(joints)
# print(pose)

# solved_joints = robot.ik(pose)
# print(np.rad2deg(solved_joints))

# exit()

kwargs = {"dense":True, "random_init":True}

env = gym.make("BallInCup3d-v0", **kwargs) # deterministic

# PPO example, PPO is on-policy
# from stable_baselines3 import PPO
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1000)


# DDPG example, DDPG is off-policy
# from stable_baselines3 import DDPG
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# # The noise objects for DDPG
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
# model.learn(total_timesteps=100, log_interval=10)
# model.save("ddpg_ball_in_cup_3d")
# env = model.get_env()

# SAC example, SAC is off-policy
# from stable_baselines3 import SAC

# model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000, log_interval=4)

# TD3 example, TD3 is off-policy
# from stable_baselines3 import TD3
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# # The noise objects for TD3
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
# model.learn(total_timesteps=10000, log_interval=10)

# Visualization Loop
obs = env.reset()
for i in range(1000):
  action = env.action_space.sample()
  # action, _states = model.predict(obs, deterministic=True)
  obs, reward, done, info = env.step(action)
  env.render()
  if done:
    print("done:",done)
    print("step:",i)
    obs = env.reset()
    
env.close()
