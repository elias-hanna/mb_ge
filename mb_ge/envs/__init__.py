from mb_ge.envs.ball_in_cup_env import BallInCupEnv
from mb_ge.envs.ball_in_cup_3d_env import BallInCup3dEnv
import gym
major, minor, patch = gym.__version__.split('.')
if int(minor) > 9 and int(minor) < 21:
    from mb_ge.envs.ball_in_cup_3d_goal_env import BallInCup3dGoalEnv
