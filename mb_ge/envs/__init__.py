from mb_ge.envs.ball_in_cup_env import BallInCupEnv
from mb_ge.envs.ball_in_cup_3d_env import BallInCup3dEnv
import gym
major, minor, patch = gym.__version__.split('.')
if minor > 9 and minor < 21:
    from mb_ge.envs.ball_in_cup_3d_goal_env import BallInCup3dGoalEnv
