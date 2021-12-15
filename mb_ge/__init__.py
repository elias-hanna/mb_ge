from gym.envs.registration import register

register(
    id='BallInCup3d-v0',
    entry_point='mb_ge.envs:BallInCup3dEnv',
)

register(
    id='BallInCup3d-goalbased-v0',
    entry_point='mb_ge.envs:BallInCup3dGoalEnv',
)
