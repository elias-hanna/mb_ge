from gym import GoalEnv
from mb_ge.envs import BallInCup3dEnv
import numpy as np

import gym.spaces as spaces

steps_done = 0
monitor_rate = 1000
total_rew_seen = 0

class BallInCup3dGoalEnv(BallInCup3dEnv, GoalEnv):

    def __init__(self, verbose=False):
        self.ball_size = 0
        ## If goal is static (relative goal-space case)
        self.desired_goal = np.zeros(3)
        BallInCup3dEnv.__init__(self, verbose=verbose)
        self.ball_size = self.sim.model.geom_size[2, 0]
        self.obs_shape = self.observation_space['observation'].shape
        self.obs_shape = BallInCup3dEnv._get_obs(self).shape
        # import pdb; pdb.set_trace()
        # Goal space is in cartesian space
        # achieved_goal = desired_goal space
        self.observation_space = spaces.Dict(
            {'observation':spaces.Box(low=-1, high=1, shape=self.obs_shape, dtype=np.float32),
             ## Relative goal-space case
             'achieved_goal':spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
             'desired_goal':spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
             ## Ball pos + target pos as goal-space
             # 'achieved_goal':spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
             # 'desired_goal':spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
            }
        )

    # goal needs to be fixed (or relative, here in end-effector frame)
    def _get_obs(self):
        normal_obs = BallInCup3dEnv._get_obs(self)
        ## Use relative position of ball and target as goal-space
        ball_to_targ = self.ball_to_target()
        achieved_goal = np.array([ball_to_targ[0], ball_to_targ[1], ball_to_targ[2]])
        desired_goal = self.desired_goal
        ## Use ball position + target position as goal-space
        # achieved_goal = normal_obs[:6]
        # target_pos = self.get_site_pos("target")
        # desired_goal = np.append(target_pos, target_pos)
        obs = {
            'observation': normal_obs,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
        }
        return obs

    def step(self, a):
        reward_ctrl = -np.square(a).sum()
        obs = self._get_obs()
        task_reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})
        reward = task_reward # + reward_ctrl
        
        self.do_simulation(a, self.frame_skip)
        self.steps +=1
        
        obs = self._get_obs()
        info = dict(task_reward=task_reward, reward_ctrl=reward_ctrl)
        done = self._is_done(info)

        if self.verbose:
            self._monitoring(reward)

        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on an a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in info and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
        """
        reward = 0
        dist_to_goal = np.linalg.norm(desired_goal - achieved_goal)
        if dist_to_goal <= self.ball_size: # if ball in cup
            reward = 1
        # print(dist_to_goal)
        # print(reward)
        return reward

if __name__ == '__main__':
    import gym
    import mb_ge
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    
    from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
    from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    model_class = DDPG  # works also with SAC, DDPG and TD3

    env = DummyVecEnv([lambda: gym.make("BallInCup3d-goalbased-v0", verbose=True)])
    # Automatically normalize the input features and reward
    env = VecNormalize(env, norm_obs=True, norm_reward=True,
                       clip_obs=10.)

    # Available strategies (cf paper): future, final, episode
    goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
    
    # If True the HER transitions will get sampled online
    online_sampling = True
    # Time limit for the episodes
    max_episode_length = 200
    
    # Initialize the model
    model = model_class(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        # Parameters for HER
        replay_buffer_kwargs=dict(
            n_sampled_goal=1,
            goal_selection_strategy=goal_selection_strategy,
            online_sampling=online_sampling,
            max_episode_length=max_episode_length,
        ),
        verbose=0,
    )
    
    # Train the model
    model.learn(10000)
    
    obs = env.reset()
    for _ in range(10000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()
