import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

# Path to module
import os
import mb_ge

steps_done = 0
monitor_rate = 1000
total_rew_seen = 0

class BallInCup3dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, relative_obs=True, dense=False, random_init=False, verbose=False):
        self.max_steps = 300
        self._max_episode_steps = 300
        self.steps = 0
        self.dense = dense
        self.random_init = random_init
        self.verbose = verbose
        self.relative_obs = relative_obs

        self.init_qpos = [0. ,0. ,0.   # initial cup position in cartesian space
                          ,0. ,0. ,0.] # initial ball posiiton in cartesian space

        self.init_qvel = [0. ,0. ,0.   # initial cup velocity
                          ,0. ,0. ,0.] # initial ball velocity
        major, minor, patch = mujoco_env.gym.__version__.split('.')
        self.major = int(major);self.minor = int(minor);self.patch = int(minor)
        
        utils.EzPickle.__init__(self)
        path_to_module = os.path.dirname(mb_ge.__file__)
        mujoco_env.MujocoEnv.__init__(self, "{}/envs/assets/ball_in_cup_3d.xml".format(path_to_module), 2)
        
    def _is_done(self, info):
        done = False
        if self.in_target():
            done = True
        if self.steps >= self.max_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        info["relative_ball_to_target_pos"] = self.ball_to_target()
        if self.minor > 9:
            info["relative_ball_to_target_vel"] = self.sim.data.qvel.flat[:3] - self.sim.data.qvel.flat[3:] # target - ball vel (target vel == cup vel)
        else:
            info["relative_ball_to_target_vel"] = self.data.qvel.flat[:3] - self.data.qvel.flat[3:] # target - ball vel (target vel == cup vel)
        return done

    def sample_q_vectors(self):
        state_min = np.array([-0.4]*6); state_max = np.array([0.4]*6)
        vel_min = -0.4; vel_max = 0.4

        qpos = np.zeros(6)
        qvel = np.zeros(6)
        ## Sample qpos
        # Sample pos for target
        qpos[:3] = np.random.uniform(low=state_min[:3], high=state_max[:3], size=(3,))
        # Sample pos for ball
        qpos[3:] = np.random.uniform(low=state_min[3:]+qpos[:3], high=state_max[3:]+qpos[:3])
        ## Sample qvel
        qvel = np.random.uniform(low=vel_min, high=vel_max, size=(6,)) 
        ## Recreate state from sampled qpos and qvel
        s = [0]*6
        s[:3] = qpos[:3] - qpos[3:]
        s[2] -= -.05 ## s reflects target pos not cup pos (which is .05 below cup)
        s[2] += .3 ## qpos is actually at joint which is .3 further on z axis
        s[3:6] = qvel[:3] - qvel[3:]
        ## Return qpos, qvel and corresponding state
        return qpos, qvel, s

    def _monitoring(self, reward):
        global steps_done, total_rew_seen
        steps_done += 1
        total_rew_seen += reward
        if steps_done%monitor_rate == 0:
            print("{} steps done. Cumulated reward seen {}".format(steps_done, total_rew_seen))
        
    def step(self, a):
        reward_ctrl = -np.square(a).sum()
        task_reward = self.get_reward()
        reward = task_reward # + reward_ctrl
        
        self.do_simulation(a, self.frame_skip)
        self.steps += 1

        obs = self.get_obs()
        info = dict(task_reward=task_reward, reward_ctrl=reward_ctrl)
        done = self._is_done(info)

        if self.verbose:
            self._monitoring(reward)

        return obs, reward, done, info

    def _step(self, a):
        return self.step(a)

    # def _render(self, mode='human', close=False):
    #     return self.render()
    
    def viewer_setup(self):
        from mujoco_py.generated import const
        self.viewer._paused = True # start viewer paused
        self.viewer.cam.fixedcamid = 0 # start viewer at cam 0
        self.viewer.cam.type = const.CAMERA_FIXED # cam 0 is a fixed cam (-1 is free)
    
    def reset_model(self):
        self.steps = 0
        # qpos is [cup_x, cup_y, cup_z, ball_x, ball_y, ball_z]
        # qvel is analogous

        # Random initialization
        if self.random_init:
            qpos = (
                self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
                + self.init_qpos
            )

            qvel = self.init_qvel + self.np_random.uniform(
                low=-0.005, high=0.005, size=self.model.nv
            )
            qpos[:3] = self.init_qpos[:3] # still set cup pose to init
            qvel[:3] = self.init_qvel[:3] # still set cup vel to init
        else:
            qpos = np.array(self.init_qpos)
            qvel = np.array(self.init_qvel)
            
        self.set_state(qpos, qvel)
        return self.get_obs()

    def get_obs(self):
        if self.minor > 9:
            if self.relative_obs:
                return np.concatenate((
                    # self.sim.data.qpos.flat[:3] - self.sim.data.qpos.flat[3:], # cup - ball pose
                    # self.sim.data.qvel.flat[:3] - self.sim.data.qvel.flat[3:],) # cup - ball vel
                    self.get_site_pos("target") - self.get_site_pos("ball"), # target - ball pose
                    self.sim.data.qvel.flat[:3] - self.sim.data.qvel.flat[3:],) # target - ball vel (target vel == cup vel)
                )
            else:
                return np.concatenate((
                    self.sim.data.qpos.flat[:3], # cup pose
                    self.sim.data.qvel.flat[:3], # cup vel
                    self.sim.data.qpos.flat[3:], # ball pose
                    self.sim.data.qvel.flat[3:],) # ball vel
                )
        else:
            if self.relative_obs:
                return np.concatenate((
                    # self.data.qpos.flat[:3] - self.data.qpos.flat[3:], # cup - ball pose
                    # self.data.qvel.flat[:3] - self.data.qvel.flat[3:],) # cup - ball vel
                    self.get_site_pos("target") - self.get_site_pos("ball"), # target - ball pose
                    self.data.qvel.flat[:3] - self.data.qvel.flat[3:],) # target - ball vel (target vel == cup vel)
                )
            else:
                return np.concatenate((
                    self.data.qpos.flat[:3], # cup pose
                    self.data.qvel.flat[:3], # cup vel
                    self.data.qpos.flat[3:], # ball pose
                    self.data.qvel.flat[3:],) # ball vel
                )
            
    def get_relative_pos(self):
        return self.get_site_pos("target") - self.get_site_pos("ball") # target - ball pose
    
    def get_site_pos(self, site_name):
        if self.minor > 9:
            if site_name == "cup":
                return self.sim.data.site_xpos[0]
            elif site_name == "target":
                return self.sim.data.site_xpos[1]
            elif site_name == "ball":
                return self.sim.data.site_xpos[2]
            raise ValueError("{} is not a valid option for get_site_pos method. Valid options are cup/target/ball".format(site_name))
        else:
            if site_name == "cup":
                return self.data.site_xpos[0]
            elif site_name == "target":
                return self.data.site_xpos[1]
            elif site_name == "ball":
                return self.data.site_xpos[2]
            raise ValueError("{} is not a valid option for get_site_pos method. Valid options are cup/target/ball".format(site_name))
        
    def ball_to_target(self):
        """Returns the vector from the ball to the target."""
        # target = self.sim.data.site_xpos[1, [0, 1, 2]]
        # ball = self.sim.data.site_xpos[2, [0, 1, 2]]
        target = self.get_site_pos("target")
        ball = self.get_site_pos("ball")
        return target - ball

    def in_target(self):
        """Returns 1 if the ball is in the target, 0 otherwise."""
        ball_to_target = abs(self.ball_to_target())
        if self.minor > 9:
            target_size = self.sim.model.site_size[1, [0, 1, 2]]
            ball_size = self.sim.model.geom_size[2, 0]
        else:
            target_size = self.model.site_size[1, [0, 1, 2]]
            ball_size = self.model.geom_size[2, 0]
        return float(all(ball_to_target < target_size - ball_size))

    def get_reward(self):
        """Returns a dense or sparse reward."""
        if self.dense:
            ## This reward function creates an empty reward zone right below the cup
            # dist = self.get_site_pos("target") - self.get_site_pos("ball")
            # if abs(dist[0]) < .1 and abs(dist[1]) < .1 and dist[2] > 0 and dist[2] < .3:
            #     return 0
            # reward = -np.linalg.norm(dist)
            
            ## This reward function rewards policies lifting the ball while
            ## mainting the rope tense as long as the ball is below the cup
            ## If the ball goes above the cup, reward switches to be maximal
            ## when the distance between the ball and the cup is minimal
            rel_pos = self.get_site_pos("target") - self.get_site_pos("ball")
            if rel_pos[2] > 0: ## Ball is below the cup
                ## tense rew is between 0 and 1 (0 when close to cup, 1 when max tense)
                tense_rew = np.linalg.norm(rel_pos)/.358 ## max norm is ~ 0.358
                ## lifting rew is between 0 and 1 (0 when below cup, 1 when going above)
                ## lowest rel_pos is ~ 0.358, highest is 0
                lifting_rew = (rel_pos[2] - .358)/(0 - .358)
                reward = lifting_rew + tense_rew
            else: ## Ball is above the cup
                ## target close rew is maximal when ball is in target
                target_close_rew = 2 + 2*(np.linalg.norm(rel_pos) - .358)/(0 - .358)
                reward = target_close_rew
        else:
            reward = self.in_target()
        return reward

if __name__ == '__main__':
    import gym
    import mb_ge
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    
    env = DummyVecEnv([lambda: gym.make("BallInCup3d-v0", verbose=True, dense=False)])
    # Automatically normalize the input features and reward
    env = VecNormalize(env, norm_obs=True, norm_reward=True,
                       clip_obs=10.)

    # PPO example, PPO is on-policy
    # from stable_baselines3 import PPO
    # model = PPO("MlpPolicy", env, verbose=0)

    # model.learn(total_timesteps=100000)

    
    # DDPG example, DDPG is off-policy, expect longer computation time
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=100000, log_interval=10)

    # SAC example, SAC is off-policy, expect longer computation time
    # from stable_baselines3 import SAC

    # model = SAC("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=10000, log_interval=4)

    # TD3 example, TD3 is off-policy, expect longer computation time
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
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()
