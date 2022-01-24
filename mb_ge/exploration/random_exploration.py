## Data manipulation imports
import numpy as np
from itertools import repeat

## Multiprocessing imports
from multiprocessing import Pool
from copy import copy
from copy import deepcopy

## Local imports
from mb_ge.exploration.exploration_method import ExplorationMethod
from mb_ge.controller.nn_controller import NeuralNetworkController
from mb_ge.utils.element import Element

class RandomExploration(ExplorationMethod):
    def __init__(self, params=None):
        super().__init__(params=params)
        self.gym_env = None
        
    def _process_params(self, params):
        super()._process_params(params)
        
    def _single_policy_eval(self, x, gym_env, prev_element):
        ## Create a copy of the controller
        controller = self.controller.copy()
        ## Verify that x and controller parameterization have same size
        # assert len(x) == len(self.controller.get_parameters())
        ## Set controller parameters
        controller.set_parameters(x)
        # env = deepcopy(gym_env) ## need to verify this works

        # print("prev desc: ", prev_element.descriptor)
        # print("prev last traj: ", prev_element.trajectory[-1])
        # print("gym_env curr obs: ", gym_env._get_obs())
        # print("prev sim state: ", prev_element.sim_state)
        env = copy(gym_env) ## need to verify this works
        env.set_state(prev_element.sim_state['qpos'], prev_element.sim_state['qvel'])
        # print("env curr obs: ", env._get_obs())
        # print("\n\n\n")
        traj = []
        obs = prev_element.trajectory[-1]
        cum_rew = 0
        # traj.append(obs)
        ## WARNING: need to get previous obs
        for _ in range(self.exploration_horizon):
            action = controller(obs)
            obs, reward, done, info = env.step(action)
            cum_rew += reward
            traj.append(obs)
        # print(env.sim.data.qpos)
        # print(env.sim.data.qvel)
        element = Element(descriptor=traj[-1][:3], trajectory=traj, reward=cum_rew,
                          policy_parameters=x, previous_element=prev_element,
                          sim_state={'qpos': copy(env.sim.data.qpos),
                                     'qvel': copy(env.sim.data.qvel)})
        ## WARNING: Need to add a bd super function somewhere in params or in Element I guess
        return element

    def _compute_spent_budget(self, elements):
        return sum([len(el.trajectory) for el in elements])
        
    # def _explore(self, gym_env, last_obs, exploration_horizon):
    def _explore(self, gym_env, prev_element, exploration_horizon):
        ## Set exploration horizon (here and not in params because it might be dynamic)
        self.exploration_horizon = exploration_horizon
        ## Setup multiprocessing pool
        pool = Pool(processes=self.nb_thread)
        ## Get policy reprensation size
        policy_representation_dim = len(self.controller.get_parameters())
        ## Inits
        to_evaluate = []
        ## Random policy parametrizations creation
        for _ in range(self.nb_eval):
            ## Create a random policy parametrization 
            x = np.random.uniform(low=self.policy_param_init_min,
                                  high=self.policy_param_init_max,
                                  size=policy_representation_dim)
            to_evaluate += [x]
        env_map_list = [gym_env for _ in range(self.nb_eval)]
        ## Evaluate all generated policies on given environment
        elements = pool.starmap(self._single_policy_eval, zip(to_evaluate, repeat(gym_env), repeat(prev_element)))
        
        ## Close the multiprocessing pool
        pool.close()

        return elements, self._compute_spent_budget(elements)

if __name__ == '__main__':
    ## Test imports
    import gym
    controller_params = \
    {
        'controller_input_dim': 2,
        'controller_output_dim': 1,
        'n_hidden_layers': 2,
        'n_neurons_per_hidden': 50
    }
    params = \
    {
        'controller_type': NeuralNetworkController, ## WARNING THIS NEED TO BE A CONTROLLER CLASS
        'controller_params': controller_params,
        'budget': 1000,
        'exploration_horizon': 10,
        'nb_eval_exploration': 10,
        'nb_thread_exploration': 6,
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,
    }
    rand_expl = RandomExploration(params)
    # rand_expl = ExplorationMethod(params=params)
    env = gym.make('MountainCarContinuous-v0')

    obs = env.reset()
    policies, trajs, budget_spent = rand_expl(env, obs, 10)

    print("Test output:")
    print(f"len of trajs list: {len(trajs)}")
    print("Len of each traj:")
    for pi, traj in zip(policies, trajs):
        print()
        print(f"len of traj: {len(traj)}")
        print(f"corresponding policy: {pi}")
