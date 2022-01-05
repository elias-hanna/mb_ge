## Data manipulation imports
import numpy as np
from itertools import repeat

## Multiprocessing imports
from multiprocessing import Pool
from multiprocessing import cpu_count
from copy import copy

## Local imports
from exploration_method import ExplorationMethod
from exploration_method import ExplorationResults
from mb_ge.controller.nn_controller import NeuralNetworkController
from mb_ge.archive.archive import Element

class RandomExploration(ExplorationMethod):
    def __init__(self, params=None):
        super().__init__(params)
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
        env = copy(gym_env) ## need to verify this works

        traj = []
        obs = prev_element.trajectory[-1]
        ## WARNING: need to get previous obs
        for _ in range(self.exploration_horizon):
            action = controller(obs)
            obs, reward, done, info = env.step(action)
            traj.append((obs, reward, done, info))
        element = Element()
        ## WARNING: Need to add a bd super function somewhere in params or in Element I guess
        element.descriptor = traj[0][-1]
        element.trajectory = traj[0]
        element.reward = sum(traj[1])
        element.policy_parameters = x
        element.previous_element = prev_element
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
            x = np.random.uniform(low=params['policy_param_init_min'],
                                  high=params['policy_param_init_max'],
                                  size=policy_representation_dim)
            to_evaluate += [x]
        env_map_list = [gym_env for _ in range(self.nb_eval)]
        ## Evaluate all generated policies on given environment
        elements = pool.starmap(self._single_policy_eval, zip(to_evaluate, repeat(gym_env), repeat(prev_element)))
        
        ## Close the multiprocessing pool
        pool.close()

        return elements, self._compute_spent_budget(elements)
    
        # results = ExplorationResults()
        # results.add(to_evaluate, trajs_list)
        
        # return to_evaluate, trajs_list, self._compute_spent_budget(trajs_list)
        # return results, self._compute_spent_budget(trajs_list)

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
