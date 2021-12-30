## Data manipulation imports
import numpy as np

## Multiprocessing imports
from multiprocessing import Pool
from multiprocessing import cpu_count

## Local imports
from exploration_method import ExplorationMethod
from exploration_method import ExplorationResults

def RandomExploration(ExplorationMethod):
    def _process_params(self, params):
        super()._process_params(params)
        
    def _single_policy_eval(self, x):
        controller = SimpleNeuralController(self.gym_env.obs_space.shape[0],
                                            self.gym_env.action_space.shape[0],
                                            params=params['controller_params'])
        env = self.gym_env.copy() ## need to verify this works

        traj = []
        ## WARNING: need to get previous obs
        for _ in range(self.exploration_horizon):
            action = controller(obs)
            obs, reward, done, info = env.step(action)
            traj.append((obs, reward, done, info))
        return traj
            
    def _explore(self, gym_env, exploration_horizon):
        ## Set exploration horizon (here and not in params because it might be dynamic)
        self.exploration_horizon = exploration_horizon
        ## Setup multiprocessing pool
        pool = Pool(processes=self.nb_thread)
        policy_representation_dim = SimpleNeuralController(controller_input_dim, controller_output_dim,
                                                           params=controller_nnparams).n_weights

        to_evaluate = []
        for _ in range(self.nb_eval):
            ## Create a random policy parametrization 
            x = np.random.uniform(low=params['policy_param_init_min'],
                                  high=params['policy_param_init_max'],
                                  size=policy_representation_dim)
            to_evaluate += [x]
            
        trajs_list = pool.map(_single_policiy_eval, to_evaluate)

        pool.close()

if __name__ == '__main__':
    pass
