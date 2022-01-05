## Abstract layer imports
from abc import abstractmethod

## Multiprocessing imports
from multiprocessing import cpu_count

class ExplorationResults():
    def __init__(self):
        self.policies = [] # Will be a list containing policy parameters
        self.trajs = [] # Will be a list containing each policy corresponding transitions (at+st)
        self.gym_env = None
        
    def add(self, policies, trajs):
        self.policies.extend(policies) # Warning: policies need to be a list [] of np.arrays
        self.trajs.extend(trajs) # Warning: trajs need to be a list [] of np.arrays

    def get_tuple_by_index(self, index):
        """
        Returns a tuple (policy, traj) corresponding to the index-th exploration policy
        """
        return (self.policies[i], self.trajs[i])
        
class ExplorationMethod():
    def __init__(self, params=None):
        ## Process exploration parameters
        self._process_params(params)
        self.exploration_results = ExplorationResults()

    def _process_params(self, params):
        if 'nb_eval_exploration' in params:
            self.nb_eval = params['nb_eval_exploration']
        else:
            self.nb_eval = 10

        if 'nb_thread_exploration' in params:
            self.nb_thread = params['nb_thread_exploration']
        else:
            self.nb_thread = cpu_count()-1 or 1

        if 'controller_type' in params:
            ## Associate an instance of controller_type with given params
            self.controller = params['controller_type'](params=params)
        else:
            raise Exception('ExplorationMethod _process_params error: controller_type not in params')

    @abstractmethod
    # def _explore(self, gym_env, last_obs, exploration_horizon):
    def _explore(self, gym_env, prev_element, exploration_horizon):
        """
        Args:
            gym_env: Environment on which to perform exploration, already in the state to explore from
            last_obs: last observation on current environment
            exploration_horizon: Max number of steps performed on gym_env by exploration policies

        Returns:
            policies: obtained policies
            # exploration_results: object storing all policies and their respective trajectories
            trajs: corresponding trajectories (ordered)
            budget_used: total amount of interactions with environment 
        """
        raise NotImplementedError

    def __call__(self, gym_env, last_obs, exploration_horizon):
        return self._explore(gym_env, last_obs, exploration_horizon)
