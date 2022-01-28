import numpy as np
from mb_ge.utils.element import Element

class GoExplore():
    def __init__(self, params=None, gym_env=None, selection_method=None,
                 go_method=None, exploration_method=None, state_archive=None):
        ## Process run parameters
        self._process_params(params)
        ## Intialize functors (do this in params?)
        self._selection_method = selection_method()
        self._go_method = go_method(params=params)
        self._exploration_method = exploration_method(params=params)
        ## Intialize state_archive
        self.state_archive = state_archive(params)
        self.gym_env = gym_env ## needs to be an already initialized env
        ## Transition Dataset initialization (used to train Dynamics Model)
        self.action_space_dim = self.gym_env.action_space.shape[0]
        self.observation_space_dim = self.gym_env.observation_space.shape[0]
        self.observed_transitions = None # init to None so it errors out if not properly initialized

    def _process_params(self, params):
        if 'budget' in params:
            self.budget = params['budget']
        if 'exploration_horizon' in params:
            self.h_exploration = params['exploration_horizon']
        if 'controller_type' in params:
            self.controller_type = params['controller_type']
            
    def _exploration_phase(self):
        ## reset gym environment
        obs = self.gym_env.reset()
        ## add first state to state_archive
        init_elem = Element(descriptor=self.gym_env.sim.data.qpos[:3], trajectory=[obs], reward=0.,
                            sim_state={'qpos': self.gym_env.sim.data.qpos,
                                       'qvel': self.gym_env.sim.data.qvel})
        # import pdb; pdb.set_trace()
        self.state_archive.add(init_elem)
        itr = 0
        budget_used = 0
        done = False
        while budget_used < self.budget and not done:
            obs = self.gym_env.reset()
            ## Select a state to return from the archive
            el = self._selection_method.select_element_from_cell_archive(self.state_archive)
            ## Go back to the selected state
            _, b_used = self._go_method.go(self.gym_env, el)
            budget_used += b_used
            ## Explore from the selected state
            elements, b_used = self._exploration_method(self.gym_env, el, self.h_exploration)
            ## Update archive and other datasets
            for elem in elements:
                self.state_archive.add(elem)
            ## Update used budget
            budget_used += b_used
            itr += 1
            print(f'b_used: {budget_used} | total_b: {self.budget}')
            
    def __call__(self):
        pass

if __name__ == '__main__':
    from mb_ge.selection.random_selection import RandomSelection
    from mb_ge.go.execute_policy_go import ExecutePolicyGo
    from mb_ge.exploration.random_exploration import RandomExploration
    from mb_ge.exploration.ns_exploration import NoveltySearchExploration
    from mb_ge.archive.fixed_grid_archive import FixedGridArchive

    from mb_ge.controller.nn_controller import NeuralNetworkController

    import gym
    import gym_wrapper # for swimmerfullobs

    controller_params = \
    {
        'controller_input_dim': 6,
        'controller_output_dim': 3,
        'n_hidden_layers': 2,
        'n_neurons_per_hidden': 50
    }
    params = \
    {
        'controller_type': NeuralNetworkController, ## WARNING THIS NEED TO BE A CONTROLLER CLASS
        'controller_params': controller_params,
        
        'budget': 100000,
        'exploration_horizon': 10,
        'nb_eval_exploration': 10,
        'nb_thread_exploration': 6,

        'archive_type': 'cell',
        'fixed_grid_min': -0.5,
        'fixed_grid_max': 0.5,
        'fixed_grid_div': 5,
        
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,
    }

    ## Framework methods
    env = gym.make('BallInCup3d-v0')

    selection_method = RandomSelection

    go_method = ExecutePolicyGo

    # exploration_method = NoveltySearchExploration
    exploration_method = RandomExploration

    state_archive_type = FixedGridArchive
    
    ge = GoExplore(params=params, gym_env=env, selection_method=selection_method,
                   go_method=go_method, exploration_method=exploration_method,
                   state_archive=state_archive_type)

    ge._exploration_phase()

    ge.state_archive.visualize()

    # import pdb; pdb.set_trace()
