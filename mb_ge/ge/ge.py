import numpy as np
from diversity_algorithms.controllers.fixed_structure_nn_numpy import SimpleNeuralControllerNumpy as SimpleNeuralController

class GoExplore():

    def __init__(self, params=None, budget=None, gym_env=None, selection_method=None, go_method=None,
                 exploration_method=None, state_archive=None):
        ## Process run parameters
        self._process_params(params)
        ## Intialize functors (do this in params?)
        self._selection_method = selection_method
        self._go_method = go_method
        self._exploration_method = exploration_method(params)
        ## Intialize state_archive
        self.state_archive = state_archive
        self.gym_env = gym_env ## needs to be an already initialized env
        ## Transition Dataset initialization (used to train Dynamics Model)
        self.action_space_dim = self.gym_env.action_space.shape[0]
        self.observation_space_dim = self.gym_env.observation_space.shape[0]
        self.observed_transitions = None # init to None so it errors out if not properly initialized

    def _process_params(self):
        if 'budget' in params:
            self.budget = params['budget']
        if 'exploration_horizon' in params:
            self.h_exploration = params['exploration_horizon']
        if 'controller_type' in params:
            self.controller_type = params['controller_type']
            
    def _exploration_phase(self):
        ## reset gym environment
        obs = gym_env.reset()
        ## add first state to state_archive
        self.state_archive.add(obs)
        itr = 0
        done = False
        while itr < self.budget and not done:
            ## Select a state to return from the archive
            s = self.selection_method(self.state_archive)
            ## Go back to the selected state
            budget_used = self.go_method(self.gym_env, s)
            ## Explore from the selected state
            exploration_results, budget_used = self.exploration_method(self.gym_env,
                                                                       self.h_exploration)
            ## Update archive and other datasets
            self.state_archive.update(exploration_results)
            
    def __call__(self):
        pass

if __name__ == '__main__':
    controller_params = \
    {
        'controller_input_dim': 0,
        'controller_output_dim': 0,
        'n_hidden_layers': 2,
        'n_neurons_per_hidden': 50
    }
    params = \
    {
        'controller_type': SimpleNeuralController, ## WARNING THIS NEED TO BE A CONTROLLER CLASS
        'controller_params': controller_params,
        'budget': 1000,
        'exploration_horizon': 10,
        'nb_eval_exploration': 10,
        'nb_thread_exploration': 6,
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,
    }
