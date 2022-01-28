import numpy as np
from mb_ge.utils.element import Element
from mb_ge.ge.ge import GoExplore
import copy

class ModelBasedGoExplore(GoExplore):
    def __init__(self, params=None, gym_env=None, selection_method=None,
                 go_method=None, exploration_method=None, state_archive=None, dynamics_model=None):
        super().__init__(params=params, gym_env=gym_env, selection_method=selection_method,
                         go_method=go_method, exploration_method=exploration_method,
                         state_archive=state_archive)
        self._dynamics_model = dynamics_model(params=params)
        
    def _process_params(self, params):
        super()._process_params(params)
        if 'model_update_rate' not in params:
            self.model_update_rate = params['model_update_rate']
        else:
            self.model_update_rate = 10
    def _correct_el(self, el, transitions):
        trajectory = []
        for t in transitions:
            trajectory.append(copy.copy(t[1]))
        el.descriptor = trajectory[-1]
        el.trajectory = trajectory[-self.h_exploration:]
        
    def _exploration_phase(self):
        ## reset gym environment
        obs = self.gym_env.reset()
        ## add first state to state_archive
        init_elem = Element(descriptor=self.gym_env.sim.data.qpos[:3], trajectory=[obs], reward=0.,
                            sim_state={'qpos': self.gym_env.sim.data.qpos,
                                       'qvel': self.gym_env.sim.data.qvel})
        self.state_archive.add(init_elem)
        
        budget_used = 0
        i_budget_used = 0
        done = False
        itr = 0
        while budget_used < self.budget and not done:
            obs = self.gym_env.reset()

            ## Select a state to return from the archive
            el = self._selection_method.select_element_from_cell_archive(self.state_archive)
            ## Go to and Explore in imagination from the selected state
            i_elements, i_b_used = self._exploration_method(self._dynamics_model, el,
                                                            self.h_exploration, eval_on_model=True)
            ## Select a state to go to from states found in imagination
            sel_i_el = self._selection_method.select_element_from_element_list(i_elements)
            ## Go back to the selected state
            transitions, b_used = self._go_method.go(self.gym_env, sel_i_el)
            ## Correct sel_i_el to have the right trajectory
            self._correct_el(sel_i_el, transitions)
            ## Update archive and other datasets
            self.state_archive.add(sel_i_el)
            ## Update used budget
            i_budget_used += i_b_used
            budget_used += b_used
            itr += 1
            print(f'b_used: {budget_used} | i_b_used: {i_budget_used} | total_b: {self.budget}')
            ## Train the dynamics model
            self._dynamics_model.add_samples_from_transitions(transitions)
            if itr%self.model_update_rate== 0:
                self._dynamics_model.train()
            
    def __call__(self):
        pass

if __name__ == '__main__':
    from mb_ge.selection.random_selection import RandomSelection
    from mb_ge.go.execute_policy_go import ExecutePolicyGo
    from mb_ge.exploration.random_exploration import RandomExploration
    from mb_ge.exploration.ns_exploration import NoveltySearchExploration
    from mb_ge.archive.fixed_grid_archive import FixedGridArchive
    from mb_ge.models.dynamics_model import DynamicsModel
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
    dynamics_model_params = \
    {
        'obs_dim': 6,
        'action_dim': 3,
        'dynamics_model_type': 'prob', # possible values: prob, det
        'ensemble_size': 4, # only used if dynamics_model_type == prob
        'layer_size': 500,
        'batch_size': 512,
        'learning_rate': 1e-3,
    }
    params = \
    {
        'controller_type': NeuralNetworkController, ## WARNING THIS NEED TO BE A CONTROLLER CLASS
        'controller_params': controller_params,
        
        'budget': 1000000,
        'exploration_horizon': 10,
        'nb_eval_exploration': 10,
        'nb_thread_exploration': 6,

        'archive_type': 'cell',
        'fixed_grid_min': -0.5,
        'fixed_grid_max': 0.5,
        'fixed_grid_div': 5,
        
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,

        'model_update_rate': 10,
        'dynamics_model_params': dynamics_model_params,
    }

    ## Framework methods
    env = gym.make('BallInCup3d-v0')

    selection_method = RandomSelection

    go_method = ExecutePolicyGo

    # exploration_method = NoveltySearchExploration
    exploration_method = RandomExploration

    state_archive_type = FixedGridArchive

    dynamics_model = DynamicsModel
    
    ge = ModelBasedGoExplore(params=params, gym_env=env, selection_method=selection_method,
                             go_method=go_method, exploration_method=exploration_method,
                             state_archive=state_archive_type, dynamics_model=dynamics_model)

    ge._exploration_phase()

    ge.state_archive.visualize()
