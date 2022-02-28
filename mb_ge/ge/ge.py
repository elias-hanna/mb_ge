import numpy as np
from mb_ge.utils.element import Element
import os

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
        self.observed_transitions = []

    def _process_params(self, params):
        if 'budget' in params:
            self.budget = params['budget']
        if 'exploration_horizon' in params:
            self.h_exploration = params['exploration_horizon']
        if 'controller_type' in params:
            self.controller_type = params['controller_type']
        if 'dump_rate' in params:
            self.dump_rate = params['dump_rate']
        else:
            self.dump_rate = 100
        if 'dump_path' in params:
            self.dump_path = params['dump_path']
        else:
            curr_dir = os.getcwd()
            self.dump_path = curr_dir
            
    def _exploration_phase(self):
        ## reset gym environment
        obs = self.gym_env.reset()
        ## add first state to state_archive
        init_elem = Element(descriptor=obs[:3], trajectory=[obs], reward=0.,
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
            transitions, b_used = self._go_method.go(self.gym_env, el)
            budget_used += b_used
            ## Explore from the selected state
            elements, b_used = self._exploration_method(self.gym_env, el, self.h_exploration)
            ## Update archive and other datasets
            for elem in elements:
                self.state_archive.add(elem)
            self.observed_transitions.append(transitions)
            ## Update used budget
            budget_used += b_used
            itr += 1
            print(f'b_used: {budget_used} | total_b: {self.budget}')
            if itr%self.dump_rate == 0:
                path_to_dir_to_create = os.path.join(self.dump_path, f'results_{itr}')
                os.makedirs(path_to_dir_to_create, exist_ok=True)
                self.state_archive.visualize(budget_used, itr=itr)
                for key in self.state_archive._archive.keys():
                    np.save(f'{self.dump_path}/results_{itr}/archive_cell_{key}_itr_{itr}',
                            self.state_archive._archive[key]._elements)
                    
    def __call__(self):
        return self._exploration_phase()
