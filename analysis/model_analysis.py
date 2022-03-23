# Data manipulation includes
import numpy as np

# OS manipulation includes
import sys
import os

# MB-GE includes
import gym
from mb_ge.models.dynamics_model import DynamicsModel
from mb_ge.utils.element import Element


def load_archive_data(folderpath):
    params = np.load(os.path.join(folderpath,'params.npz'))['arr_0']
    descriptors = np.load(os.path.join(folderpath,'descriptors.npz'))['arr_0']
    prev_descriptors = np.load(os.path.join(folderpath,'prev_descriptors.npz'))['arr_0']

    return params, descriptors, prev_descriptors

def reconstruct_elements(params, descriptors, prev_descriptors, gym_env, fetch_trajectories=True):
    assert len(params) == len(descriptors) == len(prev_descriptors)
    nb_of_elems = len(params)
    elements = []
    leaf_elems = []
    current_ends = []
    init_elem = Element(policy_parameters=params[0], descriptor=descriptors[0])
    current_ends.append(init_elem)
    elements.append(init_elem)

    while len(elements) != nb_of_elems:
        while len(current_ends) != 0:
            end = current_ends[0]
            next_elems_indexes = np.unique(np.where(prev_descriptors == end.descriptor)[0])
            ## Check if the element is a final one (no other policy starting from it)
            if len(next_elems_indexes) == 0:
                leaf_elems.append(end)
            ## Iterate over all policies starting from this element
            for next_elem_index in next_elems_indexes:
                ## Skip the init elem
                if next_elem_index == 0:
                    continue
                ## Reconstruct the element
                elem = Element(policy_parameters=params[next_elem_index],
                               descriptor=descriptors[next_elem_index],
                               previous_element=end)
                elements.append(elem)
                current_ends.append(elem)
            ## Remove the element whom children just got reconstructed
            current_ends.pop(0)

    # ## Run all leaf elements so that we fill back fully each elem found
    # for el in leaf_elems:
    #     gym_env.reset()
    #     ## reconstruct needed-policy chaining (deterministic-case)
    #     policies_to_chain = []
    #     len_under_policy = []
    #     budget_used = 0
    #     transitions = []
    #     policies_to_chain.insert(0, el.policy_parameters)
    #     len_under_policy.insert(0, len(el.trajectory))
    #     prev_el = el.previous_element
    #     ## WARNING: might have to copy values
    #     while prev_el != None:
    #         # if prev_el.policy_parameters != []:
    #         if prev_el.policy_parameters is not None:
    #             policies_to_chain.insert(0, prev_el.policy_parameters)
    #             len_under_policy.insert(0, len(prev_el.trajectory))
    #         prev_el = prev_el.previous_element
    #     ## Replay policies from initial state to el goal state
    #     obs = gym_env.get_obs()
    #     ## Check if el is init elem
    #     # if el.policy_parameters == []:
    #     if el.policy_parameters is None:
    #         transitions.append((None, obs))
    #         return transitions, budget_used
    #     for policy_params, h in zip(policies_to_chain, len_under_policy):
    #         self.controller.set_parameters(policy_params)
    #         for _ in range(h):
    #             action = self.controller(obs)
    #             transitions.append((action, obs))
    #             obs, reward, done, info = gym_env.step(action)
    #             budget_used += 1
    #     transitions.append((None, obs))
    return elements, leaf_elems

    
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print(f'usage: {sys.argv[0]} folderpath_with_npzs')
        exit()
    folderpath = sys.argv[1]
    print(f"coucou {sys.argv[1]}")

    params, descs, prev_descs = load_archive_data(folderpath)

    print(f"Loaded archive data from folder {folderpath}")
    gym_env = gym.make('BallInCup3d-v0')
    
    elements, leaf_elems = reconstruct_elements(params, descs, prev_descs, gym_env)

    print(f"Reconstructed all elements. Total number of elements: {len(elements)}")
    print(f"Total number of leaf elements: {len(leaf_elems)}")
    exit()
    nb_of_tests = 10
    
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
        'controller_type': NeuralNetworkController,
        'controller_params': controller_params,

        'action_min': -1,
        'action_max': 1,
        
        'budget': budget,
        'exploration_horizon': 10,
        'nb_eval_exploration': 10,
        'nb_thread_exploration': 6,

        'archive_type': 'cell',
        'fixed_grid_min': -0.5,
        'fixed_grid_max': 0.5,
        'fixed_grid_div': 10,
        
        'policy_param_init_min': -5,
        'policy_param_init_max': 5,

        'model_update_rate': 10,
        'dynamics_model_params': dynamics_model_params,

        'dump_rate': 50,
        'nb_of_samples_per_state':10,
    }
    
    ## Pass compared model training set on arg 1
    compared_model_transitions = np.load(sys.argv[1])
    ## Pass perfect model training set on arg 2
    perfect_model_transitions = np.load(sys.argv[2])
    ## Pass test set on arg 3
    test_transitions = np.load(sys.argv[3])

    compared_model_mean_mses = []
    perfect_model_mean_mses = []

    for _ in range(nb_of_tests):
        ## Create models
        compared_model = DynamicsModel(params=params)
        perfect_model = DynamicsModel(params=params)

        ## Add respective transitions to each DM
        for cmp_trs, perf_trs in zip(compared_model_transitions, perfect_model_transitions):
            compared_model.add_samples_from_transitions(cmp_trs)
            perfect_model.add_samples_from_transitions(perf_trs)

        ## Train both models
        compared_model.train()
        perfect_model.train()

        compared_model_mean_mse = 0
        perfect_model_mean_mse = 0

        total_trs = 0
    
        for test_trs in test_transitions:
            if len(test_trs) == 1:
                continue
            for i in range(len(test_trs) - 1):
                a = test_trs[i][0]
                s = test_trs[i][1]
                ns = test_trs[i+1][1]
                
                ## Infer using both models
                cmp_ns, _ = compared_model.forward(a, s)
                perfect_ns, _ = perfect_model.forward(a, s)
                
                cmp_mse = np.square(np.subtract(cmp_ns, ns)).mean()
                perfect_mse = np.square(np.subtract(cmp_ns, ns)).mean()
                compared_model_mean_mse += cmp_mse
                perfect_model_mean_mse += perfect_mse
                total_trs += 1

        compared_model_mean_mse /= total_trs
        perfect_model_mean_mse /= total_trs

        compared_model_mean_mses.append(compared_model_mean_mse)
        perfect_model_mean_mses.append(perfect_model_mean_mse)

        compared_model_mean_error = np.mean(compared_model_mean_mses)
        compared_model_var_error = np.var(compared_model_mean_mses)

        perfect_model_mean_error = np.mean(compared_model_mean_mses)
        perfect_model_var_error = np.var(compared_model_mean_mses)

