# Data manipulation includes
import numpy as np

import sys
from mb_ge.models.dynamics_models import DynamicsModel
from mbn

def load_archive_data(folder_path):
    descriptors = np.load('descriptors.npz')['arr_0']
    prev_descriptors = np.load('prev_descriptors.npz')['arr_0']
    params = np.load('params.npz')['arr_0']

    return params, descriptors, prev_descriptors

def reconstruct_elements(params, descriptors, prev_descriptors):
    assert len(params) == len(descriptors) == len(prev_descriptors)
    nb_of_elems = len(params)
    elements = []
    current_ends = []
    init_elem = Element(policy_parameters=params[0], descriptor=descriptors[0])
    current_ends.append(init_elem)
    elements.append(init_elem)
    
    while len(elements) != nb_of_elems:
        for end in current_ends:
            next_elems_indexes = np.unique(np.where(prev_descriptors == end.descriptor)[0])
            for next_elem_index in next_elems_indexes:
                ## Skip the init elem
                if next_elem_index == 0:
                    continue
                elem = Element(policy_parameters=params[next_elem_index],
                               descriptor=descriptor[next_elem_index],
                               previous_element=end)
                elements.append(elem)
                current_ends.append(elem)
            current_ends.pop(0)

    return elements
    
if __name__ == '__main__':

    if len(sys.argv) < 3:
        print(f'usage: {sys.argv[0]} compared_model_training_set perfect_model_training_set test_set')

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

