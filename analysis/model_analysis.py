# Data manipulation includes
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing

# OS manipulation includes
import sys
import os
import natsort
import time
import random

# MB-GE includes
import gym
from mb_ge.models.dynamics_model import DynamicsModel
from mb_ge.utils.element import Element
from mb_ge.archive.fixed_grid_archive import FixedGridArchive
from mb_ge.controller.nn_controller import NeuralNetworkController

def load_archive_data(folderpath):
    params = None
    # params = np.load(os.path.join(folderpath,'params.npz'))['arr_0']
    descriptors = np.load(os.path.join(folderpath,'descriptors.npz'))['arr_0']
    prev_descriptors = np.load(os.path.join(folderpath,'prev_descriptors.npz'))['arr_0']

    return params, descriptors, prev_descriptors

def reconstruct_elements(params, descriptors, prev_descriptors, gym_env, execute_policies=False):
    # assert len(params) == len(descriptors) == len(prev_descriptors)
    assert len(descriptors) == len(prev_descriptors)
    nb_of_elems = len(descriptors)
    elements = []
    leaf_elems = []
    current_ends = []
    # init_elem = Element(policy_parameters=params[0], descriptor=descriptors[0])
    init_elem = Element(descriptor=descriptors[0])
    current_ends.append(init_elem)
    elements.append(init_elem)

    from collections import Counter
    all_indexes = []
    while current_ends != []:
        end = current_ends[0]
        # next_elems_indexes = np.unique(np.where((prev_descriptors == end.descriptor).all())[0])
        # next_elems_indexes = np.unique(np.where((prev_descriptors == end.descriptor))[0])
        next_elems_indexes = []

        efef = time.time()
        for i in range(1, nb_of_elems):
            if (end.descriptor == prev_descriptors[i]).all():
                next_elems_indexes.append(i)
        # print(time.time()-efef)
        # all_indexes += list(next_elems_indexes)
        # count = Counter(all_indexes)

        ## Check if the element is a final one (no other policy starting from it)
        if len(next_elems_indexes) == 0:
            leaf_elems.append(end)
        ## Iterate over all policies starting from this element
        for next_elem_index in next_elems_indexes:
            ## Skip the init elem
            # if next_elem_index == 0:
                # continue
            ## Reconstruct the element
            # elem = Element(policy_parameters=params[next_elem_index],
            elem = Element(descriptor=descriptors[next_elem_index],
                           previous_element=end)
            elements.append(elem)
            current_ends.append(elem)
        ## Remove the element whom children just got reconstructed
        current_ends.pop(0)
        # print(len(elements)/nb_of_elems)
    # while len(elements) < nb_of_elems:
    #     while len(current_ends) != 0:
    #         end = current_ends[0]
    #         next_elems_indexes = np.unique(np.where(prev_descriptors == end.descriptor)[0])
    #         ## Check if the element is a final one (no other policy starting from it)
    #         if len(next_elems_indexes) == 0:
    #             leaf_elems.append(end)
    #         ## Iterate over all policies starting from this element
    #         for next_elem_index in next_elems_indexes:
    #             ## Skip the init elem
    #             if next_elem_index == 0:
    #                 continue
    #             ## Reconstruct the element
    #             # elem = Element(policy_parameters=params[next_elem_index],
    #             elem = Element(descriptor=descriptors[next_elem_index],
    #                            previous_element=end)
    #             elements.append(elem)
    #             current_ends.append(elem)
    #         ## Remove the element whom children just got reconstructed
    #         current_ends.pop(0)
    #         print(len(elements)/nb_of_elems)

    #     ## Quick fix need to look at above
    #     iter_nums_i = list(range(len(elements)))
    #     for i in iter_nums_i:
    #         iter_nums_j = list(range(i+1,len(elements)))
    #         for j in iter_nums_j:
    #             if all(elements[i].descriptor == elements[j].descriptor):
    #             # if all(elements[i].policy_parameters == elements[j].policy_parameters):
    #                 elements.pop(j)
    #                 iter_nums_i.pop()
    #                 iter_nums_j.pop()
        
    if execute_policies:
    ## Run all leaf elements so that we fill back fully each elem found
        for el in leaf_elems:
            gym_env.reset()
            ## reconstruct needed-policy chaining (deterministic-case)
            policies_to_chain = []
            len_under_policy = []
            budget_used = 0
            transitions = []
            policies_to_chain.insert(0, el.policy_parameters)
            len_under_policy.insert(0, len(el.trajectory))
            prev_el = el.previous_element
            ## WARNING: might have to copy values
            while prev_el != None:
                if prev_el.policy_parameters is not None:
                    policies_to_chain.insert(0, prev_el.policy_parameters)
                    len_under_policy.insert(0, len(prev_el.trajectory))
                    prev_el = prev_el.previous_element
                    ## Replay policies from initial state to el goal state
            obs = gym_env.get_obs()
            ## Check if el is init elem
            # if el.policy_parameters == []:
            if el.policy_parameters is None:
                transitions.append((None, obs))
                return transitions, budget_used
            for policy_params, h in zip(policies_to_chain, len_under_policy):
                self.controller.set_parameters(policy_params)
                for _ in range(h):
                    action = self.controller(obs)
                    transitions.append((action, obs))
                    obs, reward, done, info = gym_env.step(action)
                    budget_used += 1
            transitions.append((None, obs))
    return elements, leaf_elems

# def reconstruct_elements2(params, descriptors, prev_descriptors, gym_env, execute_policies=False):
#     # assert len(params) == len(descriptors) == len(prev_descriptors)
#     assert len(descriptors) == len(prev_descriptors)
#     nb_of_elems = len(descriptors)
#     elements = []
#     leaf_elems = []
#     current_ends = []
#     # init_elem = Element(policy_parameters=params[0], descriptor=descriptors[0])
#     init_elem = Element(descriptor=descriptors[0])
#     current_ends.append(init_elem)
#     elements.append(init_elem)

#     for i in range(len(descriptors)):

#         for j in range(1, nb_of_elems):
#             if (prev_descriptors[i] == descriptors[j]).all():
#                 next_elems_indexes.append(i)
                
#         new_elem = Element(descriptor=descriptors[i],
#                            previous_element=)

#     while current_ends != []:
#         end = current_ends[0]
#         # next_elems_indexes = np.unique(np.where((prev_descriptors == end.descriptor).all())[0])
#         # next_elems_indexes = np.unique(np.where((prev_descriptors == end.descriptor))[0])
#         next_elems_indexes = []
        
#         for i in range(1, nb_of_elems):
#             if (end.descriptor == prev_descriptors[i]).all():
#                 next_elems_indexes.append(i)
        
#         ## Check if the element is a final one (no other policy starting from it)
#         if len(next_elems_indexes) == 0:
#             leaf_elems.append(end)
#         ## Iterate over all policies starting from this element
#         for next_elem_index in next_elems_indexes:
#             ## Skip the init elem
#             # if next_elem_index == 0:
#                 # continue
#             ## Reconstruct the element
#             # elem = Element(policy_parameters=params[next_elem_index],
#             elem = Element(descriptor=descriptors[next_elem_index],
#                            previous_element=end)
#             elements.append(elem)
#             current_ends.append(elem)
#         ## Remove the element whom children just got reconstructed
#         current_ends.pop(0)

def compute_coverage_and_reward_for_rep(rep_dir):
        coverages = []
        rewarding_pi_count = []
        
        rep_path = os.path.join(folderpath, rep_dir)
        iter_dirs = next(os.walk(rep_path))[1]
        iter_dirs = natsort.natsorted(iter_dirs)
        real_iter_dirs = [d for d in iter_dirs if 'sim' not in d]
        sim_iter_dirs = [d for d in iter_dirs if 'sim' in d]
        print(f"Currently processing {rep_path}")
        # for iter_dir in iter_dirs:
        early_exit_cpt = 0
        for iter_dir in real_iter_dirs:
            iterpath = os.path.join(rep_path, iter_dir)
            print(f"Currently processing {iterpath}")

            pi_params, descs, prev_descs = load_archive_data(iterpath)

            gym_env = gym.make('BallInCup3d-v0')

            reconstruct_start_time = time.time()
            elements, leaf_elems = reconstruct_elements(pi_params, descs, prev_descs, gym_env,
                                                        execute_policies=False)
            print(f"Took {time.time()-reconstruct_start_time} second to reconstruct elements")
            
            archive = FixedGridArchive(params=params)

            reconstruct_start_time = time.time()            
            for el in elements:
                archive.add(el)
            print(f"Took {time.time()-reconstruct_start_time} second to add elements to archive")

            ## Compute coverage (number of filled bins vs total number of bins)
            coverage = len(archive._archive.keys())/(params['fixed_grid_div']**3)
            coverages.append(coverage)
            
            ## Compute number of policies reaching reward state
            target_size = gym_env.sim.model.site_size[1, [0, 1, 2]]
            ball_size = gym_env.sim.model.geom_size[2, 0]

            nb_of_rewarded_elems = 0
    
            reconstruct_start_time = time.time()            
            for el in elements:
                if float(all(el.descriptor < target_size - ball_size)):
                    nb_of_rewarded_elems += 1
            print(f"Took {time.time()-reconstruct_start_time} seconds to compute rewarded elements")
            rewarding_pi_count.append(nb_of_rewarded_elems)

            early_exit_cpt += 1
            if early_exit_cpt > 4:
                return (coverages, rewarding_pi_count)
        return (coverages, rewarding_pi_count)

################################################################################
################################################################################

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print(f'usage: {sys.argv[0]} path_of_folder_with_res_repetitions')
        exit()

    folderpath = sys.argv[1]

    splitted_foldername = list(filter(None, folderpath.split('/')))
    run_name = splitted_foldername[-2] + '_' +splitted_foldername[-1]
    
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
        
        'budget': 1000000,
        'exploration_horizon': 100,
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

        'epoch_mode': "model_update",
        'steps_per_epoch': 1000,
        'use_variable_model_horizon': False,
        'min_horizon': 1,
        'max_horizon': 25,
        'horizon_starting_epoch': 20,
        'horizon_ending_epoch': 100,
        
        'dump_rate': 200,
        'nb_of_samples_per_state':10,
        'dump_all_transitions': False,
    }
    rep_dirs = next(os.walk(folderpath))[1]

    number_of_reps = len(rep_dirs)

    max_values = []
    for rep_dir in rep_dirs:
        print(rep_dir)
        rep_path = os.path.join(folderpath, rep_dir)
        iter_dirs = next(os.walk(rep_path))[1]
        iter_dirs = natsort.natsorted(iter_dirs)
        real_iter_dirs = [d for d in iter_dirs if 'sim' not in d]
        real_iter_dirs_no_final = [d for d in real_iter_dirs if 'final' not in d]
        sim_iter_dirs = [d for d in iter_dirs if 'sim' in d]
        sim_iter_dirs_no_final = [d for d in sim_iter_dirs if 'final' not in d]
        values = [iter_dir.split('_')[-1] for iter_dir in real_iter_dirs_no_final]
        
        if len(values) > len(max_values):
            max_values = values

    rewarding_pis_vals = np.empty((number_of_reps, len(max_values)))
    rewarding_pis_vals[:] = np.nan
    coverage_vals = np.empty((number_of_reps, len(max_values)))
    coverage_vals[:] = np.nan
    
    curr_rep = 0

    # Create multiprocessing pool
    pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)

    results = pool.map(compute_coverage_and_reward_for_rep, [rep_dir for rep_dir in rep_dirs])

    for rep_res in results:
        coverage_vals[curr_rep, :len(rep_res[0])] = rep_res[0]
        rewarding_pis_vals[curr_rep, :len(rep_res[1])] = rep_res[1]
        curr_rep += 1
        
    # #################################################################################
    # #################################################################################
    # for rep_dir in rep_dirs:
    #     rep_path = os.path.join(folderpath, rep_dir)
    #     iter_dirs = next(os.walk(rep_path))[1]
    #     iter_dirs = natsort.natsorted(iter_dirs)
    #     print(f"Currently processing {rep_path}")
    #     curr_iter = 0
    #     for iter_dir in iter_dirs:
    #         iterpath = os.path.join(rep_path, iter_dir)
    #         print(f"Currently processing {iterpath}")

    #         pi_params, descs, prev_descs = load_archive_data(iterpath)

    #         gym_env = gym.make('BallInCup3d-v0')

    #         reconstruct_start_time = time.time()
    #         elements, leaf_elems = reconstruct_elements(pi_params, descs, prev_descs, gym_env,
    #                                                     execute_policies=False)
    #         print(f"Took {time.time()-reconstruct_start_time} second to reconstruct elements")
            
    #         archive = FixedGridArchive(params=params)

    #         reconstruct_start_time = time.time()            
    #         for el in elements:
    #             archive.add(el)
    #         print(f"Took {time.time()-reconstruct_start_time} second to add elements to archive")

    #         ## Compute coverage (number of filled bins vs total number of bins)
    #         coverage = len(archive._archive.keys())/(params['fixed_grid_div']**3)
    #         coverage_vals[curr_rep, curr_iter] = coverage

    #         ## Compute number of policies reaching reward state
    #         target_size = gym_env.sim.model.site_size[1, [0, 1, 2]]
    #         ball_size = gym_env.sim.model.geom_size[2, 0]

    #         nb_of_rewarded_elems = 0
    
    #         reconstruct_start_time = time.time()            
    #         for el in elements:
    #             if float(all(el.descriptor < target_size - ball_size)):
    #                 nb_of_rewarded_elems += 1
    #         print(f"Took {time.time()-reconstruct_start_time} seconds to compute rewarded elements")

    #         rewarding_pis_vals[curr_rep, curr_iter] = nb_of_rewarded_elems
    #         # print(f"Number of elements reaching reward zone: {nb_of_rewarded_elems}")
            
    #         ## Increment
    #         curr_iter += 1
    #     curr_rep += 1
    coverage_mean = np.nanmean(coverage_vals, axis = 0)
    reward_mean = np.nanmean(rewarding_pis_vals, axis = 0)

    coverage_error = np.nanstd(coverage_vals, axis = 0)
    reward_error = np.nanstd(rewarding_pis_vals, axis = 0)

    label = [int(val) for val in max_values]

    plt.figure()

    plt.plot(label, coverage_mean, 'k-')
    plt.fill_between(label, coverage_mean-coverage_error, coverage_mean+coverage_error,
                     facecolor='green', alpha=0.5)
    plt.title(f"Coverage depending on number of iterations for {run_name}")
    plt.savefig(f"coverage_{run_name}.jpg")

    # pos = np.interp(label, coverage_mean, label)
    # plt.xticks(pos, label)
    
    plt.figure()
    plt.plot(label, reward_mean, 'k-')
    plt.fill_between(label, reward_mean-reward_error, reward_mean+reward_error,
                     facecolor='green', alpha=0.5)
    plt.title(f"Number of rewarded policies depending on number of iterations for {run_name}")
    plt.savefig(f"reward_{run_name}.jpg")

    plt.show()

    
    exit()
    
    #########################################################################
    #########################################################################
    #########################################################################
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

