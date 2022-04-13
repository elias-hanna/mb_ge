# Data manipulation includes
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import copy
import multiprocessing
from itertools import repeat

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

from count_bins import getBinsReachable

reachable_bins = 0

def load_archive_data(folderpath):
    params = np.load(os.path.join(folderpath,'params.npz'))['arr_0']
    descriptors = np.load(os.path.join(folderpath,'descriptors.npz'))['arr_0']
    prev_descriptors = np.load(os.path.join(folderpath,'prev_descriptors.npz'))['arr_0']
    policy_horizon = np.load(os.path.join(folderpath,'policy_horizon.npz'))['arr_0']

    return params, descriptors, prev_descriptors, policy_horizon

def reconstruct_elements(params, descriptors, prev_descriptors, gym_env,
                         execute_policies=False, policy_horizon=None):
    # assert len(params) == len(descriptors) == len(prev_descriptors)
    assert len(descriptors) == len(prev_descriptors)

    loc_descriptors = descriptors.copy()
    loc_prev_descriptors = prev_descriptors.copy()
    if execute_policies:
        loc_params = params.copy()
        loc_pi_h = policy_horizon.copy()
    
    elements = []
    leaf_elems = []
    current_ends = []
    # init_elem = Element(policy_parameters=params[0], descriptor=descriptors[0])
    el_params = None
    el_traj = None
    if execute_policies:
        el_params = params[0]
        el_traj = [0.]*round(policy_horizon[0])
    init_elem = Element(descriptor=descriptors[0],
                        policy_parameters=el_params, trajectory=el_traj)
    current_ends.append(init_elem)
    elements.append(init_elem)

    from collections import Counter
    all_indexes = []
    while current_ends != []:
        nb_of_elems = len(loc_prev_descriptors)
        end = current_ends[0]
        # next_elems_indexes = np.unique(np.where((prev_descriptors == end.descriptor).all())[0])
        next_elems_indexes = []
        for i in range(1, nb_of_elems):
            if (end.descriptor == loc_prev_descriptors[i]).all():
                next_elems_indexes.append(i)

        ## Check if the element is a final one (no other policy starting from it)
        if len(next_elems_indexes) == 0:
            leaf_elems.append(end)
        ## Iterate over all policies starting from this element
        for next_elem_index in next_elems_indexes:
            ## Reconstruct the element
            el_params = None
            el_traj = None
            if execute_policies:
                el_params = loc_params[next_elem_index]
                el_traj = [0.]*round(loc_pi_h[next_elem_index])
            elem = Element(descriptor=copy.copy(loc_descriptors[next_elem_index]),
                           previous_element=end, policy_parameters=el_params, trajectory=el_traj)
            elements.append(elem)
            current_ends.append(elem)
        ## Remove from local lists the elements that were reconstructed
        loc_prev_descriptors = np.delete(loc_prev_descriptors, next_elems_indexes, axis=0)
        loc_descriptors = np.delete(loc_descriptors, next_elems_indexes, axis=0)
        if execute_policies:
            loc_params = np.delete(loc_params, next_elems_indexes, axis=0)
            loc_pi_h = np.delete(loc_pi_h, next_elems_indexes, axis=0)
        ## Remove the element whom children just got reconstructed
        current_ends.pop(0)
        
    if execute_policies:
        global g_params
        controller = NeuralNetworkController(params=g_params)
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
                    print(prev_el.descriptor)
                    print(prev_el.previous_element)
                    prev_el = prev_el.previous_element
                    print(prev_el != None)
                    ## Replay policies from initial state to el goal state
            obs = gym_env.get_obs()
            ## Check if el is init elem
            # if el.policy_parameters == []:
            if el.policy_parameters is None:
                transitions.append((None, obs))
                return transitions, budget_used
            for policy_params, h in zip(policies_to_chain, len_under_policy):
                print(h)
                controller.set_parameters(policy_params)
                for _ in range(h):
                    action = controller(obs)
                    transitions.append((action, obs))
                    time.sleep(0.01)
                    obs, reward, done, info = gym_env.step(action)
                    budget_used += 1
            transitions.append((None, obs))
    return elements, leaf_elems

def compute_coverage_and_reward_for_rep(rep_dir):
    global reachable_bins
    coverages = []
    rewarding_pi_count = []
    
    rep_path = os.path.join(folderpath, rep_dir)
    iter_dirs = next(os.walk(rep_path))[1]
    iter_dirs = natsort.natsorted(iter_dirs)
    
    real_iter_dirs = [d for d in iter_dirs if 'sim' not in d]
    real_iter_dirs_no_final = [d for d in real_iter_dirs if 'final' not in d]
    sim_iter_dirs = [d for d in iter_dirs if 'sim' in d]
    sim_iter_dirs_no_final = [d for d in sim_iter_dirs if 'final' not in d]
    
    print(f"Currently processing {rep_path}")
    # for iter_dir in iter_dirs:
    early_exit_cpt = 0
    for iter_dir in real_iter_dirs_no_final:
        iterpath = os.path.join(rep_path, iter_dir)
        print(f"Currently processing {iterpath}")
        
        pi_params, descs, prev_descs, pi_h = load_archive_data(iterpath)

        gym_env = gym.make('BallInCup3d-v0')

        reconstruct_start_time = time.time()
        elements, leaf_elems = reconstruct_elements(pi_params, descs, prev_descs, gym_env,
                                                    execute_policies=True, policy_horizon=pi_h)
        print(f"Took {time.time()-reconstruct_start_time} second to reconstruct elements")
        
        archive = FixedGridArchive(params=params)
        
        reconstruct_start_time = time.time()            
        for el in elements:
            archive.add(el)
        print(f"Took {time.time()-reconstruct_start_time} second to add elements to archive")
        
        ## Compute coverage (number of filled bins vs total number of bins)
        # coverage = len(archive._archive.keys())/(params['fixed_grid_div']**3)
        coverage = len(archive._archive.keys())/reachable_bins
        # coverage = reachable_bins/(params['fixed_grid_div']**3)
        coverages.append(coverage)
        
        ## Compute number of policies reaching reward state
        target_size = gym_env.sim.model.site_size[1, [0, 1, 2]]
        ball_size = gym_env.sim.model.geom_size[2, 0]
        
        nb_of_rewarded_elems = 0
        
        reconstruct_start_time = time.time()            
        for el in elements:
            if float(all(abs(el.descriptor) < target_size - ball_size)):
                nb_of_rewarded_elems += 1
        print(f"Took {time.time()-reconstruct_start_time} seconds to compute rewarded elements")
        rewarding_pi_count.append(nb_of_rewarded_elems)
        
        early_exit_cpt += 1
        if early_exit_cpt > 20:
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
    g_params = \
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

    reachable_bins = getBinsReachable(g_params['fixed_grid_min'], g_params['fixed_grid_max'],
                                      g_params['fixed_grid_div'])
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
    # pool = multiprocessing.Pool(1)

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
    # import pdb; pdb.set_trace()

    label = [int(val) for val in max_values]

    ## Print coverage depending 
    coverage_target_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    corr_mean_budget = [0.]*len(coverage_target_vals)

    corr_budget = np.empty((number_of_reps, len(coverage_target_vals)))
    corr_budget[:] = np.nan

    # corr_budget = dict.fromkeys(coverage_target_vals, list())
    corr_budget = {k: [] for k in coverage_target_vals}
    
    eps = 0.01 # allow 1% error margin ?
    for i in range(len(coverage_target_vals)): # over each target val
        for j in range(len(coverage_vals)): # over reps
            for k in range(len(label)): # over each budget step
                a = abs(coverage_target_vals[i] - eps)
                if coverage_vals[j][k] >= abs(coverage_target_vals[i] - eps):
                    corr_budget[coverage_target_vals[i]].append(label[k])
                    # corr_budget[j][i] = label[k]
                    break

    plt.figure()

    ticks = [i+1 for i, v in enumerate(coverage_target_vals)]

    bplot1 = plt.boxplot(corr_budget.values(), patch_artist=True)
    
    plt.xticks(ticks=ticks, labels=[str(i) for i in coverage_target_vals])

    plt.savefig(f"budget_to_reach_{run_name}.jpg")

    ## Compute coverage and reward mean/error

    coverage_mean = np.nanmean(coverage_vals, axis = 0)
    reward_mean = np.nanmean(rewarding_pis_vals, axis = 0)

    coverage_error = np.nanstd(coverage_vals, axis = 0)
    reward_error = np.nanstd(rewarding_pis_vals, axis = 0)

    ## Save the computed data
    np.savez(f'{run_name}_data', coverage_mean=coverage_mean, coverage_error=coverage_error,
             reward_mean=reward_mean, reward_error=reward_error, budget_to_reach=corr_budget)    

    plt.figure()

    plt.plot(label, coverage_mean, 'k-')
    plt.fill_between(label, coverage_mean-coverage_error, coverage_mean+coverage_error,
                     facecolor='green', alpha=0.5)
    plt.title(f"Coverage depending on number of iterations for {run_name}")
    plt.savefig(f"coverage_{run_name}.jpg")

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
        compared_model = DynamicsModel(params=g_params)
        perfect_model = DynamicsModel(params=g_params)

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

