## Data manipulation imports
import numpy as np
from itertools import repeat
from sklearn.neighbors import KDTree
import random
from copy import copy

## Multiprocessing imports
from multiprocessing import Pool

## Novelty search imports
from deap import tools, base, algorithms

## Local imports
from mb_ge.exploration.exploration_method import ExplorationMethod
from mb_ge.exploration.random_exploration import RandomExploration
from mb_ge.controller.nn_controller import NeuralNetworkController
from mb_ge.utils.element import Element


archive_list = []

# archive_add_mode = 'novelty'
# archive_add_mode = 'random'
archive_nb_to_add = 6
nb_nearest_neighbors = 15
gen = 0

class NoveltySearchExploration(ExplorationMethod):
    def __init__(self, params=None):
        super().__init__(params=params)
        self._re_procedure = RandomExploration(params=params)
        self._archive_kdt = None
        self._archive_list = []

    def _process_params(self, params):
        super()._process_params(params)

    def _eval_element(self, x, gym_env, prev_element):
        ## Create a copy of the controller
        controller = self.controller.copy()
        ## Verify that x and controller parameterization have same size
        # assert len(x) == len(self.controller.get_parameters())
        ## Set controller parameters
        controller.set_parameters(x)

        env = copy(gym_env) ## need to verify this works
        env.set_state(prev_element.sim_state['qpos'], prev_element.sim_state['qvel'])
        traj = []
        actions = []
        obs = prev_element.trajectory[-1]
        cum_rew = 0
        ## WARNING: need to get previous obs
        for _ in range(self.exploration_horizon):
            action = controller(obs)
            obs, reward, done, info = env.step(action)
            cum_rew += reward
            traj.append(obs)
            actions.append(action)
        element = Element(descriptor=traj[-1][:3], trajectory=traj, actions=actions,
                          reward=cum_rew, policy_parameters=x, previous_element=prev_element,
                          sim_state={'qpos': copy(env.sim.data.qpos),
                                     'qvel': copy(env.sim.data.qvel)})
        ## WARNING: Need to add a bd super function somewhere in params or in Element I guess
        return element

    def _eval_element_on_model(self, x, model, prev_element):
        ## Create a copy of the controller
        controller = self.controller.copy()
        ## Verify that x and controller parameterization have same size
        # assert len(x) == len(self.controller.get_parameters())
        ## Set controller parameters
        controller.set_parameters(x)

        traj = []
        actions = []
        obs = prev_element.trajectory[-1]
        cum_rew = 0
        disagreements = []
        ## WARNING: need to get previous obs
        for _ in range(self.exploration_horizon):
            action = controller(obs)
            next_step_pred, disagreement = model.forward(action, obs, mean=False, disagr=True)
            ## Compute mean prediction from model samples
            mean_pred = [np.mean(next_step_pred[:,i]) for i in range(len(next_step_pred[0]))]
            obs = mean_pred
            traj.append(mean_pred)
            disagreements.append(disagreement)
            actions.append(action)
        element = Element(descriptor=traj[-1][:3], trajectory=traj, actions=actions,
                          disagreement = disagreements,
                          reward=cum_rew, policy_parameters=x, previous_element=prev_element,)
        ## WARNING: Need to add a bd super function somewhere in params or in Element I guess
        return element
    
    def _explore(self, gym_env_or_model, prev_element, exploration_horizon, eval_on_model=False):
        ## Set exploration horizon (here and not in params because it might be dynamic)
        self.exploration_horizon = exploration_horizon
        ## NS Params
        cxpb = 0
        mutpb = 1
        indpb = 0.1
        eta_m = 15.0
        pop_size = 10
        lambda_ = int(2.*pop_size)

        ind_size = len(self.controller.get_parameters())
        nb_gen = 50

        min_ = self.policy_param_init_min
        max_ = self.policy_param_init_max

        nb_eval = 0

        if eval_on_model:
            eval_func = self._eval_element_on_model
        else:
            eval_func = self._eval_element

        ## List that keeps archived behaviour descriptors
        archive_bd_list = []
        archive_elements_list = []
        
        gen = 0
        population = []
        gen_bd_list = []

        ## Some DEAP tools
        toolbox = base.Toolbox()
        ## Polynomial mutation with eta=15, and p=0.1 as for Leni
        toolbox.register("mutate", tools.mutPolynomialBounded, eta=eta_m, indpb=indpb,
                         low=min_, up=max_)

        pool = Pool(processes=self.nb_thread)

        ## Initialize population
        to_evaluate = []
        for _ in range(pop_size):
            ## Create a random policy parametrization 
            x = np.random.uniform(low=self.policy_param_init_min,
                                  high=self.policy_param_init_max,
                                  size=ind_size)
            to_evaluate += [x]
        env_map_list = [gym_env_or_model for _ in range(self.nb_eval)]
        ## Evaluate all generated elements on given environment
        population = []
        if eval_on_model:
            for xx in to_evaluate:
                population.append(eval_func(xx, gym_env_or_model, prev_element))
        else:
            population = pool.starmap(eval_func, zip(to_evaluate, repeat(gym_env_or_model),
                                                     repeat(prev_element)))
        ## Add random elements to the archive
        random.shuffle(population)
        archive_elements_list += population[:archive_nb_to_add]
        ## Add elements bd list to archive_bd_list
        archive_bd_list += [el.descriptor for el in population[:archive_nb_to_add]]
        
        itr = 0            
        for i in range(nb_gen):
            print(f'### Generation {i} of Local NS search around BD:{prev_element.descriptor} ###')
            ## Variation
            # generate lambda_*pop_size offspring, to_evaluate contains params !
            to_evaluate = []
            for _ in range(lambda_*pop_size):
                ## Select a random individual from previous population
                prev_el = random.choice(population)
                ## Copy params
                params = copy(prev_el.policy_parameters)
                ## Mutate copy
                toolbox.mutate(params)
                ## Add params to be evaluated
                to_evaluate += [x]
                
            env_map_list = [gym_env_or_model for _ in range(self.nb_eval)]
            ## Evaluate offsprings on given environment
            offsprings = []
            if eval_on_model:
                for xx in to_evaluate:
                    offsprings.append(eval_func(xx, gym_env_or_model, prev_element))
            else:
                offsprings = pool.starmap(eval_func, zip(to_evaluate, repeat(gym_env_or_model),
                                                   repeat(prev_element)))
            ##Update generation bd list
            gen_bd_list = [el.descriptor for el in offsprings]
            # np.savez(f"bd_list_{gen}", gen_bd_list)

            ## Update generation kdt
            # Get KDT for archive + current pop
            gen_kdt = KDTree(archive_bd_list+gen_bd_list, leaf_size=30, metric='euclidean')
            ## Evaluate the offspring
            if archive_bd_list != []: # General case
                for el in offsprings:
                    ## Get k-nearest neighbours to this ind
                    k_dists, k_indexes = gen_kdt.query([el.descriptor], k=nb_nearest_neighbors)
                    el.novelty = sum(k_dists[0])/nb_nearest_neighbors
            else:
                for el in offsprings:
                    el.novelty = 0. # Initialization
    
            ## Sort offsprings by novelty
            sorted_offsprings_by_nov = sorted(offsprings, key=lambda el: el.novelty, reverse=True)
            ## Add to archive
            itr = 0
            ## Add most novel elements to the archive
            archive_elements_list += sorted_offsprings_by_nov[:archive_nb_to_add]
            ## Add most novel elements bd to the archive of bd (for kdt)
            archive_bd_list += [el.descriptor for el
                                in sorted_offsprings_by_nov[:archive_nb_to_add]]
            ## Sort previous pop + offspring by novelty
            sorted_pq_by_nov = sorted(population+offsprings,
                                      key=lambda el: el.novelty, reverse=True)
            ## Replace pop with most novel individuals from population + offspring
            population = sorted_pq_by_nov[:pop_size]

            gen += 1
            
        pool.close()

        return archive_elements_list, self._compute_spent_budget(archive_elements_list)
