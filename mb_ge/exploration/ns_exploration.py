## Data manipulation imports
import numpy as np
from itertools import repeat
from sklearn.neighbors import KDTree

## Multiprocessing imports
from multiprocessing import Pool

## Novelty search imports
from deap import tools, base, algorithms

## Local imports
from mb_ge.exploration.exploration_method import ExplorationMethod
from mb_ge.exploration.random_exploration import RandomExploration
from mb_ge.controller.nn_controller import NeuralNetworkController
from mb_ge.utils.element import Element

class NoveltySearchExploration(ExplorationMethod):
    def __init__(self, params=None):
        super().__init__(params=params)
        self._re_procedure = RandomExploration(params=params)
        self._archive_kdt = None
        self._archive_list = []

    def _process_params(self, params):
        super()._process_params(params)

    def _compute_spent_budget(self, elements):
        return sum([len(el.trajectory) for el in elements])

    def _mutate(self, genotype):
        """
        Mutates a genotype and returns it.
        Args:
            genotype: genotype to mutate
        Returns:
            mutated: mutated genotype to be used for an offspring
        """
        pass

    def _select(self, elements, n=6, crit='novelty'):
        """
        Selects n elements to add to the archive, based on a given criterion
        Args:
            elements: elements to select from
            n: number of elements to add to archive
            crit: criterion to use to determine which element to add to the archive
        Returns:
            selected: selected elements to add to the archive
        """
        pass
    
    def _explore(self, gym_env, prev_element, exploration_horizon):
        ## NS Params
        cxpb = 0
        mutpb = 1
        indpb = 0.1
        eta_m = 15.0
        pop_size = 100
        lambda_ = int(2.*pop_size)

        ind_size = len(self.controller.get_parameters())

        min_ = self.policy_param_init_min
        max_ = self.policy_param_init_max

        nb_eval = 0

        toolbox = base.Toolbox()
        
        toolbox.register("attr_float", lambda : random.uniform(min_, max_))
        
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=params["ind_size"])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        #toolbox.register("mate", tools.cxBlend, alpha=params["alpha"])
    
        # Polynomial mutation with eta=15, and p=0.1 as for Leni
        toolbox.register("mutate", tools.mutPolynomialBounded, eta=eta_m, indpb=indpb,
                         low=min_, up=max_)

        toolbox.register("select", tools.selBest, fit_attr='novelty')

        toolbox.register("evaluate", evaluate)

        population = toolbox.population(n=pop_size)
        
        pool = Pool(processes=self.nb_thread)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        nb_eval+=len(invalid_ind)
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        # fit is a list of fitness (that is also a list) and behavior descriptor
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fit = fit[0] # fit is an attribute just used to store the fitness value
            ind.parent_bd=None
            ind.bd=listify(fit[1])
            ind.id = generate_uuid()
            ind.parent_id = None

        for ind in population:
            ind.am_parent=0
        
        archive=updateNovelty(population,population,None,params)

        isortednov=sorted(range(len(population)), key=lambda k: population[k].novelty, reverse=True)
        for i,ind in enumerate(population):
            ind.rank_novelty=isortednov.index(i)
            ind.dist_to_parent=0
            ind.fitness.values=ind.fit
        gen=0    
        
        if self.archive_kdt is not None: ## general case
            # Vary the population
            for gen in range(1,nb_gen):
                offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                nb_eval+=len(invalid_ind)

                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fit = fit[0]
                    ind.fitness.values = fit[0]
                    ind.parent_bd=ind.bd
                    ind.parent_id=ind.id
                    ind.id = generate_uuid()
                    ind.bd=listify(fit[1])
                for ind in population:
                    ind.am_parent=1
                for ind in offspring:
                    ind.am_parent=0

                pq=population+offspring
                archive=updateNovelty(pq,offspring,archive,params, pop_for_novelty_estimation)
                isortednov=sorted(range(len(pq)), key=lambda k: pq[k].novelty, reverse=True)
                
                for i,ind in enumerate(pq):
                    ind.rank_novelty=isortednov.index(i)
                    if (ind.parent_bd is None):
                        ind.dist_to_parent=0
                    else:
                        ind.dist_to_parent=np.linalg.norm(np.array(ind.bd)-np.array(ind.parent_bd))
                    ind.fitness.values=ind.fit
                population[:] = toolbox.select(pq, pop_size)     

        else: ## Initialization
            elements = self._re_procedure(gym_env, prev_element, exploration_horizon)
            
        pool.close()

        return elements, self._compute_spent_budget(elements)
