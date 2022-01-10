## Data manipulation imports
import numpy as np
from itertools import repeat
from sklearn.neighbors import KDTree

## Multiprocessing imports
from multiprocessing import Pool

## Local imports
from mb_get.exploration.exploration_method import ExplorationMethod
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

        pool = Pool(processes=self.nb_thread)

        if self.archive_kdt is not None: ## general case
            pass
        else: ## Initialization
            elements = self._re_procedure(gym_env, prev_element, exploration_horizon)
            
        pool.close()

        return elements, self._compute_spent_budget(elements)
