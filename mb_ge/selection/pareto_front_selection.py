import random
import numpy as np

import kneed

## Local imports
from mb_ge.selection.state_disagreement_selection import StateDisagreementSelection

class ParetoFrontSelection(StateDisagreementSelection):        
    def _process_params(self, params):
        super()._process_params(params)

    def select_element_from_cell_archive(self, archive):
        all_elements = archive.get_all_elements()
        all_elements_ordered, mean_disagrs = self._batch_eval_all_elements(all_elements)

        novelty = []
        
        for i in range(len(all_elements_ordered)):
            # mean_disagrs[i] = round(mean_disagrs[i], 3)
            # novelty.append(round(all_elements_ordered[i].novelty, 3))
            novelty.append(all_elements_ordered[i].novelty)
        
        non_dominated_disagr = []
        non_dominated_novelty = []
        non_dominated_idx = []
        for i in range(len(mean_disagrs)):
            dominated = False
            loc_d = mean_disagrs[i]
            loc_n = novelty[i]
            for j in range(len(mean_disagrs)):
                if loc_d < mean_disagrs[j] and loc_n < novelty[j]:
                    dominated = True
                    break
            if not dominated:
                non_dominated_disagr.append(loc_d)
                non_dominated_novelty.append(loc_n)
                non_dominated_idx.append(i)
                
        ## Sort by nov
        # sorted_pairs = sorted(zip(novelty, disagr))
        # tuples = zip(*sorted_pairs)
        # s_nov, s_disagr = [list(t) for t in tuples] 
        # kneedle = kneed.KneeLocator(s_nov, s_disagr)
        # plt.ylabel('Mean disagreeement along trajectory')
        # plt.xlabel('Novelty')
        ## Sort by disagr
        # sorted_pairs = sorted(zip(disagr, novelty))
        if len(non_dominated_disagr) >= 2 and len(set(non_dominated_novelty)) > 1:
            non_dominated_disagr.reverse()
            non_dominated_novelty.reverse()
            non_dominated_idx.reverse()
            
            kneedle = kneed.KneeLocator(non_dominated_disagr, non_dominated_novelty,)
                                        # curve='concave', direction='decreasing')#, interp_method='polynomial')
            # kneedle.plot_knee_normalized()
            # import matplotlib.pyplot as plt
            # plt.xlabel('Mean disagreeement along trajectory')
            # plt.ylabel('Novelty')
            # plt.show()
            sel_idx = non_dominated_novelty.index(kneedle.knee_y) # y-axis is nov
            # print('novelty and disagr of ind choosed: ', kneedle.knee_y, ' ', kneedle.knee)
            # print('max nov and max disagr in pop: ', max(non_dominated_novelty), ' ',
                  # max(non_dominated_disagr))
            # print('num of inds on pareto front: ', len(non_dominated_novelty))
            return all_elements_ordered[non_dominated_idx[sel_idx]]
        else:
            return all_elements_ordered[non_dominated_idx[0]]
        return None

    def select_element_from_element_list(self, elements):
        all_elements_ordered, mean_disagrs = self._batch_eval_all_elements(elements)

        novelty = []
        
        for i in range(len(all_elements_ordered)):
            # mean_disagrs[i] = round(mean_disagrs[i], 3)
            # novelty.append(round(all_elements_ordered[i].novelty, 3))
            novelty.append(all_elements_ordered[i].novelty)
        
        non_dominated_disagr = []
        non_dominated_novelty = []
        non_dominated_idx = []
        for i in range(len(mean_disagrs)):
            dominated = False
            loc_d = mean_disagrs[i]
            loc_n = novelty[i]
            for j in range(len(mean_disagrs)):
                if loc_d < mean_disagrs[j] and loc_n < novelty[j]:
                    dominated = True
                    break
            if not dominated:
                non_dominated_disagr.append(loc_d)
                non_dominated_novelty.append(loc_n)
                non_dominated_idx.append(i)
                
        ## Sort by nov
        # sorted_pairs = sorted(zip(novelty, disagr))
        # tuples = zip(*sorted_pairs)
        # s_nov, s_disagr = [list(t) for t in tuples] 
        # kneedle = kneed.KneeLocator(s_nov, s_disagr)
        # plt.ylabel('Mean disagreeement along trajectory')
        # plt.xlabel('Novelty')
        ## Sort by disagr
        # sorted_pairs = sorted(zip(disagr, novelty))
        if len(non_dominated_disagr) >= 2 and len(set(non_dominated_novelty)) > 1:
            non_dominated_disagr.reverse()
            non_dominated_novelty.reverse()
            non_dominated_idx.reverse()
            
            kneedle = kneed.KneeLocator(non_dominated_disagr, non_dominated_novelty,)
                                        # curve='concave', direction='decreasing')#, interp_method='polynomial')
            # kneedle.plot_knee_normalized()
            # import matplotlib.pyplot as plt
            # plt.xlabel('Mean disagreeement along trajectory')
            # plt.ylabel('Novelty')
            # plt.show()
            sel_idx = non_dominated_novelty.index(kneedle.knee_y) # y-axis is nov
            
            return all_elements_ordered[non_dominated_idx[sel_idx]]
        else:
            return all_elements_ordered[non_dominated_idx[0]]
        return None
