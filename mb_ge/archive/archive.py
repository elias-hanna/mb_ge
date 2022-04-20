from abc import abstractmethod
import os
import numpy as np
import copy

class Archive():
    def __init__(self, params=None):
        ## Archive is a dict
        self._archive = dict()
    
    def _process_params(self, params):
        if 'dump_path' in params:
            self.dump_path = params['dump_path']
        else:
            curr_dir = os.getcwd()
            self.dump_path = curr_dir
    
    @abstractmethod
    def add(self, element):
        """
        Adds an element to the archive, given the archive adding rules.

        Args:
            element: element to be added to the archive
        """
        raise NotImplementedError

    def get_all_elements(self):
        all_elements = []
        for cell in self._archive.values():
            all_elements += cell._elements
        return all_elements
        
    @abstractmethod
    def compare(self, element1, element2):
        """
        Compare two elements given the archive rules.

        Args:
            element1: element to be compared to element2
            element2: element to be compared to element1

        Returns:
            result: 1 if element1 > element2, -1 if element2 > element1
        """
        raise NotImplementedError

    def dump_archive(self, dump_path, budget_used, itr, plot_disagr=False, plot_novelty=False):
        path_to_dir_to_create = os.path.join(self.dump_path, f'results_{itr}')
        os.makedirs(path_to_dir_to_create, exist_ok=True)
        self.visualize(budget_used, itr=itr, plot_disagr=plot_disagr, plot_novelty=plot_novelty)
        total_num_of_els = 0
        all_max_len_desc = []
        all_max_len_params = []
        all_max_len_traj = []
        for cell in self._archive.values():
            total_num_of_els += len(cell._elements)
            len_desc_in_cell = []
            len_params_in_cell = []
            len_traj_in_cell = []
            for el in cell._elements:
                if el.descriptor is not None and el.policy_parameters is not None \
                   and el.trajectory is not None:
                    len_desc_in_cell.append(len(el.descriptor))
                    len_params_in_cell.append(len(el.policy_parameters))
                    len_traj_in_cell.append(len(el.trajectory))
                if len_desc_in_cell!=[] and len_params_in_cell!=[] and len_traj_in_cell!=[]:
                    all_max_len_desc.append(max(len_desc_in_cell))
                    all_max_len_params.append(max(len_params_in_cell))
                    all_max_len_traj.append(max(len_traj_in_cell))
                    
        # total_num_of_els = sum([len(cell._elements)
        # for cell in self.state_archive._archive.values()])
        # len_desc = len(list(self.state_archive._archive.values())[0]._elements[0].descriptor)
        # len_params = len(list(self.state_archive._archive.values())[0]._elements[0].policy_parameters)
        # max_traj_len = max([len(list(self.state_archive._archive.values())[0]._elements)])
        if all_max_len_desc != [] and all_max_len_params != [] and all_max_len_traj != []:
            len_desc = max(all_max_len_desc)
            len_params = max(all_max_len_params)
            len_traj = max(all_max_len_traj)
            descriptors = np.zeros((total_num_of_els, len_desc))
            prev_descriptors = np.zeros((total_num_of_els, len_desc))
            params = np.zeros((total_num_of_els, len_params))
            policy_horizon = np.zeros((total_num_of_els))
            count = 0
            for key in self._archive.keys():
                for el in self._archive[key]._elements:
                    descriptors[count, :] = copy.copy(el.descriptor)
                    if el.previous_element is not None:
                        prev_descriptors[count, :] = copy.copy(el.previous_element.descriptor)
                    else:
                        prev_descriptors[count, :] = copy.copy(el.descriptor)
                    params[count, :] = copy.copy(el.policy_parameters)
                    policy_horizon[count] = len(el.trajectory)
                    count += 1
            np.savez(f'{dump_path}/results_{itr}/descriptors', descriptors)
            np.savez(f'{dump_path}/results_{itr}/prev_descriptors', prev_descriptors)
            np.savez(f'{dump_path}/results_{itr}/params', params)
            np.savez(f'{dump_path}/results_{itr}/policy_horizon', policy_horizon)
                # for key in self.state_archive._archive.keys():
                    # np.savez_compressed(f'{self.dump_path}/results_{itr}/archive_cell_{key}_itr_{itr}',
                            # self.state_archive._archive[key]._elements)

    def _prepare_plot(self, plt, fig, ax):
        ## Set plot labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ## Set plot limits
        x_min = y_min = z_min = self._grid_min
        x_max = y_max = z_max = self._grid_max
        
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        ax.set_zlim(z_min,z_max)

        ## Set axes ticks (for the grid)
        ticks = [self._grid_min + i*(self._grid_max - self._grid_min)/self._grid_div
                 for i in range(self._grid_div)]
        
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_zticks(ticks)

        ## Add grid to plot 
        plt.grid(True,which="both", linestyle='--')

        ## Set ticks label to show label for only 10 ticks
        mod_val = len(ticks)//10 if len(ticks) > 50 else len(ticks)//5
        ticks_labels = [round(ticks[i],2) if (i)%mod_val==0 else ' ' for i in range(len(ticks))]
        ticks_labels[-1] = self._grid_max
        ax.set_xticklabels(ticks_labels)
        ax.set_yticklabels(ticks_labels)
        ax.set_zticklabels(ticks_labels)

        ## Invert zaxis for the plot to be like reality
        plt.gca().invert_zaxis()
        
    def visualize(self, curr_budget, itr=0, show=False, plot_disagr=False, plot_novelty=False):
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        import numpy as np
        
        x = []
        y = []
        z = []
        disagrs = []
        novelties = []
        ## Add the BD data from archive:
        for key in self._archive.keys():
            elements = self._archive[key].get_elements()
            for el in elements:
                x.append(el.descriptor[0])
                y.append(el.descriptor[1])
                z.append(el.descriptor[2])
                disagrs.append(el.mean_disagr)
                novelties.append(el.novelty)

        ## Create fig and ax
        fig = plt.figure(figsize=(8, 8), dpi=160)  
        ax = fig.add_subplot(111, projection='3d')  

        self._prepare_plot(plt, fig, ax)

        ## Scatter plot 
        ax.scatter(x, y, z)

        # ## Set plot labels
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        # ## Set plot limits
        # x_min = y_min = z_min = self._grid_min
        # x_max = y_max = z_max = self._grid_max
        
        # ax.set_xlim(x_min,x_max)
        # ax.set_ylim(y_min,y_max)
        # ax.set_zlim(z_min,z_max)

        # ## Set axes ticks (for the grid)
        # ticks = [self._grid_min + i*(self._grid_max - self._grid_min)/self._grid_div
        #          for i in range(self._grid_div)]
        
        # ax.set_xticks(ticks)
        # ax.set_yticks(ticks)
        # ax.set_zticks(ticks)

        # ## Add grid to plot 
        # plt.grid(True,which="both", linestyle='--')

        # ## Set ticks label to show label for only 10 ticks
        # mod_val = len(ticks)//10 if len(ticks) > 50 else len(ticks)//5
        # ticks_labels = [round(ticks[i],2) if (i)%mod_val==0 else ' ' for i in range(len(ticks))]
        # ticks_labels[-1] = self._grid_max
        # ax.set_xticklabels(ticks_labels)
        # ax.set_yticklabels(ticks_labels)
        # ax.set_zticklabels(ticks_labels)

        # ## Invert zaxis for the plot to be like reality
        # plt.gca().invert_zaxis()

        ## Set plot title
        plt.title(f'State Archive at {curr_budget} evaluations', fontsize=8)

        ## Save fig
        plt.savefig(f"{self.dump_path}/results_{itr}/state_archive_at_{curr_budget}_eval", bbox_inches='tight')
        
        if plot_disagr:
            fig = plt.figure(figsize=(8, 8), dpi=160)  
            ax = fig.add_subplot(111, projection='3d')  

            self._prepare_plot(plt, fig, ax)

            min_disagr = np.min(disagrs)
            max_disagr = np.max(disagrs)
            norm_disagr =  (disagrs - np.min(disagrs))/(np.max(disagrs)-np.min(disagrs))

            cm = plt.cm.get_cmap()
            sc = ax.scatter(x, y, z, c=norm_disagr, cmap=cm)
            clb = plt.colorbar(sc)
            ## Round for cleaner plot
            min_disagr = round(min_disagr,4)
            max_disagr = round(max_disagr,4)
            clb.set_label('Normalized model ensemble disagreement for each reached state')
            clb.ax.set_title(f'min disagr={min_disagr} and max disagr={max_disagr}')

            ## Set plot title
            plt.title(f'State Archive at {curr_budget} evaluations', fontsize=8)

            ## Save fig
            plt.savefig(f"{self.dump_path}/results_{itr}/disagr_state_archive_at_{curr_budget}_eval", bbox_inches='tight')

        if plot_novelty:
            fig = plt.figure(figsize=(8, 8), dpi=160)  
            ax = fig.add_subplot(111, projection='3d')  

            self._prepare_plot(plt, fig, ax)

            min_nov = np.min(novelties)
            max_nov = np.max(novelties)
            norm_nov =  (novelties - np.min(novelties))/(np.max(novelties)-np.min(novelties))

            cm = plt.cm.get_cmap()
            sc = ax.scatter(x, y, z, c=norm_nov, cmap=cm)
            clb = plt.colorbar(sc)
            ## Round for cleaner plot
            min_nov = round(min_nov,4)
            max_nov = round(max_nov,4)

            clb.set_label('Normalized novelty for each reached state')
            clb.ax.set_title(f'min nov={min_nov} and max nov={max_nov}')

            ## Set plot title
            plt.title(f'State Archive at {curr_budget} evaluations', fontsize=8)

            ## Save fig
            plt.savefig(f"{self.dump_path}/results_{itr}/novelty_state_archive_at_{curr_budget}_eval", bbox_inches='tight')

        if show:
            plt.show()
        plt.close()
