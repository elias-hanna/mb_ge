from abc import abstractmethod
import os
import numpy as np
import copy

class VisualizationMethod():
    def __init__(self, params=None):
        self._process_params(params=params)
        
    def _process_params(self, params):
        if 'dump_path' in params:
            self.dump_path = params['dump_path']
        else:
            curr_dir = os.getcwd()
            self.dump_path = curr_dir

    def prepare_plot(self, plt, fig, ax, mode='3d'):
        ## Set plot labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if mode == '3d':
            ax.set_zlabel('Z')
        
        ## Set plot limits
        x_min = y_min = z_min = self._grid_min
        x_max = y_max = z_max = self._grid_max
        
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        if mode == '3d':
            ax.set_zlim(z_min,z_max)

        ## Set axes ticks (for the grid)
        ticks = [self._grid_min + i*(self._grid_max - self._grid_min)/self._grid_div
                 for i in range(self._grid_div)]
        
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        if mode == '3d':
            ax.set_zticks(ticks)

        ## Add grid to plot 
        plt.grid(True,which="both", linestyle='--')

        ## Set ticks label to show label for only 10 ticks
        mod_val = len(ticks)//10 if len(ticks) > 50 else len(ticks)//5
        ticks_labels = [round(ticks[i],2) if (i)%mod_val==0 else ' ' for i in range(len(ticks))]
        ticks_labels[-1] = self._grid_max
        ax.set_xticklabels(ticks_labels)
        ax.set_yticklabels(ticks_labels)
        if mode == '3d':
            ax.set_zticklabels(ticks_labels)
            ## Invert zaxis for the plot to be like reality
            plt.gca().invert_zaxis()
        
    @abstractmethod
    def dump_plots(self, curr_budget, itr=0, show=False):
        """
        Dump plots to file system
        """
        raise NotImplementedError
