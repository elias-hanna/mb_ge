from abc import abstractmethod

class Archive():
    def __init__(self, params=None):
        ## Archive is a dict
        self._archive = dict()

    
    def _process_params(self, params):
        pass
    
    @abstractmethod
    def add(self, element):
        """
        Adds an element to the archive, given the archive adding rules.

        Args:
            element: element to be added to the archive
        """
        raise NotImplementedError

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

    def visualize(self, curr_budget, show=False, mode='3d'):
        import matplotlib.pyplot as plt
        import numpy as np

        ## Create the grid
        x_min = y_min = z_min = self._grid_min
        x_max = y_max = z_max = self._grid_max

        fig = plt.figure()  
        ax = fig.add_subplot(111, projection='3d')  
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        ax.set_zlim(z_min,z_max)

        ticks = [self._grid_min + i*(self._grid_max - self._grid_min)/self._grid_div
                 for i in range(self._grid_div)]
        # plt.xticks(ticks)
        # plt.yticks(ticks)
        # plt.zticks(ticks)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_zticks(ticks)

        plt.grid(True,which="both", linestyle='--')
        
        plt.title(f'State Archive at {curr_budget} evaluations', fontsize=8)
        
        x = []
        y = []
        z = []
        ## Add the BD data from archive:
        for key in self._archive.keys():
            elements = self._archive[key].get_elements()
            for el in elements:
                x.append(el.descriptor[0])
                y.append(el.descriptor[1])
                z.append(el.descriptor[2])

        ax.scatter(x, y, z)  
        plt.gca().invert_zaxis()
        plt.savefig(f"state_archive_at_{curr_budget}_eval", bbox_inches='tight')
        if show:
            plt.show()
