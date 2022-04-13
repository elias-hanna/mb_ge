#Model-Based Go-Explore
This repository contains the code to run the model-based go-explore algorithm.

## Remarks on hyperparameters

### Exploration Horizon (exploration_horizon)

The exploration horizon is the maximum number of steps an exploratory policy can make.
Important remark: if keeping only one element per cell (usually the one that has the shortest trajectory to reach the cell), there needs to be a trade-off between exploration horizon and cell size, so that the maximum distance covered by the policy is equal is >= cell size

### Number of exploratory policies (nb_eval_exploration)

The number of exploratory policies is the number of policies that are tried starting from the state that was selected from the archive.
Important remark: the more exploratory policies we try, the more precise selection we can make afterwards, or the more direct exploration we do.

### Number of grid divisions (fixed_grid_div)

The number of cells for each dimension of the grid with fixed cell size.

Important remark: the size of each cell is determined by taking the fixed_grid_min and fixed_grid_max parameters and the number of cells you want to divide the grid in with fixed_grid_div.

### Model update rate (model_update_rate)

When using the algorithm iterations to update the model, this parameter specifies the update rate of the learned model.

### Epoch mode (epoch_mode)

This is for the determination of the current horizon value. There is three possible values. 'model_update', 'fixed_steps' and 'unique_fixed_steps'.
'model_update': value based on the model update rate.
'fixed_steps': value based on the number of real environment steps achieved.
'unique_fixed_steps': value based on the number of unique real environment steps achieved.
