
Data and code for publication #########.


## expt_data
`data.py` contains all experimental data used for fitting.

## code
`solve_for_rate_constants.py` runs a particle swarm optimization which minimizes the squared error between the model 
and experimental data by modify 6 constants.
`plot_data.py` plots experimental data and fits.
`constants.py` file containing physical constants.
`ode_function.py` file contains kinetic model.
`particle_swarm.py` is a particle swarm optimization algorithm. 

## fit_data
Contains the fits for each condition formatted: {expt_id}-pH{#}-Fe{loading}-O{O2 pressure}.csv
