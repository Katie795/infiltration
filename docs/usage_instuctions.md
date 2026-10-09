

The structure of the repository is

```
.infiltration_workspace
|-AI4Flood/
| |-boundary_conditions.py # in development, do not use
| |-generate_csv_files.py 
| |-infiltration.py
| |-misc_functions.py
| |-operators.py
| |-richards_coefficients.py
| |-richards_solver.py
| |-swe_solver.py
|-data/
|-data_org/
|-docs/
| |- usage_instructions.md
|-numerical_experiments
| |- diffusion_test.py
| |- fem_solver.py
| |- horung_messing_FEM_test.py
| |- horung_messing_test.py
|-plots/
|-results/
|-scripts
| |- config_rain_garden.toml
| |- create_plot.py
| |- infiltration_simulation.py
| |- swe_simulation.py
|-tests/
| |- another_simple_test.py
| |- residual_test.py
| |- simple_test.py
| |- test_coefficients.py
| |- test_csvs.py
| |- test_jacobi.py
| |- test_plots.py
```

The parameters in `infiltration_simulation.py` can be changed via `config_rain_garden.toml`. This file looks like

### Configuration file for Malago road simulation
```
[geometry]
length = 4         # length of the rain garden (m)
depth  = 1.2       # depth of the rain garden (m)
drain_l = 1.775    # x-coordinate of left hand side drain (m)
drain_r = 2.225    # y-coordinate of right hand side of drain (m)

[mesh]
nx = 255           # number of elements in the x-direction
ny = 63            # number of elements in the y-direction

[time]
dt = 0.5          # timestep (s)
T = 86400            # total length of time (s)
nt = 172800        # total number of time-steps

[IO]
rainfall_data =    "../data/Clifton_rain_modifed_for_rainfall_event.csv" # csv file containing rainfall data
save_fig = true    # whether to save the results to be post-processed
n_out = 100        # how often to save results (every n_out timesteps)

[simulation]
device = "cuda:0"     # which device to run the simulation on. Options: cpu, cuda:0
surface = "rainfall" # which type of boundary conditions to use on the surface of the soil. Options: "ponding" or "rainfall"
model = "L" # which model to use. Options: "L" or "PC"
```

These options can be changed to modify the solver.

Note that if you are using the PC method, the nx +1 and ny +1 should be divisible be divisible by 8 in order for the multigrid method to work.


