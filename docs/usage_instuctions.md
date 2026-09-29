# Using the code

To run the code using the default settings navigate to the scripts folder by using:
``` 
cd scripts
```
then to run the code using the default settings use:
```
python infiltration_simulation.py
``` 
This will produce an output of csv files for u in the `results/` folder. To visualise the results of the simulation, run
```
python create_plots.py --potential --nfig=NFIG
```
where NFIG is a number corresponding to the number of figures you want to create. This will produce an animation of the CSV files you produced.


The structure of the repository is

```
.infiltration_workspace
|-AI4Flood/
| |-boundary_conditions.py
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
|-plots
|-results
|-scripts
|-tests
```
