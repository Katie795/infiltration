# Project infiltration

## Current focus
- [] Run infiltration model for rainfall data and generate some plots for the paper
- [] Work through all tests and update them all
- [] Set up nonhomogeneous infiltration for illustration coupling shallow water - ponding allowed

## Next
- [] Add more tests to generating csvs
- [] Pick one test case and get it working
- [] Make the code CUDA friendly
- [] Write FEM solver for Richards to compare runtime

## Done 
- [x] Refactor richards_solver into parent and child classes
- [x] Get infiltration running in main file
- [x] Get test_csvs working and committed
- [x] Get test_coefficients working and committed
- [x] Check BC's are the right way round
- [x] Check u is negative everywhere
- [x] Write tests for the calculate dx functions
- [x] Set up contour plot for infiltration, is it doing the right thing?
- [x] Set up timings for simulations on CPUs and GPUs and run it for one day