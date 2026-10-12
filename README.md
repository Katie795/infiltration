# A Neural Physics Model for the Richards Equation for Soil Infiltration

This repository contains a novel approach for solving the Richards equation using the Neural Physics method. Neural Physics implements a finite element method on neural network architectures, taking advantage of the computational speedup available on GPUs using functionality from PyTorch.

## Abstract
A novel approach to discretise and solve the Richards equation based on the finite element method (FEM) is presented. The new approach, called the Neural Physics method, expresses the discretisation as a convolution neural network (CNN) where the weights are predetermined by the integrals of the basis functions in the FEM. The Neural Physics method is physically consistent and represents a discretisation of the governing equation while also gaining several advantages by being written in machine-learning software: the code is platform-agnostic and can run on CPUs, GPUs and the latest AI accelerators, resulting in an increase in computational speed; the gradient is stored by default, thus rendering a completely differentiable method suitable for optimisation problems; and it can be easily integrated with trained neural networks representing sub-grid scale processes, inputs, or boundary conditions. 
In this code, the Neural Physics method is demonstrated for two different discretisations for the Richards equation, including a multigrid method that is realised through a U-Net architecture. The accuracy and speed of these methods is tested on a problem with a known analytical solution and compared to a standard FEM, and obtains favourable results. Finally, the Neural Physics method is applied to a real-world test case with integrated experimental data and simulates the infiltration of water through a rain garden.

## Key Features:
- __Platform-Agnostic Code__: Runs seamlessly on CPUs, GPUs and AI optimised processors.
- __Neural Network Integration__: Easily integrates with trained neural networks for sub-grid scale models, surrogate models, or physics informed neural networks.
- __Accellerated development__: Leverages machine learning libraries to speed up model development.

## Applications in this code:
- __Idealised problems__: Validation against an infiltration problem with an analytical solution.
- __Real-world test case__: The code is applied to a 2D cross section of a rain garden with material properties of the soil derived fom experimental data. There is a choice between ponding or rainfall flux boundary conditions on the surface of the soil. A drain at the bottom of the rain garden is represented by seepage boundary conditions.


## Installation:
### Prerequisites
If you use Windows you might need to set up a Windows Subsystem for Linux (WSL).
To run this package, it will be helpful to have
- __Python 3.10__: This will be installed as a part of the virtual environment. Python 3.10 is used for compatibility with the required packages and libraries.
- __Conda (Preferred)__: It is recommended to use use Conda to simplify the management of your Python environment and dependencies. The installation instructions shown here are for the conda environment.
- __GPU with CUDA support (Optional)__: This code is most effective when run on a GPU, but it is perfectly possible to run it on a CPU as well. If using a GPU, please make sure that it supports CUDA and the CUDA drivers are correctly installed and configured.


First, use the terminal to create a folder on your computer then navigate to the folder:
```
mkdir infiltration_workspace
```
```
cd infiltration_workspace
```
Clone this repository into your folder:
```
git clone https://github.com/Katie795/infiltration.git
```
You will need a GitHub account and need to be added as a collaborator on the project. If you are not yet, please email me at km2560@bath.ac.uk. If you get an error `Password authentication is not supported for Git operations`, make sure your password is a Personal access token that can be found in your GitHub account: Click your avatar in the top-right corner → Settings → Developer settings → Personal access tokens → Tokens (classic).

To set up the requirement and install the required packages, run the following commands:
```
cd infiltration
conda env create -f environment.yml
```
Alternatively, you can install the required packages by running
```
pip install -r requirements.txt
```
If that doesn't work, just run
```
conda create --name flooding-env python=3.10
conda activate flooding-env
python -m pip install --editable .
```
When you run the code, if you get a `ModuleNotFoundError: No module named 'MODULE'`, run

```
conda install MODULE
```
or, if that doesn't work,
```
pip install MODULE
```
You might have to install multiple packages before the code runs.

## Using the code

To run the code using the default settings navigate to the `scripts` folder by using:
``` 
cd scripts
```
then to run the code using the default settings use:
```
python infiltration_simulation.py
``` 
This will produce an output of csv files for the capillary potential u in the `results/` folder. If you get a `results folder not found` error run
```
mkdir ../results
```

To visualise the results of the simulation, run
```
python create_plots.py --potential --nfig=72
```
where 72 is a number corresponding to the number of frames in the animation you want to create. This will produce an animation of the CSV files you produced.

Detailed instructions on how to modify the code can be found in the `docs` directory.

## License 
This project is licensed under the MIT Lisence. See the `LICENSE` file for more details.

## Citation
If you use this project or the related papaer in your research, please cite it as follows:

### Paper Citation
[Insert citation]

### GitHub Repository Citation:

```
@misc{mackenzie2026infiltration,
  author = {Katherine MacKenzie},
  title = {A Neural Physics Model for the Richards Equation for Soil Infiltration},
  year = {2026},
  url = {https://github.com/Katie795/infiltration}
}
```

## Contact and related papers

For more information please contact me at `km2560@bath.ac.uk`.

