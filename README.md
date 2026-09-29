# A Neural Physics Model for the Richards Equation for Soil Infiltration

This repository contains a novel approach and solution for solving the Richards equation using the Neural Physics method. Neural Physics implements a fintite element method for on neural network architectures, taking advantage of the computational speedup available on GPUs using functionality from PyTorch.

## Abstract
[Insert abstract from the paper]

## Key Features:
- __Platform-Agnostic Code__: Runs seamlessly on CPUs, GPUs and AI optimised processors.
- __Neural Network Integration__: Easily integrates with trained neural networks for sub-grid scale models, surrogate models, or physics informed neural networks.
- __Accellerated development__: Leverages machine learning libraries to speed up model development.

## Applications:
- __Idealised problems__: Validated against infiltration problem with analytical solution: the Horung-Messing problem,
-__Real-World test case__: Applied to a 2D cross section of a rain garden and compared to experimental data

## Installation:
### Prerequisites
If you use Windows you might need to set up a Windows Subsystem for Linux (WSL)
To run this package, it will be helpful to have
- __Python 3.10__: This will be installed as a part of the virtual environment. Python 3.10 is used for compatibility with the required packages and libraries.
- __Conda (Preferred)__: It is recommended to use use Conda to simplify the management of your Python environment and dependencies. The installation instructions shown here are for the conda environment.
- __GPU with CUDA support (Optional)__: This code gives the most speedup when it is run on a GPU, but it is also possible to run it on a CPU instead. If using a GPU, please make sure that it supports CUDA and the CUDA drivers are correctly installed and configured.


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
To set up the requirement and install the required pacakges, run:
```
conda env create -f environment.yml
```
Alternatively, you can install the required packages by running
```
pip install -r requirements.txt
```



Detailed instructions on how to run the code can be found in the docs directory.

## License 
This project is licensed under the MIT Lisence. See the LICENSE file for more details.

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

For more information please contact me at km2560@bath.ac.uk

