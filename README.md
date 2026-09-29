# A Neural Physics Model for the Richards Equation for Soil Infiltration

This repository contains a novel approach and solution for solving the Richards equation using the Neural Physics method. In this case, Neural Physics implements a fintite element method for on neural network architectures, taking advantage of the computational speedup available on GPUs.

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
To use this package with a GPU, you need to have PyTorch installed with CUDA. It is possible to install this code without using a GPU using conda. If you use Windows you might need to set up a Windows Subsystem for Linux (WSL)

First, use the terminal to create a folder on your computer 
```
mkdir infiltration_workspace
```
then navigate to the folder
```
cd infiltration_workspace
```

