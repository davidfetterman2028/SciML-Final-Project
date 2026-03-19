# SciML-Final-Project

## Overview
This project demonstrates the use of **Physics-Informed Neural Networks (PINNs)** to approximate the solutions of **ideal and damped harmonic oscillators**. PINNs embed the governing physics directly into the loss function, allowing the network to learn solutions without requiring dense labeled datasets.

The harmonic oscillator systems considered include:

- **Ideal oscillator (undamped):**  
  \[
  x'' + \omega_0^2 x = 0
  \]

- **Damped oscillator:**  
  \[
  x'' + 2 \gamma \omega_0 x' + \omega_0^2 x = 0
  \]

Analytical solutions are used as reference for evaluation:

- Ideal: \(x(t) = \cos(\omega t)\)  
- Damped: \(x(t) = e^{-\gamma t/2} \cos(\omega_d t)\), with \(\omega_d = \sqrt{\omega^2 - \gamma^2/4}\)

## Dependencies
The project requires the following Python packages:

- `numpy`
- `matplotlib`
- `torch` (PyTorch)

## File Structure

SciML-Final-Project.ipynb – Jupyter notebook containing the full workflow for FDM solutions, PINN setup, training, and evaluation.

IdealOscillatorPINNPrediction.png – PINN vs FDM comparison for the ideal oscillator.

DampedOscillatorPINNPrediction.png – PINN vs FDM comparison for the damped oscillator.

## Methodology

# Finite Difference Method (FDM)
Used to compute the true solution for both ideal and damped oscillators. Sparse noisy data points are sampled from this solution (after the addition of random noise) for PINN training.

# Physics-Informed Neural Network

Fully connected network with:

Input layer: 
t

Two hidden layers (64 neurons each) with tanh activation

Output layer: predicted solution 
x(t)

Xavier initialization was used for weights
Automatic differentiation computes 
x′ and x′′

Loss function accounts for and includes:

1) PDE residual

2) Initial conditions

3) Sparse noisy data

4) Boundary conditions (not used here)

# Training
Optimizer: Adam, learning rate 
10^-3
Number of iterations: 1500
Randomly sample collocation, initial, and data points during each iteration
Weighted loss components:

wPDE=1.0
wIC=5.0
wData=10.0
wBC=1.0

# Evaluation

Compare PINN predictions with FDM solutions for both ideal and damped oscillators

Visualize results using matplotlib

Usage

1) Run the FDM simulation to generate the reference solution.

2) Sample sparse noisy data from the FDM solution.

3) Train the PINN using the provided architecture and loss function.

4) Compute PINN predictions and plot the results againts true solution for evaluation.
Results

PINN accurately captures the dynamics of both ideal and damped oscillators.

Minor discrepancies between PINN and FDM are due to data noise and stochastic training.

Figures

Ideal Oscillator: PINN prediction matches FDM solution closely.
Damped Oscillator: PINN captures damping behavior and oscillatory dynamics effectively.

Author

David Fetterman – jst3867
March 18, 2026
