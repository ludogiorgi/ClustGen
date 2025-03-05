# ClustGen

This code implements a hybrid methodology for efficient score function estimation and response analysis in nonlinear stochastic systems. It combines a clustering-based Gaussian Mixture Model approach with neural network interpolation to estimate the score function from large datasets, and leverages the Generalized Fluctuation-Dissipation Theorem (GFDT) to construct higher-order response functions. The code is demonstrated on reduced-order models—including triad systems and slow-fast models relevant to climate dynamics—validating its ability to accurately capture non-Gaussian effects and predict system responses to small perturbations.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/ClustGen.jl")
