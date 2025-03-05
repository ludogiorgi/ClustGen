# ClustGen

A comprehensive toolkit for clustering-based generative modeling of dynamical systems. This Julia package provides tools for analyzing, clustering, and generating trajectories from dynamical systems using various machine learning and statistical techniques.

## Overview

ClustGen implements novel algorithms for estimating score functions and response properties of nonlinear stochastic systems. It combines clustering-based techniques with generative modeling to efficiently and accurately characterize complex dynamical systems, with particular applications in climate science and reduced-order modeling.

Key capabilities include:
- Score-based generative modeling of dynamical systems
- Generalized Fluctuation-Dissipation Theorem (GFDT) implementations
- Clustering algorithms for efficient statistical estimation
- Neural network autoencoders for dimensionality reduction
- Sampling methods for data generation
- Diffusion models with customizable noise schedules
- Visualization tools for high-dimensional data

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/ClustGen.jl")