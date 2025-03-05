"""
ClustGen

A comprehensive toolkit for clustering-based generative modeling of dynamical systems.
This package provides tools for analyzing, clustering, and generating trajectories 
from dynamical systems using various machine learning and statistical techniques.
"""
module ClustGen

# Core dependencies
using StateSpacePartitions
using LinearAlgebra
using Random

# Statistics and data processing
using Statistics
using StatsBase
using KernelDensity
using Optim

# I/O and visualization
using HDF5
using BSON
using Plots
using ProgressBars

# Numerical utilities
using QuadGK

# Deep learning
using Flux
using CUDA

# ===== Component modules =====
include("autoencoder.jl")       # Neural network autoencoders
include("preprocessing.jl")     # Data preprocessing utilities
include("training.jl")          # Model training functions
include("sampling.jl")          # Sampling methods
include("noising_schedules.jl") # Noise schedules for diffusion models
include("generate.jl")          # Data generation utilities
include("utils.jl")             # General utility functions
include("diffusion_matrix.jl")  # Diffusion matrix estimation
include("responses.jl")         # Response function generators

# ===== Exported functions =====
# Autoencoder functionality
export apply_autoencoder, read_autoencoder

# Clustering and preprocessing
export f_tilde, f_tilde_ssp, generate_inputs_targets

# Model training
export train, check_loss

# Sampling methods
export sample_reverse, sample_langevin, sample_langevin_Σ

# Noise schedules
export σ_variance_exploding, g_variance_exploding

# Visualization
export vectorfield2d, meshgrid

# Data generation
export evolve

# Statistical utilities
export covariance, cleaning
export computeSigma

# Response functions
export generate_numerical_response, generate_score_response, generate_numerical_response3

end # module ClustGen