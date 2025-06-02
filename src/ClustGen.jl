"""
ClustGen

A comprehensive toolkit for clustering-based generative modeling of dynamical systems.
This package provides tools for analyzing, clustering, and generating trajectories 
from dynamical systems using various machine learning and statistical techniques.
"""
module ClustGen
__precompile__(false)
# Core dependencies
using StateSpacePartitions
using LinearAlgebra
using Random
using Distributed
using SharedArrays

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
using ProgressMeter
using GLMakie

# Numerical utilities
using QuadGK
using SpecialFunctions

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
include("io.jl")                # I/O functions for saving/loading models and data
include("diffusion_matrix.jl")  # Diffusion matrix estimation
include("responses.jl")         # Response function generators
include("KSE_integrate.jl")     # KSE integration functions

# ===== Exported functions =====
# Autoencoder functionality
export apply_autoencoder, read_autoencoder

# Clustering and preprocessing
export f_tilde, f_tilde_ssp, generate_inputs_targets, f_tilde_labels

# Model training
export train, check_loss

# Sampling methods
export sample_reverse, sample_langevin, sample_langevin_Σ

# Noise schedules
export σ_variance_exploding, g_variance_exploding

# Visualization
export vectorfield2d, meshgrid

# Data generation
export evolve, evolve_ens

# Statistical utilities
export covariance, cleaning
export computeSigma
export compute_density_from_score, maxent_distribution


# Response functions
export generate_numerical_response, generate_score_response, generate_numerical_response_f,
       generate_numerical_response_HO, generate_numerical_response_f_HO

# KSE integration
export KSE_integrate, dealias, domain, field2vector, vector2field, create_ks_animation, 
       reduce_fourier_energy, reconstruct_physical_from_reduced

# ===== I/O functions =====
export save_variables_to_hdf5, read_variables_from_hdf5, save_current_workspace, 
       load_workspace_from_file

end # module ClustGen