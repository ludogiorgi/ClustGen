module ClustGen

using StateSpacePartitions
using LinearAlgebra
using Random 
using ProgressBars
using Statistics
using KernelDensity
using HDF5
using Flux
using QuadGK
using BSON
using CUDA
using Plots
# using Zygote
using StatsBase

include("autoencoder.jl")
include("preprocessing.jl")
include("training.jl")
include("sampling.jl")
include("noising_schedules.jl")
include("visualization.jl")
include("generate.jl")
include("utils.jl")

export apply_autoencoder, read_autoencoder
export f_tilde, f_tilde_labels, generate_inputs_targets
export train, check_loss
export sample_reverse, sample_langevin, sample_langevin_Σ
export σ_variance_exploding, g_variance_exploding
export vectorfield2d, meshgrid
export potential_data, ∇U_1D, ∇U_2D, simulate_lorenz96
export covariance, rk4_step!, computeSigma

end