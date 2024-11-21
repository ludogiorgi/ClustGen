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
using Zygote
using StatsBase

include("autoencoder.jl")
include("preprocessing.jl")
include("training.jl")
include("sampling.jl")
include("noising_schedules.jl")
include("visualization.jl")
include("generate.jl")

function rk4_step!(u, dt, f)
    k1 = f(u)
    k2 = f(u .+ 0.5 .* dt .* k1)
    k3 = f(u .+ 0.5 .* dt .* k2)
    k4 = f(u .+ dt .* k3)
    @inbounds u .= u .+ (dt / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
end

export apply_autoencoder, read_autoencoder
export f_tilde
export train, check_loss
export sample_reverse, sample_langevin
export σ_variance_exploding, g_variance_exploding
export vectorfield2d, meshgrid
export potential_data, ∇U_1D, ∇U_2D

end