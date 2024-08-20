module ClustGen

using StateSpacePartitions
using Distances
using LinearAlgebra
using Random 
using ParallelKMeans
using ProgressBars
using Statistics
using Flux
using BSON
using HDF5
using Plots

include("autoencoder.jl")
include("preprocessing.jl")
include("training.jl")
include("sampling.jl")

export apply_autoencoder, read_autoencoder
export f_tilde
export train_clustered, train_vanilla, check_loss
export sample

end