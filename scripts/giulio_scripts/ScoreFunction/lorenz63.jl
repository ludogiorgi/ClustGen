using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using MarkovChainHammer
using ClustGen
using KernelDensity
using HDF5
using Flux
using BSON
using Plots
using LinearAlgebra
using ProgressBars
using Distributions

include("/Users/giuliodelfelice/Desktop/MIT/ClustGen/scripts/giulio_scripts/run_experiment_lorenz63.jl")  # import Function run_experiment
@isdefined run_experiments
# Define the rhs of the Lorenz system for later integration with evolve function, changes wrt GMM_Lorenz63: x is a 4 dimensional vector that contains x, y1,y2 and y3.
function F(x, t, σ, ε ; µ=10.0, ρ=28.0, β=8/3)
    dx = x[1] * (1 - x[1]^2) + (σ / ε) * x[3]
    dy1 = µ/ε^2 * (x[3] - x[2])
    dy2 = 1/ε^2 * (x[2] * (ρ - x[4]) - x[3])
    dy3 = 1/ε^2 * (x[2] * x[3] - β * x[4])
    return [dx, dy1, dy2, dy3]
end

function sigma(x, t; noise = 0.0)
    sigma1 = noise
    sigma2 = noise
    sigma3 = noise
    sigma4 = noise #Added: This is for the 4th variable
    return [sigma1, sigma2, sigma3, sigma4]
end

function normalize_f(f, x, t, M, S)
    return f(x .* S .+ M, t) .* S
end



#call function run_experiment
result = run_experiments(fix_initial_state=false,
    σ=0.08,
    ε=0.5,
    σ_value=0.05,
    prob=0.001,
    conv_param=0.002,
    n_epochs=5000,
    batch_size=16,
    label="test_run", test_number=11, save_figs=false)

println("D_eff: ", result.D_eff)
println("Final NN loss: ", result.loss)