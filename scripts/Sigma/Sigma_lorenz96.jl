using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using ClustGen
using StateSpacePartitions
using LinearAlgebra
using Random 
using ProgressBars
using Statistics
using KernelDensity
using HDF5
using Flux
using Plots
using QuadGK
using BSON
using GLMakie
using GLMakie: xlims!, ylims!
using StatsBase
using MarkovChainHammer
import MarkovChainHammer.Trajectory: ContinuousTimeEmpiricalProcess
import LaTeXStrings
using ColorSchemes

##

function F(x, t; F0=6.0, nu=1.0, c=10.0, b=10.0, Nk=4, Nj=10)
    # Coupling constant: c1 = c/b
    c1 = c / b

    # Allocate arrays for the derivatives of the slow and fast variables.
    dx = zeros(Nk)
    dy = zeros(Nk, Nj)
    
    # Extract the slow variables xₖ from the state vector.
    x_slow = x[1:Nk]
    
    # Extract the fast variables yₖ,ⱼ.
    # They are assumed to be stored after the slow variables.
    # Reshape them into an Nk×Nj matrix, where the k-th row corresponds to the block for xₖ.
    x_fast = reshape(x[Nk+1:end], (Nj, Nk))'  # now x_fast[k, j] corresponds to yₖ,ⱼ

    # Compute the forcing for the slow variables (Eq. 10):
    # dxₖ/dt = - xₖ₋₁ (xₖ₋₂ - xₖ₊₁) - nu*xₖ + F0 + c1 * (sum of fast variables for mode k)
    for k in 1:Nk
        # Use periodic boundary conditions:
        # For index arithmetic in Julia (1-indexed):
        im1 = mod(k - 2, Nk) + 1  # index for xₖ₋₁
        im2 = mod(k - 3, Nk) + 1  # index for xₖ₋₂
        ip1 = mod(k, Nk) + 1      # index for xₖ₊₁
        dx[k] = - x_slow[im1]*(x_slow[im2] - x_slow[ip1]) - nu*x_slow[k] + F0 + c1*sum(x_fast[k, :])
    end

    # Compute the forcing for the fast variables (Eq. 11):
    # dyₖ,ⱼ/dt = - c*b * yₖ,ⱼ₊₁ (yₖ,ⱼ₊₂ - yₖ,ⱼ₋₁) - c*nu*yₖ,ⱼ + c1*xₖ
    for k in 1:Nk
        for j in 1:Nj
            # Periodic indices in the fast sub-block:
            jm1 = mod(j - 2, Nj) + 1   # index for yₖ,ⱼ₋₁
            jp1 = mod(j, Nj) + 1       # index for yₖ,ⱼ₊₁
            jp2 = mod(j + 1, Nj) + 1     # index for yₖ,ⱼ₊₂
            dy[k, j] = - c*b * x_fast[k, jp1]*(x_fast[k, jp2] - x_fast[k, jm1]) -
                       c*nu * x_fast[k, j] + c1*x_slow[k]
        end
    end

    # Combine the slow and fast derivatives into a single vector.
    # The slow derivatives come first, then the fast derivatives (flattened in row-major order).
    return vcat(dx, vec(transpose(dy)))
end


function sigma(x, t; noise = 0.2)
    return noise
end

function normalize_f(f, x, t, M, S)
    return f(x .* S .+ M, t) .* S
end

dim = 4
dt = 0.005
Nsteps = 20000000
obs_nn = evolve(0.01 .* randn(44), dt, Nsteps, F, sigma; resolution = 1)[1:4,:]

M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S

# kde_true_12 = kde((obs[1,:], obs[2,:]))
# kde_true_13 = kde((obs[1,:], obs[3,:]))
# kde_true_14 = kde((obs[1,:], obs[4,:]))

# plt1 = Plots.heatmap(kde_true_12.x, kde_true_12.y, kde_true_12.density, xlabel="X", ylabel="Y", title="True PDF")
# plt2 = Plots.heatmap(kde_true_13.x, kde_true_13.y, kde_true_13.density, xlabel="X", ylabel="Y", title="True PDF")
# plt3 = Plots.heatmap(kde_true_14.x, kde_true_14.y, kde_true_14.density, xlabel="X", ylabel="Y", title="True PDF")
# Plots.plot(plt1, plt2, plt3, layout=(1, 3), size=(1200, 400))
# ##

autocov_obs = zeros(dim, 300)
for i in 1:dim
    autocov_obs[i,:] = autocovariance(obs[i,:]; timesteps=300)
end

autocov_obs_mean = mean(autocov_obs, dims=1)

plotly()
Plots.plot(autocov_obs_mean[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")

##
obs_uncorr = obs[:, 1:1:end]

plotly()
Plots.scatter(obs_uncorr[1,1:100:end], obs_uncorr[2,1:100:end], obs_uncorr[3,1:100:end], markersize=1, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

normalization = false
σ_value = 0.08

averages, _, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.0002, do_print=true, conv_param=0.002, normalization=normalization)

if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
end

targets_norm = [norm(averages[:,i]) for i in eachindex(centers[1,:])]
Plots.scatter(centers[1,:], centers[2,:], marker_z=targets_norm, color=:viridis)

##
labels = [ssp.embedding(obs[:,i]) for i in 1:3:9000000]
averages_c, centers_c, Nc_c, labels_c = cleaning(averages, centers, labels)
Q = generator(labels_c;dt=3dt)
P_steady = steady_state(Q)

tsteps = 61
res = 50

auto_obs = zeros(dim, tsteps)
auto_Q = zeros(dim, tsteps)

for i in 1:dim
    auto_obs[i,:] = autocovariance(obs[i,1:res:10000000]; timesteps=tsteps) 
    auto_Q[i,:] = autocovariance(centers_c[i,:], Q, [0:dt*res:Int(res * (tsteps-1) * dt)...])
end

plt = Plots.plot(auto_obs[1,:])
plt = Plots.plot!(auto_Q[1,:])
##

gradLogp = zeros(dim, Nc_c)
for i in 1:Nc_c
    gradLogp[:,i] = - averages_c[:,i] / σ_value
end

C0 = centers_c * (centers_c * Diagonal(P_steady))'
C1_Q = centers_c * Q * (centers_c * Diagonal(P_steady))'
C1_grad = gradLogp * (centers_c * Diagonal(P_steady))'
Σ_test2 = C1_Q * inv(C1_grad)
Σ_test = cholesky(0.5*(Σ_test2 .+ Σ_test2')).L
println("Σ_test = ", Σ_test)

Σ_test2

##
#################### TRAINING WITH CLUSTERING LOSS ####################

@time nn_clustered, loss_clustered = train(inputs_targets, 2000, 32, [dim, 128, 64, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value
Plots.plot(loss_clustered)

##
#################### SAMPLES GENERATION ####################
score_clustered_xt(x,t) = Σ_test2 * score_clustered(x)
# score_true_xt(x,t) = Σ_test^2 * score_true(x, t)
sigma_Σ(x,t) = Matrix(Σ_test)

trj_clustered = evolve(zeros(dim), 0.2*dt, 100000, score_clustered_xt, sigma_Σ; timestepper=:euler, resolution=5, n_ens=100)
# trj_true = evolve([0.0], dt, Nsteps2, score_true_xt, sigma_Σ; timestepper=:euler, resolution=1)
# kde_true = kde(trj_true')

auto_clustered = zeros(dim, tsteps)
kde_clustered_x, kde_clustered_y = [], []
kde_obs_x, kde_obs_y = [], []
# auto_true = zeros(dim, tsteps)


for i in 1:dim
    kde_clustered_temp = kde(trj_clustered[i,:])
    push!(kde_clustered_x, [kde_clustered_temp.x...])
    push!(kde_clustered_y, kde_clustered_temp.density)
    kde_obs_temp = kde(obs[i,:])
    push!(kde_obs_x, [kde_obs_temp.x...])
    push!(kde_obs_y, kde_obs_temp.density)
    auto_clustered[i,:] = autocovariance(trj_clustered[i,1:res:end]; timesteps=tsteps) 
#    auto_true[i,:] = autocovariance(trj_true[i,1:res:end]; timesteps=tsteps)
end

Nsamples = 4000
samples_obs = (evolve(0.01*zeros(44), dt, Nsamples, F, sigma; timestepper=:rk4)[1:4,:] .- M) ./ S
samples_clustered = evolve(0.01*zeros(dim), 0.2*dt, 5*Nsamples, score_clustered_xt, sigma_Σ; timestepper=:rk4, resolution=5)

##
gr()
# Define a scientific color palette
science_colors = [:navy, :crimson, :darkgreen, :purple, :orange]
fontsize = 10
linewidth = 2

# Set global plotting theme for scientific publication
Plots.theme(:default)
default(fontfamily="Computer Modern", framestyle=:box, 
        tickfont=fontsize, guidefont=fontsize+2, legendfont=fontsize,
        titlefont=fontsize+4, margin=5Plots.mm, linewidth=linewidth,
        size=(1200, 1200), dpi=300)  # Increased height to accommodate three rows

# DIMENSION 1 PLOTS

# Plot 1: Time series comparison - Dimension 1
plt1 = Plots.plot(0:dt:Nsamples*dt, samples_obs[1,:], 
                 label="Observed", color=science_colors[1],
                 xlabel="t", ylabel="x1(t)", 
                 title="Time Series Comparison (Dim 1)",
                 legend=false)
Plots.plot!(plt1, 0:dt:Nsamples*dt, samples_clustered[1,:], 
           label="Model", color=science_colors[2], 
           linestyle=:solid)

# Plot 2: Probability density functions - Dimension 1
plt2 = Plots.plot(kde_obs_x[1], kde_obs_y[1], 
                 label="Observed", color=science_colors[1],
                 xlabel="x1", ylabel="Probability Density",
                 title="Probability Density Functions (Dim 1)",
                 xlims=(-4, 4), legend=false)
Plots.plot!(plt2, kde_clustered_x[1], kde_clustered_y[1], 
           label="Model", color=science_colors[2],
           linestyle=:solid)

# Plot 3: Autocorrelation functions - Dimension 1
plt3 = Plots.plot(auto_obs[1,:], 
                 label="Observed", color=science_colors[1],
                 xlabel="Lag × dt", ylabel="Autocorrelation",
                 title="Autocorrelation Functions (Dim 1)",
                 legend=false)
Plots.plot!(plt3, auto_clustered[1,:] ./ var(trj_clustered[1,:]), 
           label="Model", color=science_colors[2], 
           linestyle=:solid)

# DIMENSION 2 PLOTS

# Plot 4: Time series comparison - Dimension 2
plt4 = Plots.plot(0:dt:Nsamples*dt, samples_obs[2,:], 
                 label="Observed", color=science_colors[1],
                 xlabel="t", ylabel="x2(t)", 
                 title="Time Series Comparison (Dim 2)",
                 legend=false)
Plots.plot!(plt4, 0:dt:Nsamples*dt, samples_clustered[2,:], 
           label="Model", color=science_colors[2], 
           linestyle=:solid)

# Plot 5: Probability density functions - Dimension 2
plt5 = Plots.plot(kde_obs_x[2], kde_obs_y[2], 
                 label="Observed", color=science_colors[1],
                 xlabel="x2", ylabel="Probability Density",
                 title="Probability Density Functions (Dim 2)",
                 xlims=(-4, 4), legend=false)
Plots.plot!(plt5, kde_clustered_x[2], kde_clustered_y[2], 
           label="Model", color=science_colors[2],
           linestyle=:solid)

# Plot 6: Autocorrelation functions - Dimension 2
plt6 = Plots.plot(auto_obs[2,:], 
                 label="Observed", color=science_colors[1],
                 xlabel="Lag × dt", ylabel="Autocorrelation",
                 title="Autocorrelation Functions (Dim 2)",
                 legend=false)
Plots.plot!(plt6, auto_clustered[2,:] ./ var(trj_clustered[2,:]), 
           label="Model", color=science_colors[2], 
           linestyle=:solid)

# NEW PLOTS FOR DIMENSION 3

# Plot 7: Time series comparison - Dimension 3
plt7 = Plots.plot(0:dt:Nsamples*dt, samples_obs[3,:], 
                 label="Observed", color=science_colors[1],
                 xlabel="t", ylabel="x3(t)", 
                 title="Time Series Comparison (Dim 3)",
                 legend=false)
Plots.plot!(plt7, 0:dt:Nsamples*dt, samples_clustered[3,:], 
           label="Model", color=science_colors[2], 
           linestyle=:solid)

# Plot 8: Probability density functions - Dimension 3
plt8 = Plots.plot(kde_obs_x[3], kde_obs_y[3], 
                 label="Observed", color=science_colors[1],
                 xlabel="x3", ylabel="Probability Density",
                 title="Probability Density Functions (Dim 3)",
                 xlims=(-4, 4), legend=false)
Plots.plot!(plt8, kde_clustered_x[3], kde_clustered_y[3], 
           label="Model", color=science_colors[2],
           linestyle=:solid)

# Plot 9: Autocorrelation functions - Dimension 3 (with legend)
plt9 = Plots.plot(auto_obs[3,:], 
                 label="Observed", color=science_colors[1],
                 xlabel="Lag × dt", ylabel="Autocorrelation",
                 title="Autocorrelation Functions (Dim 3)",
                 legend=:topright)  # Only show legend on the last plot
Plots.plot!(plt9, auto_clustered[3,:] ./ var(trj_clustered[3,:]), 
           label="Model", color=science_colors[2], 
           linestyle=:solid)

# Combine plots in a 3×3 grid with increased left margin
figure = Plots.plot(
    plt1, plt4, plt7,  # Time series plots in first column
    plt2, plt5, plt8,  # PDF plots in second column
    plt3, plt6, plt9,  # Autocorrelation plots in third column
    layout=(3, 3),     # 3 rows, 3 columns
    size=(1800, 1200), # Wider to accommodate 3 columns
    grid=true,
    plot_title="Lorenz63 - Data from Files",
    left_margin=10Plots.mm  # Increase left margin to prevent y-label cutoff
)

# Save the figure in publication quality
# savefig(figure, "figures/Sigma_figures/lorenz96.png")

display(figure)