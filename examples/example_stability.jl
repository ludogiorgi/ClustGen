using Pkg
Pkg.activate(".")
Pkg.instantiate()
##
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
using QuadGK
using BSON
using Plots
using GLMakie
using StatsBase
using MarkovChainHammer
import MarkovChainHammer.Trajectory: ContinuousTimeEmpiricalProcess
import LaTeXStrings


function ∇U(x; A1=1.0, A2=1.0, B1=0.6, B2=0.3, C=1.0, D=0.0)
    # Conservative gradient terms
    ∇U1 = 2 * (x[1] + A1) * (x[1] - A1)^2 + 2 * (x[1] - A1) * (x[1] + A1)^2 + B1 + C * (x[1] * x[2])^2
    ∇U2 = 2 * (x[2] + A2) * (x[2] - A2)^2 + 2 * (x[2] - A2) * (x[2] + A2)^2 + B2 + C * (x[1] * x[2])^2
    
    # Non-conservative term (e.g., rotational flow)
    F1 = -D * x[2]
    F2 = D * x[1]
    
    # Total force
    return [∇U1 + F1, ∇U2 + F2]
end

function potential_data(x0, timesteps, dt, Σ, ∇U; res = 1)
    dim = length(x0)
    force(x) = -∇U(x)
    x = []
    x_temp = x0
    for i in ProgressBar(2:timesteps)
        Σ2_force(x) = Σ^2 * force(x)
        rk4_step!(x_temp, dt, Σ2_force)
        @inbounds x_temp .+= Σ * randn(dim) * sqrt(2dt)
        if i % res == 0
            push!(x, copy(x_temp))
        end
    end
    x = hcat(x...)
    return x
end

dt = 0.025
Σ_true = 0.5 * [1.0 0.5; 0.5 1.0]
obs = potential_data([0.0,0.0], 10000000, dt, Σ_true, ∇U)
dim = size(obs_c)[1]

normalization = false
σ_value = 0.05

μ = repeat(obs_c[:,1:100:end], 1, 1)

averages, centers, Nc, labels = f_tilde_labels(σ_value, μ; prob=0.001, do_print=true, conv_param=0.001, normalization=normalization)

Q = generator(labels)
P_steady = steady_state(Q)

Q_c, Q_nc = decomposition(Q)

gradLogp = zeros(dim, Nc)
for i in 1:Nc
    gradLogp[:,i] = - averages[:,i] / σ_value
end

Σ_test = computeSigma(centers', P_steady, Q_c, gradLogp')

inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=normalization)
inputs, targets = inputs_targets

nn_clustered, loss_clustered = train(inputs_targets, 1000, 128, [dim, 128, 64, dim]; use_gpu=false, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value
cluster_loss = check_loss(μ, nn_clustered_cpu, σ_value)

trj_clustered = sample_langevin_Σ(100000, dt, score_clustered, randn(dim), sqrt(Σ_test); seed=123, res = 1)

kde_gen1 = kde(trj_clustered')
Plots.heatmap(kde_gen1.x, kde_gen1.y, kde_gen1.density, aspect_ratio=:equal, color=:viridis, xlabel="x", ylabel="y", title="Generated data")
##

diffs = diff(μ, dims=2) ./ dt

function calculate_f(X, z)
    Ndim, Nz = size(z)
    Nc = maximum(X)
    f = zeros(Ndim, Nc)
    f_temp = zeros(Ndim, Nc)
    count = zeros(Ndim, Nc)
    for i in 1:Nz
        segment_index = X[i]
        for dim in 1:Ndim
            f_temp[dim, segment_index] += z[dim, i]
            count[dim, segment_index] += 1
        end
    end
    for dim in 1:Ndim
        for i in 1:Nc
            if count[dim, i] != 0
                f[dim, i] = f_temp[dim, i] / count[dim, i]
            end
        end
    end
    return f
end

_, centers_f, Nc_f, labels_f = f_tilde_labels(σ_value, obs[:,1:1000000]; prob=0.001, do_print=true, conv_param=0.001, normalization=normalization)
f = calculate_f(labels, diffs)
##

inputs_targets = generate_inputs_targets(f, centers, Nc; normalization=normalization)
inputs, targets = inputs_targets

nn_clustered_f, loss_clustered_f = train(inputs_targets, 1000, 128, [dim, 128, 64, dim]; use_gpu=false, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_f_cpu  = Chain(nn_clustered_f, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_f_cpu = nn_clustered_f |> cpu
end
score_clustered_f(x) = nn_clustered_f_cpu(Float32.([x...]))[:]

trj_clustered_f = sample_langevin_Σ(100000, dt, score_clustered_f, randn(dim), sqrt(Σ_true); seed=123, res = 1)

kde_gen2 = kde(trj_clustered_f')
Plots.heatmap(kde_gen2.x, kde_gen2.y, kde_gen2.density, aspect_ratio=:equal, color=:viridis, xlabel="x", ylabel="y", title="Generated data")

##

function plot_vector_field!(ax, f; range_x=(-2, 2), range_y=(-2, 2), N=20)
    xs = range(range_x[1], range_x[2], length=N)
    ys = range(range_y[1], range_y[2], length=N)
    Ux = zeros(Float32, N, N)
    Uy = zeros(Float32, N, N)

    for i in 1:N
        for j in 1:N
            vec_field = f([xs[i], ys[j]])
            Ux[i, j] = vec_field[1]
            Uy[i, j] = vec_field[2]
        end
    end

    # Calculate vector magnitudes for coloring
    strength = vec(sqrt.(Ux.^2 .+ Uy.^2))
    
    # Use arrows! instead of quiver!
    arrows!(ax, xs, ys, Ux, Uy, 
           arrowsize = 10, 
           lengthscale = 0.3,
           arrowcolor = strength, 
           linecolor = strength,
           normalize=true,
           colormap=:viridis)
end

resolution=(1000, 1000)
set_theme!(Theme(fontsize=18, backgroundcolor=:white, colormap=:viridis))
fig = Figure(resolution=resolution)
ax1 = Axis(fig[1, 1], xlabel="x", ylabel="y", title="True vector field")
plot_vector_field!(ax1, score_clustered; range_x=(-2, 2), range_y=(-2, 2), N=20)

ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", title="Estimated vector field")
plot_vector_field!(ax2, score_clustered_f; range_x=(-2, 2), range_y=(-2, 2), N=20)

fig
##
plotly()
Plots.scatter(centers[1,:], centers[2,:], f[1,:] ./ std(f[1,:]))

Plots.scatter!(centers[1,:], centers[2,:], .- averages[1,:] ./ std(averages[1,:]))
