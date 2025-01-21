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
using GLMakie
using GLMakie: xlims!, ylims!
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

function potential_data(x0, timesteps, dt, Σ; res = 1)
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
Σ_true = [1.0 0.5; 0.5 1.0]
obs = potential_data(randn(2), 10000000, dt, Σ_true)
dim = size(obs)[1]

normalization = false
σ_value = 0.05

μ = repeat(obs[:,1:100:end], 1, 1)

averages, centers, Nc, labels = f_tilde_labels(σ_value, μ; prob=0.001, do_print=true, conv_param=0.001, normalization=normalization)

Q = generator(labels)
P_steady = steady_state(Q)

Q_c, Q_nc = decomposition(Q)

inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=normalization)
inputs, targets = inputs_targets

nn_clustered, loss_clustered = train(inputs_targets, 1000, 128, [dim, 128, 64, dim]; use_gpu=false, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value
cluster_loss = check_loss(obs, nn_clustered_cpu, σ_value)

gradLogp = zeros(dim, Nc)
for i in 1:Nc
    gradLogp[:,i] = - averages[:,i] / σ_value
end

Σ_test = computeSigma(centers', P_steady, Q_c, gradLogp')

trj_clustered = sample_langevin_Σ(100000, dt, score_clustered, randn(dim), sqrt(Σ_test); seed=123, res = 1)

res = 10
tsteps = 81

auto_Q = zeros(dim, tsteps)
auto_gen = zeros(dim, tsteps)

for i in 1:dim
    auto_Q[i,:] = autocovariance(centers[i,:], Q_c, [0:dt*res:Int(res * (tsteps-1) * dt)...]) ./ std(centers[i,:])^2
    auto_gen[i,:] = autocovariance(trj_clustered[i,1:res:end]; timesteps=tsteps) ./ std(trj_clustered[i,:])^2
end

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
f = Figure(resolution=resolution)
len_corr = length(auto_Q[1,:])

kde_obs_12 = kde(obs[[1,2],:]')
kde_clustered_12 = kde(trj_clustered[[1,2],:]') 
    
# Get colors from viridis palette
color1 = cgrad(:viridis)[0.6]  # First color
color2 = cgrad(:viridis)[0.2]  # Second color
    

ax1 = Axis(f[1, 1], xlabel="Time", ylabel="Correlation",
            title=L"\textbf{C_{11}}", xticksize=10, yticksize=10, titlesize=20)
lines!(ax1, 0:res*dt:res*dt*(len_corr-1), auto_Q[1,:], label="From Data", linewidth=2, color=color1)
lines!(ax1, 0:res*dt:res*dt*(len_corr-1), auto_gen[1,:], label="From Score", linewidth=2, color=color2)
axislegend(ax1, position=:rt, framevisible=false)

ax2 = Axis(f[1, 2], xlabel="Time", ylabel="Correlation",
            title=L"\textbf{C_{22}}", xticksize=10, yticksize=10, titlesize=20)
lines!(ax2, 0:res*dt:res*dt*(len_corr-1), auto_Q[2,:], label="From Data", linewidth=2, color=color1)
lines!(ax2, 0:res*dt:res*dt*(len_corr-1), auto_gen[2,:], label="From Score", linewidth=2, color=color2)

# PDF from Data
ax3 = Axis(f[2, 1], title="PDF from Data", xlabel="x₁", ylabel="x₂")
xlims!(ax3, -2, 2)
ylims!(ax3, -2, 2)

# Then call heatmap! without xlims/ylims:
hm1 = GLMakie.heatmap!(
    ax3,
    kde_obs_12.x,
    kde_obs_12.y,
    kde_obs_12.density,
    colorrange=(0, maximum(kde_clustered_12.density))
)

# PDF from Score
ax4 = Axis(f[2, 2], title="PDF from Score", xlabel="x₁", ylabel="x₂")
xlims!(ax4, -2, 2)
ylims!(ax4, -2, 2)
hm2 = GLMakie.heatmap!(ax4, kde_clustered_12.x, kde_clustered_12.y, kde_clustered_12.density, 
                   colorrange=(0, maximum(kde_clustered_12.density)))

        
# Add colorbar for PDFs
Colorbar(f[2, 3], hm1, label="Probability density")

# save("figures/fig_pub_potential.png", f, resolution=resolution)
                                     
f


