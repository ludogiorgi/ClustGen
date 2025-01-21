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
using Plots
using StatsBase
using MarkovChainHammer
import MarkovChainHammer.Trajectory: ContinuousTimeEmpiricalProcess
import LaTeXStrings

function cleaning(averages, centers, labels)
    unique_clusters = sort(unique(labels))
    mapping = Dict(old_cluster => new_cluster for (new_cluster, old_cluster) in enumerate(unique_clusters))
    labels_new = [mapping[cluster] for cluster in labels]
    averages_new = averages[:, unique_clusters]
    centers_new = centers[:, unique_clusters]
    return averages_new, centers_new, length(unique_clusters), labels_new
end

##
T = 10000000.0
dt = 0.01
dim = 4
F = 6.0
u0 = randn(dim)
noise = 0.1

u_lorenz96 = simulate_lorenz96(T, dt, F, u0, noise; res=1)
obs = (u_lorenz96 .- mean(u_lorenz96, dims=2)) ./ std(u_lorenz96, dims=2)

##
dim = size(obs)[1]

normalization = false
σ_value = 0.05

μ = repeat(obs[:,1:100:end], 1, 1)

averages, centers, Nc, labels = f_tilde_labels(σ_value, μ; prob=0.0005, do_print=true, conv_param=0.001, normalization=normalization)

averages, centers, Nc, labels = cleaning(averages, centers, labels)

gradLogp = zeros(dim, Nc)
for i in 1:Nc
    gradLogp[:,i] = - averages[:,i] / σ_value
end

Q = generator(labels)
P_steady = steady_state(Q)
Q_c, Q_nc = decomposition(Q)

# C0 = centers * (centers * Diagonal(P_steady))'
# C1_Q = centers * Q_c * (centers * Diagonal(P_steady))'
# C1_grad = gradLogp * (centers * Diagonal(P_steady))'
# Σ_test = sqrt(C1_Q * inv(C1_grad))

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
cluster_loss = check_loss(obs[:,1:100:end], nn_clustered_cpu, σ_value)

trj_clustered = sample_langevin_Σ(10000, 0.2*dt, score_clustered, randn(dim), Σ_test; seed=123, res = 5)

res = 10
tsteps = 51

auto_Q = zeros(dim, tsteps)
auto_gen = zeros(dim, tsteps)

for i in 1:dim
    auto_Q[i,:] = autocovariance(centers[i,:], Q_c, [0:dt*res:Int(res * (tsteps-1) * dt)...]) ./ std(centers[i,:])^2
    auto_gen[i,:] = autocovariance(trj_clustered[i,1:res:end]; timesteps=tsteps) ./ std(trj_clustered[i,:])^2
end

resolution=(1000, 1500)
set_theme!(Theme(fontsize=18, backgroundcolor=:white, colormap=:viridis))
f = Figure(resolution=resolution)
len_corr = length(auto_Q[1,:])

kde_obs_12 = kde(obs[[1,2],:]')
kde_clustered_12 = kde(trj_clustered[[1,2],:]') 
kde_obs_13 = kde(obs[[1,3],:]')
kde_clustered_13 = kde(trj_clustered[[1,3],:]')
kde_obs_14 = kde(obs[[1,4],:]')
kde_clustered_14 = kde(trj_clustered[[1,4],:]')

    
# Get colors from viridis palette
color1 = cgrad(:viridis)[0.6]  # First color
color2 = cgrad(:viridis)[0.2]  # Second color
    
# C₁₁
ax1 = Axis(f[1, 1], xlabel="Time", ylabel="Correlation",
            title=L"\textbf{C_{11}}", xticksize=10, yticksize=10, titlesize=20)
lines!(ax1, 0:res*dt:res*dt*(len_corr-1), auto_Q[1,:], label="From Data", linewidth=2, color=color1)
lines!(ax1, 0:res*dt:res*dt*(len_corr-1), auto_gen[1,:], label="From Score", linewidth=2, color=color2)
axislegend(ax1, position=:rt, framevisible=false)

# C₁₂
ax2 = Axis(f[1, 2], xlabel="Time", ylabel="Correlation",
            title=L"\textbf{C_{22}}", xticksize=10, yticksize=10, titlesize=20)
lines!(ax2, 0:res*dt:res*dt*(len_corr-1), auto_Q[2,:], label="From Data", linewidth=2, color=color1)
lines!(ax2, 0:res*dt:res*dt*(len_corr-1), auto_gen[2,:], label="From Score", linewidth=2, color=color2)

# C₂₁
ax3 = Axis(f[2, 1], xlabel="Time", ylabel="Correlation",
            title=L"\textbf{C_{33}}", xticksize=10, yticksize=10, titlesize=20)
lines!(ax3, 0:res*dt:res*dt*(len_corr-1), auto_Q[3,:], label="From Data", linewidth=2, color=color1)
lines!(ax3, 0:res*dt:res*dt*(len_corr-1), auto_gen[3,:], label="From Score", linewidth=2, color=color2)

# C₂₂
ax4 = Axis(f[2, 2], xlabel="Time", ylabel="Correlation",
            title=L"\textbf{C_{44}}", xticksize=10, yticksize=10, titlesize=20)
lines!(ax4, 0:res*dt:res*dt*(len_corr-1), auto_Q[4,:], label="From Data", linewidth=2, color=color1)
lines!(ax4, 0:res*dt:res*dt*(len_corr-1), auto_gen[4,:], label="From Score", linewidth=2, color=color2)

# PDF from Data
ax5 = Axis(f[3, 1], title="PDF from Data", xlabel="x₁", ylabel="x₂")
xlims!(ax5, -2, 2)
ylims!(ax5, -2, 2)

# Then call heatmap! without xlims/ylims:
hm1 = GLMakie.heatmap!(
    ax5,
    kde_obs_12.x,
    kde_obs_12.y,
    kde_obs_12.density,
    colorrange=(0, maximum(kde_clustered_12.density))
)

# PDF from Score
ax6 = Axis(f[3, 2], title="PDF from Score", xlabel="x₁", ylabel="x₂")
xlims!(ax6, -2, 2)
ylims!(ax6, -2, 2)
hm2 = GLMakie.heatmap!(ax6, kde_clustered_12.x, kde_clustered_12.y, kde_clustered_12.density, 
                   colorrange=(0, maximum(kde_clustered_12.density)))

ax7 = Axis(f[4, 1], title="PDF from Data", xlabel="x₁", ylabel="x₃")
xlims!(ax7, -2, 2)
ylims!(ax7, -2, 2)
hm3 = GLMakie.heatmap!(ax7, kde_obs_13.x, kde_obs_13.y, kde_obs_13.density, 
                   colorrange=(0, maximum(kde_clustered_12.density)))
                   
ax8 = Axis(f[4, 2], title="PDF from Score", xlabel="x₁", ylabel="x₃")
xlims!(ax8, -2, 2)
ylims!(ax8, -2, 2)
hm4 = GLMakie.heatmap!(ax8, kde_clustered_13.x, kde_clustered_13.y, kde_clustered_13.density, 
                   colorrange=(0, maximum(kde_clustered_12.density)))

ax9 = Axis(f[5, 1], title="PDF from Data", xlabel="x₁", ylabel="x₄")
xlims!(ax9, -2, 2)
ylims!(ax9, -2, 2)
hm5 = GLMakie.heatmap!(ax9, kde_obs_14.x, kde_obs_14.y, kde_obs_14.density, 
                   colorrange=(0, maximum(kde_clustered_12.density)))

ax10 = Axis(f[5, 2], title="PDF from Score", xlabel="x₁", ylabel="x₄")
xlims!(ax10, -2, 2)
ylims!(ax10, -2, 2)
hm6 = GLMakie.heatmap!(ax10, kde_clustered_14.x, kde_clustered_14.y, kde_clustered_14.density, 
                   colorrange=(0, maximum(kde_clustered_12.density)))

        
# Add colorbar for PDFs
Colorbar(f[3:5, 3], hm1, label="Probability density")

# save("figures/fig_pub.png", f, resolution=resolution)
                                     
f