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
T = 50000000.0
dt = 0.01
dim = 8
F = 6.0
u0 = randn(dim)
noise = 0.2

u_lorenz96 = simulate_lorenz96(T, dt, F, u0, noise; res=1)
obs = (u_lorenz96 .- mean(u_lorenz96, dims=2)) ./ std(u_lorenz96, dims=2)

##
dim = size(obs)[1]

normalization = false
σ_value = 0.2

μ = repeat(obs[:,1:100:end], 1, 1)

averages, centers, Nc, labels = f_tilde_labels(σ_value, μ; prob=0.00025, do_print=true, conv_param=0.001, normalization=normalization)

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
##
########### Additional training on the full data ############

@time nn_clustered_double, loss_clustered_double = train(μ, 0, 128, nn_clustered, σ_value; use_gpu=true)
nn_clustered_double_cpu = nn_clustered_double |> cpu
score_clustered_double(x) = .- nn_clustered_double_cpu(Float32.([x...]))[:] ./ σ_value
Plots.plot(loss_clustered_double)
hline!([cluster_loss])
##
@time nn_vanilla, loss_vanilla = train(μ, 100, 128, [dim, 128, 64, dim], σ_value; use_gpu=false, opt=Adam(0.001))
nn_vanilla_cpu = nn_vanilla |> cpu
score_vanilla(x) = .- nn_vanilla_cpu(Float32.([x...])) ./ σ_value
Plots.plot(loss_vanilla)
hline!([cluster_loss])
##
trj_clustered = sample_langevin_Σ(10000, 0.2*dt, score_clustered, randn(dim), Σ_test; seed=123, res = 5)
##
res = 10
tsteps = 51

auto_Q = zeros(dim, tsteps)
auto_gen = zeros(dim, tsteps)

for i in 1:dim
    auto_Q[i,:] = autocovariance(centers[i,:], Q_c, [0:dt*res:Int(res * (tsteps-1) * dt)...]) ./ std(centers[i,:])^2
    auto_gen[i,:] = autocovariance(trj_clustered[i,1:res:end]; timesteps=tsteps) ./ std(trj_clustered[i,:])^2
end

len_corr = length(auto_Q[1,:])

kde_obs_12 = kde(obs[[1,2],:]')
kde_clustered_13 = kde(trj_clustered[[1,2],:]') 
kde_obs_13 = kde(obs[[1,3],:]')
kde_clustered_13 = kde(trj_clustered[[1,3],:]')
kde_obs_14 = kde(obs[[1,4],:]')
kde_clustered_14 = kde(trj_clustered[[1,4],:]')
kde_obs_15 = kde(obs[[1,5],:]')
kde_clustered_15 = kde(trj_clustered[[1,5],:]')
kde_obs_16 = kde(obs[[1,6],:]')
kde_clustered_16 = kde(trj_clustered[[1,6],:]')
kde_obs_17 = kde(obs[[1,7],:]')
kde_clustered_17 = kde(trj_clustered[[1,7],:]')
##
resolution=(1500, 1500)
set_theme!(Theme(fontsize=18, backgroundcolor=:white, colormap=:viridis))
f = Figure(resolution=resolution)

# Get colors from viridis palette
color1 = cgrad(:viridis)[0.6]  # First color
color2 = cgrad(:viridis)[0.2]  # Second color
    
# C11
ax1 = Axis(f[1, 1], xlabel="Time", ylabel="Correlation",
            title=L"\textbf{C_{11}}", xticksize=10, yticksize=10, titlesize=20)
lines!(ax1, 0:res*dt:res*dt*(len_corr-1), auto_Q[1,:], label="From Data", linewidth=2, color=color1)
lines!(ax1, 0:res*dt:res*dt*(len_corr-1), auto_gen[1,:], label="From Score", linewidth=2, color=color2)
axislegend(ax1, position=:rt, framevisible=false)

# C22
ax2 = Axis(f[1, 2], xlabel="Time", ylabel="Correlation",
            title=L"\textbf{C_{22}}", xticksize=10, yticksize=10, titlesize=20)
lines!(ax2, 0:res*dt:res*dt*(len_corr-1), auto_Q[2,:], label="From Data", linewidth=2, color=color1)
lines!(ax2, 0:res*dt:res*dt*(len_corr-1), auto_gen[2,:], label="From Score", linewidth=2, color=color2)

# C33
ax3 = Axis(f[1, 3], xlabel="Time", ylabel="Correlation",
            title=L"\textbf{C_{33}}", xticksize=10, yticksize=10, titlesize=20)
lines!(ax3, 0:res*dt:res*dt*(len_corr-1), auto_Q[3,:], label="From Data", linewidth=2, color=color1)
lines!(ax3, 0:res*dt:res*dt*(len_corr-1), auto_gen[3,:], label="From Score", linewidth=2, color=color2)

# C44
ax4 = Axis(f[1, 4], xlabel="Time", ylabel="Correlation",
            title=L"\textbf{C_{44}}", xticksize=10, yticksize=10, titlesize=20)
lines!(ax4, 0:res*dt:res*dt*(len_corr-1), auto_Q[4,:], label="From Data", linewidth=2, color=color1)
lines!(ax4, 0:res*dt:res*dt*(len_corr-1), auto_gen[4,:], label="From Score", linewidth=2, color=color2)

# C55
ax5 = Axis(f[2, 1], xlabel="Time", ylabel="Correlation",
            title=L"\textbf{C_{55}}", xticksize=10, yticksize=10, titlesize=20)
lines!(ax5, 0:res*dt:res*dt*(len_corr-1), auto_Q[5,:], label="From Data", linewidth=2, color=color1)
lines!(ax5, 0:res*dt:res*dt*(len_corr-1), auto_gen[5,:], label="From Score", linewidth=2, color=color2)

# C66
ax6 = Axis(f[2, 2], xlabel="Time", ylabel="Correlation",
            title=L"\textbf{C_{66}}", xticksize=10, yticksize=10, titlesize=20)
lines!(ax6, 0:res*dt:res*dt*(len_corr-1), auto_Q[6,:], label="From Data", linewidth=2, color=color1)
lines!(ax6, 0:res*dt:res*dt*(len_corr-1), auto_gen[6,:], label="From Score", linewidth=2, color=color2)

# C77
ax7 = Axis(f[2, 3], xlabel="Time", ylabel="Correlation",
            title=L"\textbf{C_{77}}", xticksize=10, yticksize=10, titlesize=20)
lines!(ax7, 0:res*dt:res*dt*(len_corr-1), auto_Q[7,:], label="From Data", linewidth=2, color=color1)
lines!(ax7, 0:res*dt:res*dt*(len_corr-1), auto_gen[7,:], label="From Score", linewidth=2, color=color2)

# C88
ax8 = Axis(f[2, 4], xlabel="Time", ylabel="Correlation",
            title=L"\textbf{C_{88}}", xticksize=10, yticksize=10, titlesize=20)
lines!(ax8, 0:res*dt:res*dt*(len_corr-1), auto_Q[8,:], label="From Data", linewidth=2, color=color1)
lines!(ax8, 0:res*dt:res*dt*(len_corr-1), auto_gen[8,:], label="From Score", linewidth=2, color=color2)

Mc = 0.18

# PDF from Data
ax9 = Axis(f[3, 1], title="PDF from Data", xlabel="x₁", ylabel="x₂")
xlims!(ax9, -3, 3)
ylims!(ax9, -3, 3)

# Then call heatmap! without xlims/ylims:
hm1 = GLMakie.heatmap!(
    ax9,
    kde_obs_12.x,
    kde_obs_12.y,
    kde_obs_12.density,
    colorrange=(0, Mc)
)

# PDF from Score
ax10 = Axis(f[3, 2], title="PDF from Score", xlabel="x₁", ylabel="x₂")
xlims!(ax10, -3, 3)
ylims!(ax10, -3, 3)
hm2 = GLMakie.heatmap!(ax10, kde_clustered_12.x, kde_clustered_12.y, kde_clustered_12.density, 
                   colorrange=(0, Mc))

ax11 = Axis(f[4, 1], title="PDF from Data", xlabel="x₁", ylabel="x₃")
xlims!(ax11, -3, 3)
ylims!(ax11, -3, 3)
hm3 = GLMakie.heatmap!(ax11, kde_obs_13.x, kde_obs_13.y, kde_obs_13.density, 
                   colorrange=(0, Mc))
                   
ax12 = Axis(f[4, 2], title="PDF from Score", xlabel="x₁", ylabel="x₃")
xlims!(ax12, -3, 3)
ylims!(ax12, -3, 3)
hm4 = GLMakie.heatmap!(ax12, kde_clustered_13.x, kde_clustered_13.y, kde_clustered_13.density, 
                   colorrange=(0, Mc))

ax13 = Axis(f[5, 1], title="PDF from Data", xlabel="x₁", ylabel="x₄")
xlims!(ax13, -3, 3)
ylims!(ax13, -3, 3)
hm5 = GLMakie.heatmap!(ax13, kde_obs_14.x, kde_obs_14.y, kde_obs_14.density, 
                   colorrange=(0, Mc))

ax14 = Axis(f[5, 2], title="PDF from Score", xlabel="x₁", ylabel="x₄")
xlims!(ax14, -3, 3)
ylims!(ax14, -3, 3)

hm6 = GLMakie.heatmap!(ax14, kde_clustered_14.x, kde_clustered_14.y, kde_clustered_14.density, 
                   colorrange=(0, Mc))
                

# PDF from Data
ax15 = Axis(f[3, 3], title="PDF from Data", xlabel="x₁", ylabel="x₅")
xlims!(ax15, -3, 3)
ylims!(ax15, -3, 3)

# Then call heatmap! without xlims/ylims:
hm7 = GLMakie.heatmap!(
    ax15,
    kde_obs_15.x,
    kde_obs_15.y,
    kde_obs_15.density,
    colorrange=(0, Mc)
)

# PDF from Score
ax16 = Axis(f[3, 4], title="PDF from Score", xlabel="x₁", ylabel="x₅")
xlims!(ax16, -3, 3)
ylims!(ax16, -3, 3)
hm8 = GLMakie.heatmap!(ax16, kde_clustered_15.x, kde_clustered_15.y, kde_clustered_15.density, 
                   colorrange=(0, Mc))

ax17 = Axis(f[4, 3], title="PDF from Data", xlabel="x₁", ylabel="x₆")
xlims!(ax17, -3, 3)
ylims!(ax17, -3, 3)
hm9 = GLMakie.heatmap!(ax17, kde_obs_16.x, kde_obs_16.y, kde_obs_16.density, 
                   colorrange=(0, Mc))
                   
ax18 = Axis(f[4, 4], title="PDF from Score", xlabel="x₁", ylabel="x₆")
xlims!(ax18, -3, 3)
ylims!(ax18, -3, 3)
hm10 = GLMakie.heatmap!(ax18, kde_clustered_16.x, kde_clustered_16.y, kde_clustered_16.density, 
                   colorrange=(0, Mc))

ax19 = Axis(f[5, 3], title="PDF from Data", xlabel="x₁", ylabel="x₇")
xlims!(ax19, -3, 3)
ylims!(ax19, -3, 3)
hm11 = GLMakie.heatmap!(ax19, kde_obs_17.x, kde_obs_17.y, kde_obs_17.density, 
                   colorrange=(0, Mc))

ax20 = Axis(f[5, 4], title="PDF from Score", xlabel="x₁", ylabel="x₇")
xlims!(ax20, -3, 3)
ylims!(ax20, -3, 3)
hm12 = GLMakie.heatmap!(ax20, kde_clustered_17.x, kde_clustered_17.y, kde_clustered_17.density, 
                   colorrange=(0, Mc))

        
# Add colorbar for PDFs
Colorbar(f[3:5, 5], hm1, label="Probability density")

save("figures/fig_pub_lorenz96_8D.png", f, resolution=resolution)
                                     
f