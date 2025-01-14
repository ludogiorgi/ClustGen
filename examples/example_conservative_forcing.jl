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
using StatsBase
using MarkovChainHammer
import MarkovChainHammer.Trajectory: ContinuousTimeEmpiricalProcess
import LaTeXStrings

##
function ∇U(x; A1=1.0, A2=1.0, B1=0.6, B2=0.3, C=1.0, D=1.0)
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

trj_clustered = sample_langevin_Σ(100000, dt, score_clustered, randn(2), Σ_true; seed=123, res = 1)

dec_times_M = 400
_, corr_gen = ClustGen.decorrelation_times(trj_clustered[:,1:100000], dec_times_M)

corr_Qc_11 = [compute_corr(centers[1,:], centers[1,:], P_steady, Q_c, dt*i) for i in 0:dec_times_M-1]
corr_Qc_12 = [compute_corr(centers[1,:], centers[2,:], P_steady, Q_c, dt*i) for i in 0:dec_times_M-1]
corr_Qc_21 = [compute_corr(centers[2,:], centers[1,:], P_steady, Q_c, dt*i) for i in 0:dec_times_M-1]
corr_Qc_22 = [compute_corr(centers[2,:], centers[2,:], P_steady, Q_c, dt*i) for i in 0:dec_times_M-1]

##

# Example function to style correlation plots
function plot_correlations_pub(corr_Qc_11, corr_Qc_12, corr_Qc_21, corr_Qc_22, corr_gen; resolution=(1200, 800))
    set_theme!(Theme(fontsize=18, backgroundcolor=:white, colormap=:viridis))
    f = Figure(resolution=resolution)
    len_corr = length(corr_Qc_11)
    
    # Get colors from viridis palette
    color1 = cgrad(:viridis)[0.6]  # First color
    color2 = cgrad(:viridis)[0.2]  # Second color
    
    # C₁₁
    ax1 = Axis(f[1, 1], xlabel="Time", ylabel="Correlation",
               title=L"\textbf{C_{11}}", xticksize=10, yticksize=10, titlesize=20)
    lines!(ax1, dt:dt:dt*len_corr, corr_Qc_11, label="From Data", linewidth=2, color=color1)
    lines!(ax1, dt:dt:dt*len_corr, corr_gen[1, 1, :], label="From Score", linewidth=2, color=color2)
    axislegend(ax1, position=:rt, framevisible=false)

    # C₁₂
    ax2 = Axis(f[1, 2], xlabel="Time", ylabel="Correlation",
               title=L"\textbf{C_{12}}", xticksize=10, yticksize=10, titlesize=20)
    lines!(ax2, dt:dt:dt*len_corr, corr_Qc_12, label="From Data", linewidth=2, color=color1)
    lines!(ax2, dt:dt:dt*len_corr, corr_gen[1, 2, :], label="From Score", linewidth=2, color=color2)

    # C₂₁
    ax3 = Axis(f[2, 1], xlabel="Time", ylabel="Correlation",
               title=L"\textbf{C_{21}}", xticksize=10, yticksize=10, titlesize=20)
    lines!(ax3, dt:dt:dt*len_corr, corr_Qc_21, label="From Data", linewidth=2, color=color1)
    lines!(ax3, dt:dt:dt*len_corr, corr_gen[2, 1, :], label="From Score", linewidth=2, color=color2)

    # C₂₂
    ax4 = Axis(f[2, 2], xlabel="Time", ylabel="Correlation",
               title=L"\textbf{C_{22}}", xticksize=10, yticksize=10, titlesize=20)
    lines!(ax4, dt:dt:dt*len_corr, corr_Qc_22, label="From Data", linewidth=2, color=color1)
    lines!(ax4, dt:dt:dt*len_corr, corr_gen[2, 2, :], label="From Score", linewidth=2, color=color2)

    display(f)
    save("figures/fig1_pub.png", f, resolution=resolution)
    return f
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


function plot_pdfs_and_vectorfields_pub(kde_obs, kde_clustered, vecfun; resolution=(1200, 800))
    set_theme!(Theme(fontsize=18, backgroundcolor=:white, colormap=:viridis))
    f2 = Figure(resolution=resolution)

    # PDF from Data
    ax1 = Axis(f2[1, 1], title="PDF from Data", xlabel="x", ylabel="y")
    hm1 = heatmap!(ax1, kde_obs.x, kde_obs.y, kde_obs.density, 
                   colorrange=(0, maximum(kde_obs.density)))

    # PDF from Score
    ax2 = Axis(f2[2, 1], title="PDF from Score", xlabel="x", ylabel="y")
    hm2 = heatmap!(ax2, kde_clustered.x, kde_clustered.y, kde_clustered.density, 
                   colorrange=(0, maximum(kde_clustered.density)))

    # Add colorbar for PDFs
    Colorbar(f2[1:2, 2], hm1, label="Probability density")

    # True Vector Field
    ax3 = Axis(f2[1, 3], title="True Vector Field", xlabel="x", ylabel="y")
    vf1 = plot_vector_field!(ax3, vecfun)

    # Vector Field from Score
    ax4 = Axis(f2[2, 3], title="Vector Field from Score", xlabel="x", ylabel="y")
    vf2 = plot_vector_field!(ax4, vecfun)

    # Add colorbar for vector fields
    Colorbar(f2[1:2, 4], vf1, label="Vector magnitude")

    display(f2)
    save("figures/fig2_pub.png", f2, resolution=resolution)
    return f2
end

# Create individual figures first
f = plot_correlations_pub(corr_Qc_11, corr_Qc_12, corr_Qc_21, corr_Qc_22, corr_gen)

kde_obs = kde(obs')
kde_clustered = kde(trj_clustered')

f2 = plot_pdfs_and_vectorfields_pub(kde_obs, kde_clustered, ∇U)
