using Pkg
Pkg.activate(".")
Pkg.instantiate()

##
using Statistics
using LinearAlgebra
using HDF5
using Plots
using KernelDensity
using LaTeXStrings
using StatsPlots
using ClustGen
using Flux
using BSON
using ProgressBars
using StatsBase
using Revise
using MarkovChainHammer


# === Load all PCA reconstruction variables from file ===
data_dir = joinpath(@__DIR__, "PlaSim_data")
pc_scores_file = joinpath(data_dir, "pc_scores.h5")
println("Loading PCA variables from $pc_scores_file ...")
h5f = h5open(pc_scores_file, "r")
pc_scores = read(h5f["pc_scores"])
year_mean = read(h5f["year_mean"])
data_mean = read(h5f["data_mean"])
eigenvecs = read(h5f["eigenvecs"])
times = read(h5f["times"])
close(h5f)
println("Loaded pc_scores with shape ", size(pc_scores))
println("Loaded year_mean with shape ", size(year_mean))
println("Loaded data_mean with shape ", size(data_mean))
println("Loaded eigenvecs with shape ", size(eigenvecs))
println("Loaded times with shape ", size(times))
##

times_sin = sin.(2π * times / 360)
times_cos = cos.(2π * times / 360)
times_mod = vcat(reshape(times_sin, 1, :), reshape(times_cos, 1, :))
obs_nn = vcat(times_mod, pc_scores[1:20,:])

# M = mean(obs_nn, dims=2)
# S = std(obs_nn, dims=2)
# obs = (obs_nn .- M) ./ S

M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = zeros(size(obs_nn))
obs[1,:] = (obs_nn[1,:] .- M[1]) ./ S[1]
obs[2:end,:] = (obs_nn[2:end,:] .- M[2]) ./ S[2]

dim = size(obs, 1)

plotly()
obs_uncorr = obs[:, 1:1:end]

Plots.scatter(obs_uncorr[1,1:10000], obs_uncorr[3,1:10000], markersize=1, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

normalization = false
σ_value = 0.05

# averages, _, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.000025, do_print=true, conv_param=0.005, normalization=normalization)
averages, _, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.00001, do_print=true, conv_param=0.02, normalization=normalization)


if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
end

# plotly()
# targets_norm = [norm(averages[:,i]) for i in eachindex(centers[1,:])]
# Plots.scatter(centers[1,:], centers[2,:], centers[3,:], marker_z=targets_norm, color=:viridis)

##
#################### TRAINING WITH CLUSTERING LOSS ####################
@time nn_clustered, loss_clustered = train(inputs_targets, 300, 128, [dim, 128, 64, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.(x))[:] ./ σ_value
Plots.plot(loss_clustered)

##

normalization = false
σ_value = 0.05

# averages, _, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.000025, do_print=true, conv_param=0.005, normalization=normalization)
averages_phi, _, centers_phi, Nc_phi, ssp_phi = f_tilde_ssp(σ_value, obs_uncorr; prob=0.00025, do_print=true, conv_param=0.01, normalization=normalization)


if normalization == true
    inputs_targets_phi, M_averages_values_phi, m_averages_values_phi = generate_inputs_targets(averages_phi, centers_phi, Nc_phi; normalization=true)
else
    inputs_targets_phi = generate_inputs_targets(averages_phi, centers_phi, Nc_phi; normalization=false)
end

labels = [ssp_phi.embedding(obs[:,i]) for i in eachindex(obs[1,1:100000])]
averages_c, centers_c, Nc_c, labels_c = cleaning(averages_phi, centers_phi, labels)

##
dt = 1.0
Q = 1.3*generator(labels_c;dt=dt)
P_steady = steady_state(Q)
         
tsteps = 25
res = 1

auto_obs = zeros(dim, tsteps)
auto_Q = zeros(dim, tsteps)

for i in 1:dim
    auto_obs[i,:] = autocovariance(obs[i,1:res:end]; timesteps=tsteps) 
    auto_Q[i,:] = autocovariance(centers_c[i,:], Q, [0:dt*res:Int(res * (tsteps-1) * dt)...])
end

plt = Plots.plot(auto_obs[3,:])
plt = Plots.plot!(auto_Q[3,:])
##

gradLogp = zeros(dim, Nc_c)
for i in 1:Nc_c
    gradLogp[:,i] = - averages_c[:,i] / σ_value
end

C0 = centers_c * (centers_c * Diagonal(P_steady))'
C1_Q = centers_c * Q * (centers_c * Diagonal(P_steady))'
C1_grad = gradLogp * (centers_c * Diagonal(P_steady))'
Φ = C1_Q * inv(C1_grad)

eye = Matrix{Float64}(I, dim, dim)  # If you need a dense identity matrix, use 'eye' instead of 'I'
Σ = cholesky(0.5*(Φ .+ Φ'+0.001*eye)).L
println("Σ = ", Σ)
##
#################### SAMPLES GENERATION ####################

Φ = Matrix{Float64}(I, dim, dim)
Σ_full = Matrix{Float64}(I, dim, dim)
# The dimension of the physical (observed) system
dim_obs = dim - 2  # Now we have 2 time dimensions (sin and cos)

# --- Corrected Drift and Noise Functions ---

# 1. Calculate the full Cholesky factor from the symmetric part of Φ
#    This Σ_full matrix is (dim x dim)
#Σ_full = cholesky(0.5 * (Φ .+ Φ'+0.001*eye)).L

# 2. Define a helper function to compute the full score vector.
#    This is for clarity and avoids repetition.
#    (Assumes M and S are pre-calculated mean/std for normalization)
function full_score(x_obs, t)
    # Augment the state with the normalized time variables (sin and cos)
    time_sin = (sin(2π * t / 360) - M[1]) / S[1]
    time_cos = (cos(2π * t / 360) - M[2]) / S[2]
    augmented_state = vcat(time_sin, time_cos, x_obs)
    
    # Return the full score vector (size: dim x 1)
    return score_clustered(augmented_state)
end

# 3. Define the corrected drift function for the physical variables 'x'
#    It uses the bottom `dim_obs` rows of the full Φ matrix and the full score vector.
drift_x_corrected(x_obs, t) = Φ[3:end, :] * full_score(x_obs, t)

# 4. Define the corrected noise matrix 'g' for the g*dW term
#    It uses the bottom `dim_obs` rows of the full Σ matrix.
#    This returns a matrix of size (dim_obs x dim)
sigma_x_corrected(x_obs, t) = Σ_full[3:end, :]


# --- Integration Step ---

Nsteps = 500000
dt = 0.01

# The initial condition vector should have size `dim_obs`
initial_condition = zeros(dim_obs)

# Evolve the system with the corrected functions.
# Note: Your SDE solver (`evolve`) must now use a noise vector `dW` that has `dim`
# components, not `dim_obs`, because sigma_x_corrected is a (dim_obs x dim) matrix.
trj_clustered = evolve(
    initial_condition,
    dt,
    100 * Nsteps,
    drift_x_corrected,
    sigma_x_corrected;
    timestepper=:euler,
    resolution=100,
    boundary=[-5, 5]
)

res = 1
tsteps = 26
auto_clustered = zeros(dim_obs, tsteps)
auto_obs = zeros(dim_obs, tsteps)

for i in 1:dim_obs
    auto_clustered[i,:] = autocovariance(trj_clustered[i,:]; timesteps=tsteps) 
    auto_obs[i,:] = autocovariance(obs[i+1,:]; timesteps=tsteps)
end

## Plotting
gr()

t_mod = obs[1,1:Nsteps+1]
obs_clustered = vcat(reshape(t_mod, 1, :), trj_clustered)

# --- Univariate PDFs (Figure 1) ---
dim_obs = size(obs, 1) - 1  # Number of physical variables
colors = [:blue, :red]
labels = ["Generated", "Observed"]

#(Removed intermediate PDF plots, only final figure will be displayed)

# === Improved Combined 6-column Figure: Trajectories | Autocovariances | Univariate PDFs | Bivariate PDFs (Generated) | Bivariate PDFs (Observed) ===

default(fontfamily="Computer Modern", lw=2, framestyle=:box, legendfontsize=10, guidefontsize=14, tickfontsize=12, titlefontsize=16)

traj_plots = []
for i in 1:dim_obs
    p = plot(
        1:1000, trj_clustered[i,1:1000],
        label="Generated", color=:blue, lw=2,
        xlabel=i==dim_obs ? "Time step" : "", ylabel="x_$i",
        legend=:topright, grid=false, framestyle=:box, margin=5Plots.mm,
        xguidefont=font(13), yguidefont=font(13), xtickfont=font(11), ytickfont=font(11),
        title=i==1 ? "Trajectory" : ""
    )
    if size(obs, 1) >= i+1
        plot!(p, 1:1000, obs[i+1,1:1000], label="Observed", color=:red, lw=2)
    end
    push!(traj_plots, p)
end

auto_plots = []
for i in 1:dim_obs
    p = plot(
        0:tsteps-1, auto_clustered[i, :],
        label="Generated", color=:blue, lw=2,
        xlabel=i==dim_obs ? "Lag" : "", ylabel="Autocov(x_$i)",
        legend=:topright, grid=false, framestyle=:box, margin=5Plots.mm,
        xguidefont=font(13), yguidefont=font(13), xtickfont=font(11), ytickfont=font(11),
        title=i==1 ? "Autocovariance" : ""
    )
    plot!(p, 0:tsteps-1, auto_obs[i, :], label="Observed", color=:red, lw=2)
    push!(auto_plots, p)
end

# Univariate PDFs
uni_plots = []
for i in 1:dim_obs
    p = plot()
    kde_gen = kde(obs_clustered[i+1, :])
    kde_obs = kde(obs[i+1, :])
    plot!(p, kde_gen.x, kde_gen.density, color=:blue, label="Generated", lw=2, grid=false)
    plot!(p, kde_obs.x, kde_obs.density, color=:red, label="Observed", lw=2, grid=false)
    ylabel!(p, "pdf")
    xlabel!(p, "x_$i")
    # Only top row gets a title
    if i == 1
        title!(p, "Univariate PDF")
    end
    push!(uni_plots, p)
end

# Bivariate PDFs (Generated and Observed)
bi_gen_plots = []
bi_obs_plots = []
for i in 1:dim_obs
    # Generated
    p_gen = plot()
    kde_gen = kde((obs_clustered[1, :], obs_clustered[i+1, :]))
    heatmap!(p_gen, kde_gen.x, kde_gen.y, kde_gen.density,
        color=:blues, ylabel="t_mod", xlabel="x_$i",
        colorbar=false)
    if i == 1
        title!(p_gen, "Bivariate Gen")
    end
    push!(bi_gen_plots, p_gen)
    # Observed
    p_obs = plot()
    kde_obs = kde((obs[1, :], obs[i+1, :]))
    heatmap!(p_obs, kde_obs.x, kde_obs.y, kde_obs.density,
        color=:reds, ylabel="t_mod", xlabel="x_$i",
        colorbar=false)
    if i == 1
        title!(p_obs, "Bivariate Obs")
    end
    push!(bi_obs_plots, p_obs)
end

# Combine all plots in a 5-column panel: [Trajectory | Autocov | Univariate PDF | Bivariate Gen | Bivariate Obs]
panel_plots = []
for i in 1:dim_obs
    push!(panel_plots, traj_plots[i])
    push!(panel_plots, auto_plots[i])
    push!(panel_plots, uni_plots[i])
    push!(panel_plots, bi_gen_plots[i])
    push!(panel_plots, bi_obs_plots[i])
end

fig = plot(
    panel_plots...,
    layout = (dim_obs, 5),
    size = (2400, 300*dim_obs),
    left_margin=8Plots.mm, bottom_margin=8Plots.mm, top_margin=8Plots.mm, right_margin=8Plots.mm,
    titlefont=font(20, "Computer Modern"),
    dpi=300,
    plot_title=""
)

# savefig(fig, "combined_figure.pdf")

