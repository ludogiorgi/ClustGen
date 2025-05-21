using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Plots
using Revise
using MarkovChainHammer
using ClustGen
using KernelDensity
using HDF5
using Flux
using BSON
using LinearAlgebra
using ProgressBars
using Random
using Distributions
##

# Define the rhs of the Lorenz system for later integration with evolve function, changes wrt GMM_Lorenz63: x is a 4 dimensional vector that contains x, y1,y2 and y3.
function F(x, t, σ, ε ; µ=10.0, ρ=28.0, β=8/3)
    dx = x[1] * (1 - x[1]^2) + (σ / ε) * x[3]
    dy1 = µ/ε^2 * (x[3] - x[2])
    dy2 = 1/ε^2 * (x[2] * (ρ - x[4]) - x[3])
    dy3 = 1/ε^2 * (x[2] * x[3] - β * x[4])
    return [dx, dy1, dy2, dy3]
end

function sigma(x, t; noise = 1/√2)
    sigma1 = 0.0
    sigma2 = 0.0
    sigma3 = 0.0
    sigma4 = 0.0 #Added: This is for the 4th variable
    return [sigma1, sigma2, sigma3, sigma4]
end

function score_true(x)
    u = x[1]
    s = sigma_eff * (σ / ε)
    return [2 * (u * (1-u^2)) / (s^2)]
end

dim = 1
dt = 0.01
ε = 0.5
σ = 0.08
Nsteps = 1000000000
f = (x, t) -> F(x, t, σ, ε)
obs = evolve(randn(4), dt, Nsteps, f, sigma; timestepper=:rk4, resolution=100)

autocov_obs = zeros(dim, 1000)
for i in 1:dim
    autocov_obs[i,:] = autocovariance(obs[i,1:1000000]; timesteps=1000)
end

std(obs[3,:])
kde_obs = kde(obs[1,200:end])

autocov_obs_mean = mean(autocov_obs, dims=1)

plt1 = Plots.plot(autocov_obs_mean[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")
plt2 = Plots.plot(kde_obs.x, kde_obs.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")

Plots.plot(plt1, plt2, layout=(2, 1), size=(800, 800))

##
obs_uncorr = obs[1:1, 1:1:end]

Plots.scatter(obs_uncorr[1,1:10000], markersize=2, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

Plots.plot(obs[3,1:10:1000])

normalization = false
σ_value = 0.05

averages, averages_residual, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.002, do_print=true, conv_param=0.02, normalization=normalization)

if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
    inputs_targets_residual, M_averages_values_residual, m_averages_values_residual = generate_inputs_targets(averages_residual, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
    inputs_targets_residual = generate_inputs_targets(averages_residual, centers, Nc; normalization=false)
end


##
centers_sorted_indices = sortperm(centers[1,:])
centers_sorted = centers[:,centers_sorted_indices][:]
scores = .- averages[:,centers_sorted_indices][:] ./ σ_value
# scores_true = [ score_true(centers_sorted[i])[1] for i in eachindex(centers_sorted)]

Plots.plot(centers_sorted[:], scores[:], label="Learned", xlabel="X", ylabel="Force", title="Forces", xlims=(-1.3, 1.3), ylims=(-5, 5))
# Plots.plot!(centers_sorted[:], scores_true, label="True", xlabel="X", ylabel="Force", title="Forces")

##
#################### TRAINING WITH CLUSTERING LOSS ####################

@time nn_clustered, loss_clustered = train(inputs_targets, 5000, 16, [dim, 50, 25, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value

Plots.plot(loss_clustered)

##
#################### VECTOR FIELDS ####################

xax = [-1.25:0.005:1.25...]

# s_true = [score_true_norm(xax[i])[1] for i in eachindex(xax)]
s_gen = [score_clustered(xax[i])[1] for i in eachindex(xax)]

Plots.scatter(centers_sorted, scores, color=:blue, alpha=0.2, label="Cluster centers", xlims=(-1.3, 1.3), ylims=(-5, 5))
# Plots.plot!(xax, s_true, label="True", xlabel="X", ylabel="Force", title="Forces", lw=3)
Plots.plot!(xax, s_gen, label="Learned", lw=3)

##
xax = [-1.6:0.02:1.6...]

pdf_kgmm = compute_density_from_score(xax, score_clustered)

Plots.plot(xax, pdf_kgmm, label="Learned", xlabel="x", ylabel="Density", title="PDFs", lw=3, legend=:bottomright)
Plots.plot!(kde_obs.x, kde_obs.density, label="Observed",lw=3)

##
plotly()
xax2 = [0:dt:50*dt...]

C = autocovariance(obs[3,:], timesteps=51)
sigma_eff = sqrt(2*sum(C[2:end]) * dt)

Plots.plot(xax2, C, label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")
