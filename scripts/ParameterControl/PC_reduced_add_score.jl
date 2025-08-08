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
using Random
using QuadGK
using Plots
using Base.Threads
using StatsBase
##

# Coefficients of the reduced model
a = -0.0222
b = -0.2
c = 0.0494
F_tilde = 0.6
s = 0.7071

function F(x,t; F_tilde=F_tilde)
    u = x[1]
    return [F_tilde + a * u + b * u^2 - c * u^3]
end

function sigma(x,t)
    return s/√2
end

function score_true(x; F_tilde=F_tilde)
    u = x[1]
    return [2 * (F_tilde + a*u + b*u^2 - c*u^3) / (s^2)]
end

function normalize_f(f, x, M, S)
    return (f(x .* S .+ M) .* S)[:]
end

function unnormalize_f(f_norm, x, M, S)
    return f_norm((x .- M) ./ S) ./ S
end

function score_true_norm(x)
    return normalize_f(score_true, x, M, S)
end

dim = 1
dt = 0.01
Nsteps = 100_000_000
obs_nn = evolve([0.0], dt, Nsteps, F, sigma; timestepper=:euler, resolution=10)
M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S

autocov_obs = zeros(dim, 300)
for i in 1:dim
    autocov_obs[i,:] = autocovariance(obs[i,1:1000000]; timesteps=300)
end

kde_obs = kde(obs[200:end])

autocov_obs_mean = mean(autocov_obs, dims=1)

plt1 = Plots.plot(autocov_obs_mean[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")
plt2 = Plots.plot(kde_obs.x, kde_obs.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")

Plots.plot(plt1, plt2, layout=(2, 1), size=(800, 800))

##
############################ CLUSTERING ####################

obs_uncorr = obs[:, 1:1:end]
normalization = false
σ_value = 0.05

averages, averages_residual, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.0005, do_print=true, conv_param=0.001, normalization=normalization)

if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
    inputs_targets_residual, M_averages_values_residual, m_averages_values_residual = generate_inputs_targets(averages_residual, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
    inputs_targets_residual = generate_inputs_targets(averages_residual, centers, Nc; normalization=false)
end

centers_sorted_indices = sortperm(centers[1,:])
centers_sorted = centers[:,centers_sorted_indices][:]
scores = .- averages[:,centers_sorted_indices][:] ./ σ_value
scores_true = [score_true_norm([centers_sorted[i]])[1] for i in eachindex(centers_sorted)]

Plots.scatter(centers_sorted, scores, color=:blue)
Plots.plot!(centers_sorted, scores_true, color=:red)
##
#################### TRAINING WITH CLUSTERING LOSS ####################
inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
normalization = false
σ_value = 0.05

@time nn_clustered, loss_clustered = train(inputs_targets, 10000, 128, [dim, 50, 25, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.(x))[:] ./ σ_value

function score_clustered_nn(x)
    return unnormalize_f(score_clustered, x, M, S)
end

# Save the trained neural network (using safe method)
save_model_safe(nn_clustered_cpu, "NNs/nn_reduced_add", [dim, 50, 25, dim]; activation=swish, last_activation=identity)

Plots.plot(loss_clustered)

##
#################### TRAINING WITH VANILLA LOSS ####################
# σ_value = 0.05
# @time nn_vanilla, loss_vanilla = train(obs_uncorr, 40, 16, [dim, 50, 25, dim], σ_value; use_gpu=true, opt=Adam(0.001))
# nn_vanilla_cpu = nn_vanilla |> cpu
# score_vanilla(x) = .- nn_vanilla_cpu(Float32.([x...])) ./ σ_value
# Plots.plot(loss_vanilla)
##
#################### VECTOR FIELDS ####################
#plotly()
gr()
xax = [centers_sorted[1]:0.005:centers_sorted[end]...]

s_true = [score_true_norm([xax[i]])[1] for i in eachindex(xax)]
s_gen = [score_clustered([xax[i]])[1] for i in eachindex(xax)]

Plots.scatter(centers_sorted, scores, color=:blue)
Plots.plot!(xax, s_true, label="True", xlabel="X", ylabel="Force", title="Forces")
Plots.plot!(xax, s_gen, label="Learned")

##
xax_nn = [-2.5:0.01:3.5...]
s_true_nn = [score_true([xax_nn[i]])[1] for i in eachindex(xax_nn)]
s_gen_nn = [score_clustered_nn([xax_nn[i]])[1] for i in eachindex(xax_nn)]
plot(xax_nn, s_true_nn, label="True", xlabel="X", ylabel="Force", title="Forces")
plot!(xax_nn, s_gen_nn, label="Learned")

# plot(obs_nn[1,1:100:100000])

##
xax = [-2.5:0.05:3.5...]

pdf_obs = compute_density_from_score(xax, score_true)
pdf_kgmm = compute_density_from_score(xax, score_clustered_nn)

Plots.plot(xax, pdf_obs, label="True", xlabel="X", ylabel="Density", title="True PDF")
Plots.plot!(xax, pdf_kgmm, label="Learned", xlabel="X", ylabel="Density", title="Learned PDF")


##

