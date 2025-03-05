using Pkg
Pkg.activate(".")
Pkg.instantiate()
##

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

N = 2

file = h5open("/Users/ludovico/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Documents/JuliaProjects/Stroboscopic/data/stroboscopic_N$N.h5", "r")
obs_nn = read(file["obs"])
dt = read(file["dt"])
τ = read(file["tau"])
k₀ = read(file["k0"])
alpha = read(file["alpha"])
close(file)

obs = (obs_nn .- mean(obs_nn, dims=2)) ./ std(obs_nn, dims=2)
obs = obs[[1], :]
# obs[2,:] .*= 0.1
dim = size(obs, 1)

autocov_obs = zeros(dim, 600)
for i in 1:dim
    autocov_obs[i,:] = autocovariance(obs[i,:]; timesteps=600)
end

autocov_obs_mean = mean(autocov_obs, dims=1)

Plots.plot(autocov_obs_mean[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")
##
obs_uncorr = obs[:, 1:20:end]

plot(obs[1,1:1000])

plotly()
Plots.scatter(obs_uncorr[1,1:10000], markersize=1, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

normalization = false
σ_values = [0.15, 0.10, 0.05]
C = []
scores_residual = []
scores = []
averages_vec, averages_residual_vec, centers_vec, Nc_vec = [], [], [], []

for σ_value in σ_values
    averages, averages_residual, centers, Nc, _ = f_tilde_ssp(σ_value, obs_uncorr; prob=0.001, do_print=true, conv_param=0.001, normalization=normalization)
    push!(averages_vec, averages)
    push!(averages_residual_vec, averages_residual)
    push!(centers_vec, centers)
    push!(Nc_vec, Nc)
    centers_sorted_indices = sortperm(centers[1,:])
    centers_sorted = centers[:,centers_sorted_indices][:]
    scores_temp = .- averages[:,centers_sorted_indices][:] ./ σ_value
    scores_residual_temp = (averages_residual[:,centers_sorted_indices][:] .- centers_sorted)./ σ_value^2
    push!(C, centers_sorted)
    push!(scores_residual, scores_residual_temp)
    push!(scores, scores_temp)
end

##
gr()

Plots.plot(C[1], scores[1], label="σ = $(σ_values[1])", xlabel="x", ylabel="score", lw=2, dpi=300)
Plots.plot!(C[2], scores[2], label="σ = $(σ_values[2])", lw=2)
Plots.plot!(C[3], scores[3], label="σ = $(σ_values[3])", lw=2)

#savefig("figures/averages_N$N.png")
##
#################### TRAINING WITH CLUSTERING LOSS ####################

σ_index = 3
σ_value = σ_values[σ_index]
averages = averages_vec[σ_index]
centers = centers_vec[σ_index]
Nc = Nc_vec[σ_index]

if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
end

@time nn_clustered, loss_clustered = train(inputs_targets, 5000, 8, [dim, 128, 64, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value

# @time nn_clustered_residual, loss_clustered_residual = train(inputs_targets_residual, 2000, 8, [dim, 128, 64, dim]; use_gpu=true, activation=swish, last_activation=identity)
# if normalization == true
#     nn_clustered_residual_cpu  = Chain(nn_clustered_residual, x -> x .* (M_averages_values_residual .- m_averages_values_residual) .+ m_averages_values_residual) |> cpu
# else
#     nn_clustered_residual_cpu = nn_clustered_residual |> cpu
# end
# score_clustered_residual(x) = (nn_clustered_residual_cpu(Float32.([x...]))[:] .- x) ./ σ_value^2

Plots.plot(loss_clustered)
# Plots.plot!(loss_clustered_residual)


##
xax = [-2:0.01:2...]
scores_nn = [score_clustered_corr(xax[i])[1] for i in eachindex(xax)]
# scores_nn_residual = [score_clustered_residual(centers_sorted[i])[1] for i in eachindex(centers_sorted)]

plotly()
Plots.plot(C[σ_index], scores[σ_index], label="σ = $(σ_values[σ_index])", xlabel="x", ylabel="score", lw=2)
Plots.plot!(xax, scores_nn, label="σ = $(σ_values[σ_index])", lw=2)
# Plots.plot!(centers_sorted, scores_nn_residual, color=:green)

##
#################### SAMPLES GENERATION ####################

function score_clustered_corr(x; a=(-C[σ_index][1]+C[σ_index][end])/2, K=100)
    if x[1] >= -a && x[1] <= 0.0
        return 0.5*score_clustered(x) .- 0.5*score_clustered(.- x)
    elseif x[1] > 0.0 && x[1] <= a
        return (0.5*score_clustered(x) .- 0.5*score_clustered(.- x))
    elseif x[1] < -a
        return [-(x[1]+a) * K + score_clustered(-a)[1]]
    elseif x[1] > a
        return [-(x[1]-a) * K + score_clustered(a)[1]]
    end
end

score_clustered_xt(x,t) = score_clustered_corr(x)
sigma_I(x,t) = 1.0

trj_clustered = evolve(obs[:,1000], 0.01*dt, 10000000, score_clustered_xt, sigma_I; timestepper=:euler, resolution=20)

kde_clustered1 = kde(trj_clustered[1,:])
kde_true1 = kde(obs[1,:])

gr()
plt1 = Plots.plot(kde_clustered1.x, kde_clustered1.density, label="Sampled PDF", xlabel="x", ylabel="Density", title="PDFs", lw=2, dpi=300)
plt1 = Plots.plot!(kde_true1.x, kde_true1.density, label="Observed PDF", xlabel="x", ylabel="Density", lw=2)

savefig("figures/sampled_pdf_N$N.png")
##
gr()
plot(dt:dt:30000*dt, trj_clustered[1,10001:40000], xlabel="t", ylabel="x", legend=false, title="Sampled Trajectory", dpi=300)

savefig("figures/sampled_trajectory_N$N.png")