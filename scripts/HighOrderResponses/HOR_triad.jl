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

function F(u, t; L11 = -2.0, L12 = 0.2, L13 = 0.1, g2 = 0.6, g3 = 0.4, I = 1.0, ϵ = 0.1)
    u1, u2, u3 = u
    du1 = (L11 * u1 + L12 * u2 + L13 * u3 + I * u1 * u2)
    du2 = (-L12 * u1 - I * u1^2 - g2 * u2 / ϵ)
    du3 = (-L13 * u1 - g3 * u3 / ϵ)
    return [du1, du2, du3]
end

function sigma(x, t; s2 = 1.2, s3 = 0.8, ϵ = 0.1)
    sigma1 = 0.0
    sigma2 = s2 / sqrt(2ϵ)
    sigma3 = s3 / sqrt(2ϵ)
    return [sigma1, sigma2, sigma3]
end

dim = 3
dt = 0.02
Nsteps = 20000000
obs_nn = evolve(randn(dim), dt, Nsteps, F, sigma; timestepper=:rk4)
obs = (obs_nn .- mean(obs_nn, dims=2)) ./ std(obs_nn, dims=2)

autocov_obs = zeros(dim, 300)
for i in 1:dim
    autocov_obs[i,:] = autocovariance(obs[i,:]; timesteps=300)
end

autocov_obs_mean = mean(autocov_obs, dims=1)

plot(autocov_obs_mean[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")

##

cov(obs')
plotly()

obs_uncorr = obs[:, 1:100:end]

scatter(obs_uncorr[1,1:10000], obs_uncorr[2,1:10000], obs_uncorr[3,1:10000], markersize=1, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

normalization = false
σ_value = 0.1

averages, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.001, do_print=true, conv_param=0.001, normalization=normalization)

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

@time nn_clustered, loss_clustered = train(inputs_targets, 1000, 64, [dim, 128, 64, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value
# cluster_loss = check_loss(obs_uncorr, nn_clustered_cpu, σ_value)
# println(cluster_loss)
# Plots.plot(loss_clustered)

##
#################### TRAINING WITH VANILLA LOSS ####################

@time nn_vanilla, loss_vanilla = train(obs_uncorr, 200, 32, [dim, 128, 64, dim], σ_value; use_gpu=true, opt=Adam(0.001))
nn_vanilla_cpu = nn_vanilla |> cpu
score_vanilla(x) = .- nn_vanilla_cpu(Float32.([x...])) ./ σ_value
Plots.plot(loss_vanilla)
hline!([cluster_loss])

##
#################### SAMPLES GENERATION ####################
Σ = I(dim)
# score_test(x) = score_clustered(x .+ m_gen)
trj_clustered = sample_langevin_Σ(1000000, 0.5*dt, score_clustered, randn(dim), Σ; seed=123, res = 2, boundary = [-7,7])

kde_obs_x = kde(obs[1,:])
kde_obs_y = kde(obs[2,:])
kde_obs_z = kde(obs[3,:])

kde_clustered_x = kde(trj_clustered[1,:])
kde_clustered_y = kde(trj_clustered[2,:])
kde_clustered_z = kde(trj_clustered[3,:])

gr()
plt1 = Plots.plot(kde_obs_x.x, kde_obs_x.density, xlabel="X", ylabel="Density", title="Observed PDF X")
plt2 = Plots.plot(kde_obs_y.x, kde_obs_y.density, xlabel="Y", ylabel="Density", title="Observed PDF Y")
plt3 = Plots.plot(kde_obs_z.x, kde_obs_z.density, xlabel="Z", ylabel="Density", title="Observed PDF Z")

plt1 = plot!(plt1, kde_clustered_x.x, kde_clustered_x.density, xlabel="X", ylabel="Density", title="Sampled PDF X")
plt2 = plot!(plt2, kde_clustered_y.x, kde_clustered_y.density, xlabel="Y", ylabel="Density", title="Sampled PDF Y")
plt3 = plot!(plt3, kde_clustered_z.x, kde_clustered_z.density, xlabel="Z", ylabel="Density", title="Sampled PDF Z")

plot(plt1, plt2, plt3, layout=(3, 1), size=(800, 800))
##

# m_gen = mean(trj_clustered, dims=2)
# s_gen = std(trj_clustered, dims=2)

mean(trj_clustered, dims=2)
std(trj_clustered, dims=2)

##
kde_obs_xy = kde(obs[[1,2],:]')
kde_obs_xz = kde(obs[[1,3],:]')
kde_obs_yz = kde(obs[[2,3],:]')

kde_clustered_xy = kde(trj_clustered[[1,2],:]')
kde_clustered_xz = kde(trj_clustered[[1,3],:]')
kde_clustered_yz = kde(trj_clustered[[2,3],:]')

gr()

plt1 = Plots.heatmap(kde_obs_xy.x, kde_obs_xy.y, kde_obs_xy.density, xlabel="X", ylabel="Y", title="Observed PDF XY")
plt2 = Plots.heatmap(kde_obs_xz.x, kde_obs_xz.y, kde_obs_xz.density, xlabel="X", ylabel="Z", title="Observed PDF XZ")
plt3 = Plots.heatmap(kde_obs_yz.x, kde_obs_yz.y, kde_obs_yz.density, xlabel="Y", ylabel="Z", title="Observed PDF YZ")
plt4 = Plots.heatmap(kde_clustered_xy.x, kde_clustered_xy.y, kde_clustered_xy.density, xlabel="X", ylabel="Y", title="Sampled PDF XY")
plt5 = Plots.heatmap(kde_clustered_xz.x, kde_clustered_xz.y, kde_clustered_xz.density, xlabel="X", ylabel="Z", title="Sampled PDF XZ")
plt6 = Plots.heatmap(kde_clustered_yz.x, kde_clustered_yz.y, kde_clustered_yz.density, xlabel="Y", ylabel="Z", title="Sampled PDF YZ")

Plots.plot(plt1, plt2, plt3, plt4, plt5, plt6)
##

using ClustGen
f(t) = 1.0

res_trj = 50
steps_trj = 10000
trj = obs[:,1:res_trj:steps_trj*res_trj]

u(x) = [0.1, 0.0, 0.0]
#u(x) = [0.0, 0.0, 0.3] #./ norm([1, 2, 3],2)
div_u(x) = -0.0
m = mean(obs, dims=2)
Obs(x) = (x .- m).^2
dim_Obs = 3

F_pert(x,t) = F(x,t) + u(x) * f(t)

invC0 = inv(cov(obs'))
score_lin(x) = - invC0*(x .- m)
score_gen(x) = score_clustered(x .+ 0.01)


n_tau = 30
δObs_num = generate_numerical_response(F, F_pert, dim, dt, n_tau, 10000, sigma, Obs, dim_Obs; n_ens=1000, resolution=res_trj)
R_num = generate_numerical_response3(F, u, dim, dt, n_tau, 10000, sigma, Obs, dim_Obs; n_ens=1000, resolution=res_trj)
R_lin, δObs_lin = generate_score_response(trj, u, div_u, f, score_lin, res_trj*dt, n_tau, Obs, dim_Obs)
R_gen, δObs_gen = generate_score_response(trj, u, div_u, f, score_gen, res_trj*dt, n_tau, Obs, dim_Obs)


##

index = 1

plt1 = plot(0:dt*res_trj:(n_tau-0.5)*dt*res_trj, δObs_num[index,:], xlabel="Lag", ylabel="Response", title="Responses", label="Numerical")
plt1 = plot!(0:dt*res_trj:(n_tau-0.5)*dt*res_trj, δObs_lin[index,:], xlabel="Lag", ylabel="Response", label="Linear")
plt1 = plot!(0:dt*res_trj:(n_tau-0.5)*dt*res_trj, δObs_gen[index,:] , xlabel="Lag", ylabel="Response", label="Generative")

plt2 = plot(0:dt*res_trj:(n_tau-0.5)*dt*res_trj, R_num[index,:], xlabel="Lag", ylabel="Response", title="Responses", label="Numerical")
plt2 = plot!(0:dt*res_trj:(n_tau-0.5)*dt*res_trj, R_lin[index,:], xlabel="Lag", ylabel="Response", label="Linear")
plt2 = plot!(0:dt*res_trj:(n_tau-0.5)*dt*res_trj, R_gen[index,:], xlabel="Lag", ylabel="Response", label="Generative")

plot(plt1, plt2, layout=(2, 1), size=(800, 800))

##