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

function F(x,t)
    # Unpack variables
    x1 = x[1]
    x2 = x[2]
    
    # Compute common exponentials
    expT1 = exp(-x1^2 - (x2 - 1/3)^2)
    expT2 = exp(-x1^2 - (x2 - 5/3)^2)
    expT3 = exp(- (x1 + 1)^2 - x2^2)
    expT4 = exp(- (x1 - 1)^2 - x2^2)
    
    # Compute partial derivatives with respect to x
    dV_dx = -6*x1*expT1 +
             6*x1*expT2 +
             10*(x1 + 1)*expT3 +
             10*(x1 - 1)*expT4 +
             (4/5)*x1^3
    
    # Compute partial derivatives with respect to y
    dV_dy = -6*(x2 - 1/3)*expT1 +
             6*(x2 - 5/3)*expT2 +
             10*x2*expT3 +
             10*x2*expT4 +
             (4/5)*(x2 - 1/3)^3 - 1
    
    return [-dV_dx, -dV_dy]
end

function sigma(x, t; σ=0.7)
    sigma1 = σ
    sigma2 = σ
    return [sigma1, sigma2]
end

function score_true(x)
    return F(x, 0.0) ./ sigma(x, 0.0) .^ 2
end

dim = 2
dt = 0.05
Nsteps = 10000000
obs_nn = evolve([0.0, 0.0], dt, Nsteps, F, sigma; resolution = 1, timestepper=:rk4)
obs = obs_nn #(obs_nn .- mean(obs_nn, dims=2)) ./ std(obs_nn, dims=2)

autocov_obs = zeros(dim, 300)
for i in 1:dim
    autocov_obs[i,:] = autocovariance(obs[i,:]; timesteps=300)
end

autocov_obs_mean = mean(autocov_obs, dims=1)

Plots.plot(autocov_obs_mean[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")

##
obs_uncorr = obs[:, 1:1:end]

Plots.scatter(obs_uncorr[1,1:10000], obs_uncorr[2,1:10000], markersize=1, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

normalization = false
σ_value = 0.025

averages, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.001, do_print=true, conv_param=0.002, normalization=normalization)

if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
end

plotly()
targets_norm = [norm(averages[:,i]) for i in eachindex(centers[1,:])]
Plots.scatter(centers[1,:], centers[2,:], marker_z=targets_norm, color=:viridis)
##
#################### TRAINING WITH CLUSTERING LOSS ####################

@time nn_clustered, loss_clustered = train(inputs_targets, 1000, 64, [dim, 128, 64, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value
Plots.plot(loss_clustered)

##
#################### SAMPLES GENERATION ####################

invC0 = inv(cov(obs'))
score_qG(x) = - invC0*x

score_clustered_xt(x,t) = score_clustered(x)
score_qG_xt(x,t) = score_qG(x)
sigma_I(x,t) = 1.0

trj_clustered = evolve(zeros(dim), dt, 1000000, score_clustered_xt, sigma_I; timestepper=:rk4, resolution=1)
trj_qG = evolve(zeros(dim), dt, 1000000, score_qG_xt, sigma_I; timestepper=:rk4, resolution=1)

kde_clustered_x = kde(trj_clustered[1,:])
kde_true_x = kde(obs[1,:])
kde_qG_x = kde(trj_qG[1,:])

kde_clustered_y = kde(trj_clustered[2,:])
kde_true_y = kde(obs[2,:])
kde_qG_y = kde(trj_qG[2,:])

gr()

plt1 = Plots.plot(kde_clustered_x.x, kde_clustered_x.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
plt1 = Plots.plot!(kde_true_x.x, kde_true_x.density, label="True", xlabel="X", ylabel="Density", title="True PDF")
plt1 = Plots.plot!(kde_qG_x.x, kde_qG_x.density, label="qG", xlabel="X", ylabel="Density", title="qG PDF")

plt2 = Plots.plot(kde_clustered_y.x, kde_clustered_y.density, label="Observed", xlabel="Y", ylabel="Density", title="Observed PDF")
plt2 = Plots.plot!(kde_true_y.x, kde_true_y.density, label="True", xlabel="Y", ylabel="Density", title="True PDF")
plt2 = Plots.plot!(kde_qG_y.x, kde_qG_y.density, label="qG", xlabel="Y", ylabel="Density", title="qG PDF")

Plots.plot(plt1, plt2)
##
kde_obs_xy = kde(obs[[1,2],:]')

kde_clustered_xy = kde(trj_clustered[[1,2],:]')

gr()

plt1 = Plots.heatmap(kde_obs_xy.x, kde_obs_xy.y, kde_obs_xy.density, xlabel="X", ylabel="Y", title="Observed PDF XY")
plt2 = Plots.heatmap(kde_clustered_xy.x, kde_clustered_xy.y, kde_clustered_xy.density, xlabel="X", ylabel="Y", title="Sampled PDF XY", xrange=(kde_obs_xy.x[1], kde_obs_xy.x[end]), yrange=(kde_obs_xy.y[1], kde_obs_xy.y[end]))

Plots.plot(plt1, plt2)
##

using ClustGen
f(t) = 1.0

res_trj = 1
steps_trj = 100000
trj = obs[:,1:res_trj:steps_trj*res_trj]

ϵ = 0.05
u(x) = [0.05, 0.0]
div_u(x) = -0.0
m = mean(obs, dims=2)

F_pert(x,t) = F(x,t) + u(x) * f(t)

score_gen(x) = score_clustered(x)

dim_Obs = 2
n_tau = 100

R_num, δObs_num = zeros(dim, n_tau+1), zeros(dim, n_tau+1)
R_lin, δObs_lin = zeros(dim, n_tau+1), zeros(dim, n_tau+1)
R_true, δObs_true = zeros(dim, n_tau+1), zeros(dim, n_tau+1)
R_gen, δObs_gen = zeros(dim, n_tau+1), zeros(dim, n_tau+1)

#Obs(x) = reshape([(x[1] - m[1]), (x[1] - m[1])^2, (x[1] - m[1])*(x[2] - m[2])], 1, 3)

Obs(x) = x

R_num[:,:] = generate_numerical_response3(F, u, dim, dt, n_tau, 1000, sigma, Obs, dim_Obs; n_ens=10000, resolution=10*res_trj)
R_lin[:,:], δObs_lin[:,:] = generate_score_response(trj, u, div_u, f, score_qG, res_trj*dt, n_tau, Obs, dim_Obs)
R_gen[:,:], δObs_gen[:,:] = generate_score_response(trj, u, div_u, f, score_gen, res_trj*dt, n_tau, Obs, dim_Obs)
R_true[:,:], δObs_true[:,:] = generate_score_response(trj, u, div_u, f, score_true, res_trj*dt, n_tau, Obs, dim_Obs)


##

plt1 = Plots.plot(0:dt:n_tau*dt, R_num[1,:]./ϵ, label="Numerical", xlabel="Lag", ylabel="Response", title="Responses")
plt1 = Plots.plot!(0:dt:n_tau*dt, R_lin[1,:]./ϵ, label="Linear", xlabel="Lag", ylabel="Response", title="Responses")
plt1 = Plots.plot!(0:dt:n_tau*dt, R_gen[1,:]./ϵ, label="Generative", xlabel="Lag", ylabel="Response", title="Responses")
plt1 = Plots.plot!(0:dt:n_tau*dt, R_true[1,:]./ϵ, label="True", xlabel="Lag", ylabel="Response", title="Responses")

plt2 = Plots.plot(0:dt:n_tau*dt, R_num[2,:]./ϵ, label="Numerical", xlabel="Lag", ylabel="Response", title="Responses")
plt2 = Plots.plot!(0:dt:n_tau*dt, R_lin[2,:]./ϵ, label="Linear", xlabel="Lag", ylabel="Response", title="Responses")
plt2 = Plots.plot!(0:dt:n_tau*dt, R_gen[2,:]./ϵ, label="Generative", xlabel="Lag", ylabel="Response", title="Responses")
plt2 = Plots.plot!(0:dt:n_tau*dt, R_true[2,:]./ϵ, label="True", xlabel="Lag", ylabel="Response", title="Responses")

Plots.plot(plt1, plt2)