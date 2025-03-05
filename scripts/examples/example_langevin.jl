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

function ∇U_2D(x, t; A1=1.0, A2=1.2, B1=0.6, B2=0.3)
    ∇U1 = 2 * (x[1] + A1) * (x[1] - A1)^2 + 2 * (x[1] - A1) * (x[1] + A1)^2 + B1
    ∇U2 = 2 * (x[2] + A2) * (x[2] - A2)^2 + 2 * (x[2] - A2) * (x[2] + A2)^2 + B2
    return [-∇U1, -∇U2]
end

dim = 2
dt = 0.025
Nsteps = 20000000
sigma = 1.0
obs = evolve(randn(dim), dt, Nsteps, ∇U_2D, sigma)

autocov_obs = zeros(dim, 300)
for i in 1:dim
    autocov_obs[i,:] = autocovariance(obs[i,:]; timesteps=300) ./ std(obs[i,:])^2
end

autocov_obs_mean = mean(autocov_obs, dims=1)

plot(autocov_obs_mean[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")

##

obs_uncorr = obs[:, 1:200:end]

scatter(obs_uncorr[1,1:10000], obs_uncorr[2,1:10000], markersize=2, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

normalization = false
σ_value = 0.05

averages, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.0005, do_print=true, conv_param=0.001, normalization=normalization)

if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
end

targets_norm = [norm(averages[:,i]) for i in eachindex(centers[1,:])]
Plots.scatter(centers[1,:], centers[2,:], marker_z=targets_norm, color=:viridis)

##
#################### TRAINING WITH CLUSTERING LOSS ####################

@time nn_clustered, loss_clustered = train(inputs_targets, 1000, 128, [dim, 128, 64, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value
cluster_loss = check_loss(obs_uncorr, nn_clustered_cpu, σ_value)
println(cluster_loss)
Plots.plot(loss_clustered)

##
########### Additional training on the full data (not used in the draft) ############

@time nn_clustered, loss_clustered = train(obs_uncorr, 20, 128, nn_clustered, σ_value; use_gpu=true)

##
#################### TRAINING WITH VANILLA LOSS ####################

@time nn_vanilla, loss_vanilla = train(obs_uncorr, 200, 128, [dim, 128, 64, dim], σ_value; use_gpu=true, opt=Adam(0.001))
nn_vanilla_cpu = nn_vanilla |> cpu
score_vanilla(x) = .- nn_vanilla_cpu(Float32.([x...])) ./ σ_value
Plots.plot(loss_vanilla)
hline!([cluster_loss])

##
#################### VECTOR FIELDS ####################

gr()
force(x) = ∇U_2D(x, 0.0)
vf(x, y) = force([x,y])
vf_c(x, y) = score_clustered([x, y])
# vf_v(x, y) = score_vanilla([x, y])

n_grid = 30
d_grid = 1/10
c_grid = [((n_grid+1)*d_grid)/2, ((n_grid+1)*d_grid)/2]
grid = meshgrid(n_grid) .* d_grid .- c_grid

plt1 = vectorfield2d(vf_c, grid, xlabel="X", ylabel="Y", title="Clustered Force Field")
# plt2 = vectorfield2d(vf_v, grid, xlabel="X", ylabel="Y", title="Vanilla Force Field")
plt3 = vectorfield2d(vf, grid, xlabel="X", ylabel="Y", title="Observed Force Field")
# plot(plt1, plt2, plt3, layout=(3, 1), size=(800, 1200))
plot(plt1, plt3, layout=(2, 1), size=(800, 800))

##
#################### SAMPLES GENERATION ####################
Σ = I(dim)
trj_clustered = sample_langevin_Σ(100000, 0.1*dt, score_clustered, randn(2), Σ; seed=123, res = 10)
# trj_vanilla = sample_langevin_Σ(10000, dt, score_vanilla, randn(2), Σ; seed=123, res = 1)
#plot(trj_clustered[1,1:1500], trj_clustered[2,1:1500])
kde_clustered = kde(trj_clustered')
# kde_vanilla = kde(trj_vanilla')
kde_obs = kde(obs')

plt1 = heatmap(kde_clustered.x, kde_clustered.y, kde_clustered.density, xlabel="X", ylabel="Y", title="Clustered PDF")
# plt2 = heatmap(kde_vanilla.x, kde_vanilla.y, kde_vanilla.density, xlabel="X", ylabel="Y", title="Vanilla PDF")
plt3 = heatmap(kde_obs.x, kde_obs.y, kde_obs.density, xlabel="X", ylabel="Y", title="Observed PDF")
# plot(plt1, plt2, plt3, layout=(3, 1), size=(800, 1200))
plot(plt1, plt3, layout=(2, 1), size=(800, 800))

##
####################### AUTOCORRELATION ####################


reso = 100
labels = [ssp.embedding(obs[:,i]) for i in 1:reso:length(obs[1,:])]

averages, centers, Nc, labels = cleaning(averages, centers, labels)

gradLogp = zeros(dim, Nc)
for i in 1:Nc
    gradLogp[:,i] = - averages[:,i] / σ_value
end

Q = generator(labels)
P_steady = steady_state(Q)

Σ = computeSigma(centers, P_steady, Q, score_clustered, 5, 40, dt; iterations=25)

trj_clustered = sample_langevin_Σ(500000, 0.2*dt, score_clustered, randn(dim), Σ_fin; seed=123, res = 5)

res = 20
tsteps = 51

auto_Q = zeros(dim, tsteps)
auto_gen = zeros(dim, tsteps)
auto_obs = zeros(dim, tsteps)

for i in 1:dim
    auto_obs[i,:] = autocovariance(obs[i,1:res:length(trj_clustered[1,:])]; timesteps=tsteps) ./ std(obs[i,:])^2
    auto_Q[i,:] = autocovariance(centers[i,:], Q, [0:dt*res:Int(res * (tsteps-1) * dt)...]) ./ std(centers[i,:])^2
    auto_gen[i,:] = autocovariance(trj_clustered[i,1:res:end]; timesteps=tsteps) ./ std(trj_clustered[i,:])^2
end

auto_Q_mean = mean(auto_Q, dims=1)
auto_gen_mean = mean(auto_gen, dims=1)

plot(auto_Q_mean[1,:], label="Q", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Clustered Trajectory")
plot!(auto_gen_mean[1,:], label="gen")
plot!(auto_obs[1,:], label="obs")



##
#################### NN SAVINGS ####################

BSON.@save pwd() * "/NNs/nn_clustered_D$(dim)_$(normalization)_$(σ_value).bson" nn_clustered_cpu
BSON.@save pwd() * "/NNs/nn_vanilla_D$(dim)_$(normalization)_$(σ_value).bson" nn_vanilla_cpu

##
#################### NN LOADINGS ####################

BSON.load(pwd() * "/NNs/nn_clustered_D$(dim)_$(normalization)_$(σ_value).bson")[:nn_clustered_cpu]
BSON.load(pwd() * "/NNs/nn_vanilla_D$(dim)_$(normalization)_$(σ_value).bson")[:nn_vanilla_cpu]

##
#################### DRAFT PLOTS ####################

plt1 = vectorfield2d(vf_c, grid, xlabel="X", ylabel="Y", title="Learned Force Field")
plt2 = heatmap(kde_clustered.x, kde_clustered.y, kde_clustered.density, xlabel="X", ylabel="Y", title="Learned PDF", clims=(0, 0.5), colormap=:viridis)
plt3 = vectorfield2d(vf, grid, xlabel="X", ylabel="Y", title="Observed Force Field")
plt4 = heatmap(ekde_obs.x, kde_obs.y, kde_obs.density, xlabel="X", ylabel="Y", title="Observed PDF", clims=(0, 0.5), colormap=:viridis)
plot(plt1, plt2, plt3, plt4, layout=(2, 2), size=(800, 800))
#savefig(pwd() * "/figures/2D_pot_fig2.png")

##
scatter(inputs[1,:], inputs[2,:], marker_z=targets_norm, label="", xlabel="X", ylabel="Y", title="Vector Field Norm", size=(600, 600), colormap=:viridis, markersize=7)
savefig(pwd() * "/figures/2D_pot_fig1.png")


##



f(t; a=0.1) = (0 ≤ t ≤ a) ? 10 : 0

# function f(t; a=1/3)
#     t_max = 1 / a  # Time at which f(t) reaches 1
#     t_end = 2 * t_max  # Time at which f(t) returns to 0

#     if 0 ≤ t ≤ t_max
#         return a * t
#     elseif t_max < t ≤ t_end
#         return 1 - a * (t - t_max)
#     else
#         return 0  # f(t) is defined only in [0, 2/a], extend if needed
#     end
# end

# f(t) = 0.5

u(x) = [0.1; 0.0]
div_u(x) = 0.0
m = mean(obs, dims=2)
Obs(x) = (x[1:1,:] .- m[1]) .^ 2 + (x[2:2,:] .- m[2]) .^ 2 
Obs(x) = x[1:1,:] + x[2:2,:]

∇U_2D_pert(x,t) = ∇U_2D(x,t) + u(x) * f(t)

invC0 = inv(cov(obs'))
score_true(x) = ∇U_2D(x,0)
score_lin2(x) = - invC0*(x-m)
score_gen(x) = score_clustered(x)

trj = obs[:,1:100000]

n_tau = 100
δObs_num = generate_numerical_response(∇U_2D, ∇U_2D_pert, dim, dt, n_tau, 1000, sigma, Obs, 1; n_ens=10000)
R_true, δObs_true = generate_score_response(trj, u, div_u, f, score_true, dt, n_tau, Obs, 1)
R_lin, δObs_lin = generate_score_response(trj, u, div_u, f, score_lin2, dt, n_tau, Obs, 1)
R_gen, δObs_gen = generate_score_response(trj, u, div_u, f, score_gen, dt, n_tau, Obs, 1)

plot(dt:dt:n_tau*dt, δObs_num[1,:], xlabel="Lag", ylabel="Response", title="Responses", label="Numerical")
plot!(dt:dt:n_tau*dt, δObs_true[1,:], xlabel="Lag", ylabel="Response", label="True")
plot!(dt:dt:n_tau*dt, δObs_lin[1,:], xlabel="Lag", ylabel="Response", label="Linear")
plot!(dt:dt:n_tau*dt, δObs_gen[1,:], xlabel="Lag", ylabel="Response", label="Generative")