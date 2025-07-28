# MAIN SCRIPT — Score Function Training from Clustering (formerly run_experiments)

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
using Distributions
using QuadGK
using LaTeXStrings
using StatsBase


########### USEFUL FUNCTIONS ###########
function F(x, t, σ, ε ; µ=10.0, ρ=28.0, β=8/3)
    dx = x[1] * (1 - x[1]^2) + (σ / ε) * x[3]
    dy1 = µ/ε^2 * (x[3] - x[2])
    dy2 = 1/ε^2 * (x[2] * (ρ - x[4]) - x[3])
    dy3 = 1/ε^2 * (x[2] * x[3] - β * x[4])
    return [dx, dy1, dy2, dy3]
end

function sigma(x, t; noise = 0.0)
    sigma1 = noise
    sigma2 = noise
    sigma3 = noise
    sigma4 = noise #Added: This is for the 4th variable
    return [sigma1, sigma2, sigma3, sigma4]
end

function normalize_f(f, x, t, M, S)
    return f(x .* S .+ M, t) .* S
end
########### END USEFUL FUNCTIONS ###########

# Parameters
fix_initial_state = false
σ=0.08
ε=0.5
save_figs = false
dim = 4 # Number of dimensions in the system

########## 1. Simulate System ##########
dt = 0.01
Nsteps = 100000000
f(x, t) = F(x, t, σ, ε)
obs_nn = evolve(randn(4), dt, Nsteps, f, sigma; timestepper=:rk4, resolution=10)

########## 2. Normalize and autocovariance ##########
M = mean(obs_nn, dims=2)[1]
S = std(obs_nn, dims=2)[1]
obs = (obs_nn[1:1,:] .- M) ./ S

autocov_obs = autocovariance(obs[1, 1:100000]; timesteps=500)
kde_obs = kde(obs[1, :])

plt1 = plot(kde_obs.x, kde_obs.density, label="Observed", lw=2, color=:blue)
plt2 = plot(autocov_obs)
plot(plt1, plt2, layout=(2, 1), size=(800, 600),
     xlabel="x", ylabel="Density / Autocovariance",
     title=["KDE of Observed Data" "Autocovariance of Observed Data"],
     legend=:topright)

##
#training and clustering parameters 
σ_value=0.05
prob=0.001
conv_param=0.01
n_epochs=5000
batch_size=16

########## 3. Clustering ##########
averages, centers, Nc, labels = f_tilde_labels(σ_value, obs[:,1:10:end]; prob=prob, do_print=false, conv_param=conv_param, normalization=false)
inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)

########## 4. Score Functions ##########
centers_sorted_indices = sortperm(centers[1, :])
centers_sorted = centers[:, centers_sorted_indices][:]
scores = .- averages[:, centers_sorted_indices][:] ./ σ_value

scatter(centers_sorted[:], scores[:], label="Score Function", color=:blue, markersize=3,
    xlabel="x", ylabel="Score", title="Score Function vs Centers")

##
########## 5. Train NN ##########
@time nn, losses = train(inputs_targets, n_epochs, batch_size, [1, 50, 25, 1];
    opt=Flux.Adam(0.001), activation=swish, last_activation=identity,
    use_gpu=false)

nn_clustered_cpu = nn |> cpu
score_clustered(x) = .- nn_clustered_cpu(reshape(Float32[x...], :, 1))[:] ./ σ_value

s_gen = [score_clustered(c)[1] for c in centers_sorted]

plot(s_gen)
scatter!(scores)

##
########## Phi calculation ##########
dt= 0.1
#rate matrix
Q = generator(labels; dt=dt)*0.16
P_steady = steady_state(Q)

#test if Q approximates well the dynamics
tsteps = 51
res = 10

auto_obs = autocovariance(obs[1:res:end]; timesteps=tsteps) 
auto_Q = autocovariance(centers[1,:], Q, [0:dt*res:Int(res * (tsteps-1) * dt)...])

plt = Plots.plot(auto_obs)
plt = Plots.plot!(auto_Q)
##
#compute the score function
gradLogp = - averages ./ σ_value


#compute Phi and Σ
M_Q = centers * Q * (centers *Diagonal(P_steady))'
V_Q = gradLogp * (centers * Diagonal(P_steady))'
Φ = (M_Q * inv(V_Q))[1,1]
Σ = sqrt(Φ)


########## Test effective dynamics ##########
score_clustered_xt(x, t) = Φ * score_clustered(x)
sigma_Langevin(x, t) = Σ 

Nsamples = 10000000
dt = 0.001
trj_langevin = evolve([0.0], dt, Nsamples, score_clustered_xt, sigma_Langevin;
                      timestepper=:euler, resolution=100)
dt = 0.1

# PDF of Langevin trajectory
kde_langevin = kde(trj_langevin[1, :])

# Autocovariance of Langevin trajectory vs observed
auto_langevin = autocovariance(trj_langevin[1, 1:res:end]; timesteps=tsteps)

plt1 = Plots.plot(kde_obs.x, kde_obs.density, label="Observed", lw=2, color=:blue, title="PDF")
plt1 = Plots.plot!(kde_langevin.x, kde_langevin.density, label="Generated", lw=2, color=:red)

plt2 = plot(auto_langevin, label="", lw=2, color=:red, title="Autocovariance")
plt2 = plot!(auto_obs, label="", lw=2, color=:blue)
plt = plot(plt1, plt2, layout=(2, 1), size=(800, 600))
savefig(plt, "generated_vs_observed.pdf")
