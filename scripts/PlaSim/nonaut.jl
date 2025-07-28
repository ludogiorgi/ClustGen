using Pkg
Pkg.activate(".")
Pkg.instantiate()

##
using NetCDF
using Plots
using ImageMagick  # Optional, for GIF creation
using Statistics  # For mean function in PCA
using LinearAlgebra  # For eigen decomposition
using Revise
using MarkovChainHammer
using ClustGen
using KernelDensity
using HDF5
using Flux
using BSON
using ProgressBars

##

ω = π/5
a = 1.0
b = 0.1

function F(x, t; ω=ω, a=a)
    return cos(ω*t) .- a * x
end

function sigma(x, t; b=b)
    return [b]
end

function x_mean_ou_full(t; ω=ω, a=a, x0=0.0)
    num = a * cos.(ω .* t) .+ ω * sin.(ω .* t) .- a * exp.(-a .* t)
    den = a^2 + ω^2
    return x0 * exp.(-a .* t) .+ num ./ den
end

dt = 0.1

# Use consistent time points
Nsteps = 1000000
x_timeseries = evolve([0.0], 0.01, Nsteps*10, F, sigma; resolution = 10)
tarr = 0:dt:(size(x_timeseries, 2)-1)*dt  # Match actual obs size

plot(tarr[100:500], x_timeseries[1, 100:500], label="Trajectory")
plot!(tarr[100:500], x_mean_ou_full(tarr[100:500]), label="Mean", color=:red)
##
times = [0.0:dt:Nsteps*dt...]
times_mod = mod.(times, 10)
obs_nn = vcat(reshape(times_mod, 1, :), x_timeseries)

M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S

obs_uncorr = obs[:, 1:1:end]

Plots.scatter(obs_uncorr[1,1:10000], obs_uncorr[2,1:10000], markersize=1, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

normalization = false
σ_value = 0.05

averages, _, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.001, do_print=true, conv_param=0.001, normalization=normalization)

if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
end

gr()
targets_norm = [norm(averages[:,i]) for i in eachindex(centers[1,:])]
Plots.scatter(centers[1,:], centers[2,:], marker_z=targets_norm, color=:viridis)

##
#################### TRAINING WITH CLUSTERING LOSS ####################
dim = 2
@time nn_clustered, loss_clustered = train(inputs_targets, 2000, 16, [dim, 100, 50, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value
Plots.plot(loss_clustered)
##
plot(xax, F.(xax, 1))
plot!(xax, score_clustered_xt.(xax, 1))


##
#################### SAMPLES GENERATION ####################
dt = 0.1
tax = [dt:dt:50...]
plot(tax, obs[1,1:500])
plot!(tax, (mod.(tax, 10) .- M[1]) ./ S[1])


##
plotly()
score_clustered_xt(x,t) =  score_clustered([(mod.(t, 10) .- M[1]) ./ S[1], x[1]])[2:2]

# Surface plot for score_clustered_xt(x, t) over x, t in [-2, 2]
xax = -2:0.05:2
tarr = 0:0.1:20
Z = [score_clustered_xt([x], t)[1] for x in xax, t in tarr]

plt_surface = surface(xax, tarr, Z', xlabel="x", ylabel="t", zlabel="score_clustered_xt(x, t)", title="Surface of score_clustered_xt(x, t)")
display(plt_surface)

##
#labels = [ssp.embedding(obs[:,i]) for i in eachindex(obs[1,:])]
# averages_c, centers_c, Nc_c, labels_c = cleaning(averages, centers, labels)
Q = generator(labels_c;dt=dt)
P_steady = steady_state(Q)

tsteps = 61
res = 10

auto_obs = zeros(dim, tsteps)
auto_Q = zeros(dim, tsteps)

for i in 1:dim
    auto_obs[i,:] = autocovariance(obs[i,1:res:100000]; timesteps=tsteps) 
    auto_Q[i,:] = autocovariance(centers_c[i,:], Q, [0:dt*res:Int(res * (tsteps-1) * dt)...])
end

plt = Plots.plot(auto_obs[1,:])
plt = Plots.plot!(auto_Q[1,:])
##
dim
gradLogp = zeros(dim, Nc_c)
for i in 1:Nc_c
    gradLogp[:,i] = - averages[:,i] / σ_value
end

C0 = centers_c * (centers_c * Diagonal(P_steady))'
C1_Q = centers_c * Q * (centers_c * Diagonal(P_steady))'
C1_grad = gradLogp * (centers_c * Diagonal(P_steady))'
Σ_test2 = C1_Q * inv(C1_grad)
Σ_test = cholesky(0.5*(Σ_test2 .+ Σ_test2')).L[1,1]
println("Σ_test = ", Σ_test)

Σ_test = computeSigma(centers_c, P_steady, Q, gradLogp)

##

dt = 0.1
Nsteps = 100000
sigma_I(x,t) = 1.0
tax = [0.0:dt:Nsteps*dt...]
trj_clustered = evolve([0.0], 0.01, 10*Nsteps, score_clustered_xt, sigma_I; timestepper=:euler, resolution=10, boundary=[-15,15])

kde_xt = kde(trj_clustered')
##
plt1 = plot(tax[1:1000], trj_clustered[1,1:1000], label="Generated", color=:blue)
plt1 = plot!(tax[1:1000], obs[2,1:1000], label="Observed", color=:red)
plt1 = plot!(tax[1:1000], (x_mean_ou_full.(tax[1:1000]) .- M[2]) ./ S[2], label="Mean", color=:black)

##

res = 2
tsteps = 101
auto_clustered = zeros(dim, tsteps)
auto_obs = zeros(dim, tsteps)
kde_clustered_x, kde_clustered_y = [], []
kde_obs_x, kde_obs_y = [], []


for i in 1:1
    # kde_clustered_temp = kde(trj_clustered[i,:])
    # push!(kde_clustered_x, [kde_clustered_temp.x...])
    # push!(kde_clustered_y, kde_clustered_temp.density)
    # kde_obs_temp = kde(obs[i,:])
    # push!(kde_obs_x, [kde_obs_temp.x...])
    # push!(kde_obs_y, kde_obs_temp.density)
    auto_clustered[i,:] = autocovariance(trj_clustered[i,1:res:10000]; timesteps=tsteps) 
    auto_obs[i,:] = autocovariance(obs[i+1,1:res:10000]; timesteps=tsteps)
end

plot(auto_clustered[1,:], label="Clustered", color=:blue)
plot!(auto_obs[1,:], label="Observed", color=:red)