#MAIN SCRIPT Slow Dynamics with score function estimation using white noise as stovhastiv term

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
using BSON: @load, @save
using Plots
using LinearAlgebra
using ProgressBars
using Distributions
using QuadGK
using LaTeXStrings
using StatsBase
using DifferentialEquations
using OrdinaryDiffEq
using Random
using Statistics
using SciMLSensitivity
using Optimisers
using ProgressMeter

########### USEFUL FUNCTIONS ###########
function F(x, t, σ, ε ; µ=10.0, ρ=28.0, β=8/3)
    dx = x[1] * (1 - x[1]^2) + (σ / ε) * x[3]
    dy1 = µ/ε^2 * (x[3] - x[2])
    dy2 = 1/ε^2 * (x[2] * (ρ - x[4]) - x[3])
    dy3 = 1/ε^2 * (x[2] * x[3] - β * x[4])
    return [dx, dy1, dy2, dy3]
end

function F2(x, t, m, score, sigma, model)
    dy = zeros(m+1)
    dy[1:1] .= score(x[1:1]) .+ x[2:2] .* sigma
    dy[2:m+1] .= model(x[2:end])
    return dy
end

function sigma_extended(sigma; m=m)
    Σmat = zeros(Float32, m + 1, m + 1)
    Σmat[1, 2:end] .= sigma
    return Σmat
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


#function to predict y2 from the trained model in the file .bson
function predict_with_model(u0, model, tspan, t)
    function dudt!(du, u, _, t)
        du .= model(u)
    end
    prob = ODEProblem(dudt!, u0, tspan)
    sol = solve(prob, Tsit5(), saveat=t)
    return hcat(sol.u...)
end

function delay_embedding(x; τ, m)
    q = round(Int, τ / dt)
    start_idx = 1 + (m - 1) * q
    Z = [ [x[i - j*q] for j in 0:m-1] for i in start_idx:length(x) ]
    return hcat(Z...)
end

#integrate the full system  [\dot(x), \dot(\vec(y))]

function integrate_full_system(u0, score, sigma, model; tspan, t, m::Int)
    # Define ODE
    function dudt!(du, u, p, t)
        du .= F2(u, t, m, score, sigma, model)
    end

    # Create problem
    prob = ODEProblem(dudt!, u0, tspan)

    # Integrate with progress bar (safe version)
    sol = solve(prob, Tsit5(); saveat=t)
    


    return hcat(sol.u...)
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
#obs_uncorr = obs_nn[1:1, 1:1:end]
@show size(obs_nn)
########## 2. Normalize and autocovariance ##########
M = mean(obs_nn, dims=2)[1]
S = std(obs_nn, dims=2)[1]
obs = (obs_nn[1:1,:] .- M) ./ S
plotlyjs()
plot(obs[1, 1:100000], label="Normalized x", xlabel="Time", ylabel="x", title="Normalized x time series")
kde_obs_100000 = kde(obs[1, 1:100000]; bandwidth=0.05)
plot(kde_obs_100000.x, kde_obs_100000.density, label="PDF of x", xlabel="x", ylabel="Density", title="PDF of x", linewidth=2)
autocov_obs = autocovariance(obs[1, 1:100000]; timesteps=500)
kde_obs = kde(obs[1, :])

autocov_obs_nn = zeros(4, 100)

for i in 1:4
    autocov_obs_nn[i, :] = autocovariance(obs_nn[i, :]; timesteps=100)
end

D_eff = dt * (0.5 * autocov_obs_nn[3, 1] + sum(autocov_obs_nn[3, 2:end-1]) + 0.5 * autocov_obs_nn[3, end])
D_eff = 0.3
@show D_eff

# plt_12 = plot(autocov_obs_nn[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of x")


#training and clustering parameters 
σ_value=0.05
prob=0.001
conv_param=0.02
n_epochs=5000
batch_size=16


########## 3. Clustering ##########
averages, centers, Nc, labels = f_tilde_labels(σ_value, obs[:,1:10:end]; prob=prob, do_print=false, conv_param=conv_param, normalization=false)
inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)

########## 4. Score Functions ##########

#analytic score function
f1(x,t) = x .- x.^3
score_true(x, t) = normalize_f(f1, x, t, M, S)

#learned score function
#kde_x = kde(obs_nn[1, :])
centers_sorted_indices = sortperm(centers[1, :])
centers_sorted = centers[:, centers_sorted_indices][:]
scores = .- averages[:, centers_sorted_indices][:] ./ σ_value

########## 5. Train NN ##########
@time nn, losses = train(inputs_targets, n_epochs, batch_size, [1, 50, 25, 1];
    opt=Flux.Adam(0.001), activation=swish, last_activation=identity,
    use_gpu=false)

nn_clustered_cpu = nn |> cpu
score_clustered(x) = .- nn_clustered_cpu(reshape(Float32[x...], :, 1))[:] ./ σ_value
score_clustered([0.1])
########## 6. Compute PDF ##########
function true_pdf_normalized(x)
    x_phys = x .* S[1] .+ M[1]
    U = .-0.5 .* x_phys.^2 .+ 0.25 .* x_phys.^4
    p = exp.(-2 .* U ./ D_eff)
    return p ./ S[1]
end

xax = [-1.25:0.005:1.25...]
xax_2 = [-1.6:0.02:1.6...]
interpolated_score = [score_clustered(xax[i])[1] for i in eachindex(xax)]
true_score = [2 * score_true(xax[i], 0.0)[1] / D_eff for i in eachindex(xax)]
pdf_interpolated_norm = compute_density_from_score(xax_2, score_clustered)
pdf_true = true_pdf_normalized(xax_2)
scale_factor = maximum(kde_obs.density) / maximum(pdf_true)
pdf_true .*= scale_factor

########## Phi calculation ##########
dt = 0.1
#rate matrix
Q = generator(labels; dt=dt)*0.18
P_steady = steady_state(Q)

#test if Q approximates well the dynamics
tsteps = 500
res = 10

auto_obs = autocovariance(obs[1:end]; timesteps=tsteps) 
auto_Q = autocovariance(centers[1,:], Q, [0:dt*res:Int(res * (tsteps-1) * dt)...])


plt = Plots.plot(auto_obs)
plt = Plots.plot!(auto_Q)

#compute the score function
gradLogp = - averages ./ σ_value


#compute Phi and Σ
M_Q = centers * Q * (centers *Diagonal(P_steady))'
V_Q = gradLogp * (centers * Diagonal(P_steady))'
Φ = (M_Q * inv(V_Q))[1,1]
Σ = sqrt(Φ)


########## Test effective dynamics ##########
score_clustered_xt(x) = Φ * score_clustered(x)

score_clustered_t(x, t) = Φ * score_clustered(x)
sigma_Langevin(x, t) = Σ 


# Simulate Langevin dynamics
Nsamples = 100000000
dt = 0.01
t = collect(0:dt:dt*(Nsamples-1))  # esplicitamente un vettore
length(t)
tspan = (t[1], t[end])
u0 = [randn()]

traj_langevin = evolve(u0, dt, Nsteps, score_clustered_t, sigma_Langevin; timestepper=:euler, resolution=10)
size(traj_langevin)
length(traj_langevin[1,:])
kde_langevin = kde(traj_langevin[1,:])
auto_langevin = autocovariance(traj_langevin[1,:]; timesteps=tsteps)
# plot(kde_y2_test.x, kde_y2_test.density, label="PDF of Langevin y2(t)", xlabel="y2", ylabel="Density", title="Distribution of Langevin y2(t)", linewidth=2)

# plot(traj_langevin, label="Langevin x", xlabel="Time", ylabel="x", title="Langevin x time series")

########## 7. Plotting ##########
Plots.default(fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10, legendfontsize=10)
plotlyjs()



#Plot PDF
p_pdf = plot(kde_obs.x, kde_obs.density, label="Observed", lw=2, color=:red)
plot!(p_pdf, kde_langevin.x, kde_langevin.density, label="Langevin", lw=2, color=:blue)
xlabel!("x"); ylabel!("Density"); title!("PDF comparison")
plot!(p_pdf, xax_2, pdf_true; label="PDF analytic", linewidth=2, linestyle=:dash, color=:lime)
# plot!(p_pdf, xax_2, pdf_interpolated_norm; label="PDF learned", linewidth=2,color=:cyan)


#Plot autocovariance
p_acf = plot(auto_obs, label="Observed", lw=2, color=:red)
xlabel!("Lag"); ylabel!("Autocorrelation"); title!("Autocorrelation: NN vs Observed")
plot!(p_acf, auto_langevin, label="Effective Langvein dynamics", lw=2, color=:blue)
xlabel!("Time steps"); ylabel!("Autocorrelation"); title!("ACF comparison")



#Plot Score
p_score = scatter(centers_sorted, scores; color=:blue, alpha=0.2, label="Cluster centers",
    xlims=(-1.3, 1.3), ylims=(-5, 5), xlabel="𝑥", ylabel="Score(𝑥)", title="Score Function Estimate")
plot!(p_score, xax, interpolated_score; label="NN interpolation", linewidth=2, color=:red)
plot!(p_score, xax, true_score; label="Score analytic", linewidth=2, color=:lime)

#Plot trajectories
p_traj = plot(obs[1,1:1000], label = "Observed dynamics", lw=2, color=:red)
plot!(traj_langevin[1,1:1000], label = "Effective Langevin dynamics", lw=2, color=:blue)
xlabel!("Time (steps)"); ylabel!("x"); title!("Trajectories")

display(p_score)
display(p_pdf) 
display(p_acf)
display(p_traj)



n_ic = 3  # numero di condizioni iniziali da confrontare
steps_to_plot = 1000
res = 10
dt = 0.01
Nsteps = steps_to_plot * res

for i in 1:n_ic
    Random.seed!()  # Reset seed usando l’orologio → random diversi ogni volta
    start_idx = rand(1:size(obs, 2) - steps_to_plot)
    @show start_idx
    obs_i = obs[1, start_idx : start_idx + steps_to_plot - 1]
    u0 = [obs[1, start_idx]] 
    @show u0 # questa è la C.I. per entrambe le traiettorie

    # Integro la Langevin con la stessa C.I.
    traj_langevin_i = evolve(u0, dt, Nsteps, score_clustered_t, sigma_Langevin;
                             timestepper=:euler, resolution=res)

    # Plot dedicato per ogni traiettoria
    p_i = plot(obs_i, label="Observed", lw=2, color=:red,
               xlabel="Time (steps)", ylabel="x",
               title="Trajectory comparison – IC $i")
    plot!(p_i, traj_langevin_i[1, 1:steps_to_plot],
          label="Langevin", lw=2, color=:blue)

    display(p_i)
end










# # #compute y2 trajectory
# dt = 0.001f0
# n_steps = 10000000
# t = collect(0.0f0:dt:dt*(n_steps-1))
# tspan = (t[1], t[end])
# u0 = Z[:, rand(1:end)]
# y2_generated_from_NODE = predict_with_model(u0, model_trained, tspan, t)

# dt = 0.1
# x_series = traj_langevin[1, :]
# y2_series = traj_langevin[2, :]
# var_y2_from_model = var(y2_generated_from_NODE)
# var_y2_from_integrated_ODE = var(y2_series)

# @show var_y2_from_model, var_y2_from_integrated_ODE
# cor_xy = cor(x_series, y2_series)
# @show cor_xy


# using Interpolations, DifferentialEquations, ProgressMeter
# Nsamples = 100000
# dt = 0.01
# t = 0:dt:(dt*(Nsamples-1))
# function integrate_x_with_interpolated_y2(x0, t, y2_series, score, Φ, Σ, τ_y2)
#     # Interpolazione di y2(t)
#     itp_y2 = LinearInterpolation(t, y2_series, extrapolation_bc=Line())

#     # Drift per x(t)
#     function dxdt!(du, u, p, t)
#         du[1] = Φ * score(u[1])[1] + Σ / sqrt(2 * τ_y2) * itp_y2(t)
#     end

#     tspan = (t[1], t[end])
#     prob = ODEProblem(dxdt!, [x0], tspan)

#     # ProgressBar
#     prog = Progress(length(t), 1, "Integrating x(t)")
#     function progress_cb(int)
#         ProgressMeter.next!(prog)
#         return nothing
#     end
#     cb = DiscreteCallback((u, t, int)->true, (int)->progress_cb(int))

#     sol = solve(prob, Tsit5(), saveat=t, callback=cb)
#     return hcat(sol.u...)  # shape: (1, length(t))
# end
# Y2 = y2_generated_from_NODE 
# y2_series = Y2[1, 1:100000]  # la prima componente del delay embedding
# x0 = 0.0
# traj_x = integrate_x_with_interpolated_y2(x0, t, y2_series, score_clustered, Φ, Σ, τ_y2)
# plot(traj_x)

# # PDF of Langevin trajectory
# kde_langevin = kde(trj_langevin_m[1, :])
# #kde_langevin_test = kde(trj_langevin_test[1, :])

# # Autocovariance of Langevin trajectory vs observed
# auto_langevin = autocovariance(trj_langevin_m[1, 1:res:end]; timesteps=tsteps)
# #auto_langevin_test = autocovariance(trj_langevin_test[1, 1:res:end]; timesteps=tsteps)