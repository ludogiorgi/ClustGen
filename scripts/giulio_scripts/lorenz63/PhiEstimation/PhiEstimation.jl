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
    sol = solve(prob, Tsit5(), saveat=t, maxiters=2*10^7)
    return hcat(sol.u...)
end

function delay_embedding(x; τ, m)
    q = round(Int, τ / dt)
    start_idx = 1 + (m - 1) * q
    Z = [ [x[i - j*q] for j in 0:m-1] for i in start_idx:length(x) ]
    return hcat(Z...)
end


function F2(x, t, m, score, sigma, model)
    dy = zeros(m+1)
    dy[1:1] .= score(x[1:1], t) .+ x[2:2] .* sigma
    dy[2:m+1] .= model(x[2:end])
    return dy
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
Nsteps = 10000000
f(x, t) = F(x, t, σ, ε)
init = vcat(0.0, randn(3))
obs_nn = evolve(init, dt, Nsteps, f, sigma; timestepper=:rk4, resolution=10)
#obs_uncorr = obs_nn[1:1, 1:1:end]
@show size(obs_nn)
########## 2. Normalize and autocovariance ##########
M = mean(obs_nn, dims=2)[1]
S = std(obs_nn, dims=2)[1]
obs = (obs_nn[1:1,:] .- M) ./ S
plot(obs[1,1:10000])
#check autocov and pdf of simulated data
autocov_obs = autocovariance(obs[1, 1:100000]; timesteps=500)
kde_obs = kde(obs[1, :])

autocov_obs_nn = zeros(4, 100)

for i in 1:4
    autocov_obs_nn[i, :] = autocovariance(obs_nn[i, :]; timesteps=100)
end

D_eff = dt * (0.5 * autocov_obs_nn[3, 1] + sum(autocov_obs_nn[3, 2:end-1]) + 0.5 * autocov_obs_nn[3, end])
D_eff = 0.3
@show D_eff


#We want to compute the delay embedding of y2 to use Z[:,1] as initial condition
y = Float32.(obs_nn[3, :])
μ, σ = mean(y), std(y)
y_norm = (y .- μ) ./ σ
@show size(y_norm)

autocov_y_norm = autocovariance(y_norm[1:10:end]; timesteps=100)

τ = 0.25* 0.061 # delay time for embedding
m = 10 # embedding dimension

Z = Float32.(delay_embedding(y_norm; τ=τ, m=m))

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
Q = generator(labels; dt=dt)*0.20
P_steady = steady_state(Q)

#test if Q approximates well the dynamics
tsteps = 51
res = 10

auto_obs = autocovariance(obs[1:res:end]; timesteps=tsteps) 
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
score_clustered_xt(x, t) = Φ * score_clustered(x)

#load parameters of NODE model
@load "/Users/giuliodelfelice/Desktop/MIT/ClustGen/model_epoch_500.bson" p



#reconstruct nn with trained parameters
function create_nn(neurons::Vector{Int}; activation=swish, last_activation=identity)
    layers = Vector{Any}(undef, length(neurons) - 1)
    
    # Create hidden layers with specified activation
    for i in 1:length(neurons)-2
        layers[i] = Flux.Dense(neurons[i], neurons[i+1], activation)
    end
    
    # Create output layer with specified activation
    layers[end] = Flux.Dense(neurons[end-1], neurons[end], last_activation)
    
    return Flux.Chain(layers...)
end

layers = [10, 256, 256, 10]
activation = swish
last_activation = identity
model = create_nn(layers; activation=activation, last_activation=last_activation)

#deconstruct model and upload parameters
flat_p, re = Flux.destructure(model)
model_trained = re(p)

# #compute y2 trajectory
dt = 0.001f0
n_steps = 10000000
t = collect(0.0f0:dt:dt*(n_steps-1))
tspan = (t[1], t[end])
u0 = Z[:, rand(1:end)]#initial condition for Langevin

#compute trajectory of y2 chaotic forcing with NODE model

traj_y2 = predict_with_model(Float32.(u0), model_trained, tspan, t)
Y2_series = traj_y2[1,:] #first component of delay embedding
length(Y2_series)
plot(Y2_series)  # primi 100k punti
kde_y = kde(Y2_series)
plt_compare = plot(kde_y.x, kde_y.density, label="PDF of y2(t)", xlabel="y2", ylabel="Density", title="Distribution of y2(t)", linewidth=2)

#normalize y2 and compute its autocovariance
μ, σ = mean(Y2_series), std(Y2_series)
y2_norm = (Y2_series.- μ) ./ σ
length(y2_norm)
ked_y_norm = kde(y2_norm)
kde_observed_y = kde(y_norm)
plotlyjs()
plt_compare = plot(kde_observed_y.x, kde_observed_y.density, label="Normalized PDF of y2(t) from observations", xlabel="y2", ylabel="Density", title="Normalized Distribution of y2(t)", linewidth=2)
plot!(plt_compare, ked_y_norm.x, ked_y_norm.density, label="Normalized PDF of y2(t)", xlabel="y2", ylabel="Density", title="Normalized Distribution of y2(t)", linewidth=2)

autocov_y2 = autocovariance(y2_norm[1:100:end]; timesteps=100) 
plt_autocov = plot(autocov_y_norm, label="y2 autocovariance", xlabel="Lag", ylabel="Autocovariance", title="ACF of y2")
plot!(plt_autocov, autocov_y2, label="y2 NODE autocovariance", xlabel="Lag", ylabel="Autocovariance", title="ACF of y2")

τ_y2 = 1

# Diffusion coefficient √(2Φ)
sigma_Langevin(x,t) = Σ / sqrt(2 * τ_y2) 

# Langevin dynamics
n_steps = 1000000
dt = 0.01
traj_langevin = evolve_chaos([0.0], dt, n_steps,score_clustered_xt, sigma_Langevin, y2_norm[1:10:end]; timestepper=:euler, resolution=10)




M_langevin = mean(traj_langevin[1, :])[1]
S_langevin = std(traj_langevin[1, :])[1]
traj_langevin_norm = (traj_langevin[1:1, :] .- M_langevin) ./ S_langevin

kde_langevin = kde(traj_langevin_norm[1,:])
autocov_langevin = autocovariance(traj_langevin_norm[1,1:100000]; timesteps=500)

Plots.default(fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10, legendfontsize=10)
plotlyjs()
#PDF comparison
PDF_plot = plot(kde_langevin.x, kde_langevin.density, label="PDF Langevin", xlabel="x", ylabel="Density", title="Langevin PDF of x(t)", linewidth=2)
plot!(PDF_plot, kde_obs.x, kde_obs.density, label="PDF observed", xlabel="y2", ylabel="Density", title="Comparison of PDFs of x(t)", linewidth=2)

#Autocovariance comparison
autocov_plot = plot(autocov_langevin, label="Langevin autocovariance", xlabel="Lag", ylabel="Autocovariance", title="Comparison of ACFs of x(t)", linewidth=2)
plot!(autocov_plot, autocov_obs, label="Observed autocovariance", xlabel="Lag", ylabel="Autocovariance", title="ACF of x(t)", linewidth=2)

#Plot Score
p_score = scatter(centers_sorted, scores; color=:blue, alpha=0.2, label="Cluster centers",
    xlims=(-1.3, 1.3), ylims=(-5, 5), xlabel="𝑥", ylabel="Score(𝑥)", title="Score Function Estimate")
plot!(p_score, xax, interpolated_score; label="NN interpolation", linewidth=2, color=:red)
plot!(p_score, xax, true_score; label="Score analytic", linewidth=2, color=:lime)

#plot trajectories for comparison
p_traj = plot(traj_langevin_norm[1:10000], label="Learned",  xlabel = "time", ylabel="x", title="Trajectories compared", linewidth=2)
plot!(p_traj, obs[1,1:10000], label = "observed", linewidth=2)

display(p_score)
display(PDF_plot)
display(autocov_plot)
display(p_traj)





########### ORA PROVIAMO A ESTRAPOLARE SEGNALE VELOCE DA QUELLO LENTO ###########

#MOVING average
using ImageFiltering
obs_signal = obs[1, :]
window_size = 299  # deve essere dispari

kernel = fill(1.0 / window_size, window_size)  # media uniforme
smoothed = imfilter(obs_signal, kernel, "reflect")  # media mobile centrata con padding

residual = obs_signal .- smoothed
mean_res = mean(residual)
std_res = std(residual)
res_norm = (residual .- mean_res) ./ std_res


#PLOT VERIFICA
plot(smoothed[1:10000], label="Smoothed signal", xlabel="Time", ylabel="Amplitude", title="Moving Average of Observed Signal", linewidth=2)
plot!(residual[1:10000], label="Residual", xlabel="Time", ylabel="Amplitude", title="Residual of Observed Signal", linewidth=2)


kde_res = kde(residual)
kde_res_norm = kde(res_norm)
plt_pdf_residual = plot(kde_res.x, kde_res.density, label="PDF Residual", xlabel="y2", ylabel="Density", title="PDF of Residual Signal", linewidth=2)
plot!(plt_pdf_residual, kde_res_norm.x, kde_res_norm.density, label="PDF Residual Normalized", xlabel="y2", ylabel="Density", title="PDF of Normalized Residual Signal", linewidth=2)


kde_smoothed = kde(smoothed)
plot(kde_smoothed.x, kde_smoothed.density, label="PDF Smoothed", xlabel="y2", ylabel="Density", title="PDF of Smoothed Signal", linewidth=2)










# #TEST: COMPARE TIME TO INTEGRATE NODE WITH evolve AND WITH predict_with_model
# dt = 0.001f0
# n_steps = 10000000
# u0 = rand(m) .* 0.01
# boundary = (-1.5, 1.5)

# function y2_dynamics(x, m, t, model)
#     dy = zeros(m)
#     dy[:] = model(x)
# end

# function sigma(x, t; noise = 0.0)
#   sigma = zeros(length(x))
#   return sigma
# end

# large_F(x, t) =  y2_dynamics(x, m, t, model_trained)
# evolve(u0, dt, n_steps, large_F, sigma; timestepper=:euler, resolution=1, boundary=boundary)

# u0 = Z[:, rand(1:end)]
# @time traj_y2 = predict_with_model(u0, model_trained, tspan, t)



# #NEW ATTEMPT TO INTEGRATE EVERYTHING WITH ONE FUNCTION
# sigma_Lang = Σ / sqrt(2 * τ_y2)

# init_cond = vcat(Float32(0.0), Z[:, rand(1:end)])


# dt = 0.0001f0
# n_steps = 10000000
# Large_F2(x, t) = F2(x, t, m, score_clustered_xt, sigma_Lang, model_trained)
# traj_langevin = evolve(init_cond, dt, n_steps, Large_F2, sigma; timestepper=:euler, resolution=1, boundary=false)



# M_langevin = mean(traj_langevin[1, :])[1]
# S_langevin = std(traj_langevin[1, :])[1]
# traj_langevin_norm = (traj_langevin[1:1, :] .- M_langevin) ./ S_langevin

# plot(traj_langevin_norm[1, 1:end], label="Langevin trajectory", xlabel="Time", ylabel="x(t)", title="Langevin trajectory of x(t)", linewidth=2)

# kde_langevin = kde(traj_langevin_norm[1,:])
# autocov_langevin = autocovariance(traj_langevin_norm[1,1:100000]; timesteps=500)

# plotlyjs()
# #PDF comparison
# PDF_plot = plot(kde_langevin.x, kde_langevin.density, label="PDF Langevin", xlabel="x", ylabel="Density", title="Langevin PDF of x(t)", linewidth=2)
# plot!(PDF_plot, kde_obs.x, kde_obs.density, label="PDF observed", xlabel="y2", ylabel="Density", title="Comparison of PDFs of x(t)", linewidth=2)

# #Autocovariance comparison
# autocov_plot = plot(autocov_langevin, label="Langevin autocovariance", xlabel="Lag", ylabel="Autocovariance", title="Comparison of ACFs of x(t)", linewidth=2)
# plot!(autocov_plot, autocov_obs, label="Observed autocovariance", xlabel="Lag", ylabel="Autocovariance", title="ACF of x(t)", linewidth=2)

# display(PDF_plot)
# display(autocov_plot)



#VERIFICA DISTRIBUZIONE OSCILLAZIONI RESIDUI VS GAUSSIANA
#y2_samples = res_norm #occhio a 200:end qui
# kde_y2 = kde(y2_samples)
# μ_y2 = mean(y2_samples)
# σ_y2 = std(y2_samples)
# gauss_y2(x) = pdf(Normal(μ_y2, σ_y2), x)
# xax_y2 = kde_y2.x
# dx = xax_y2[2] - xax_y2[1]
# pdf_kde = kde_y2.density ./ (sum(kde_y2.density) * dx)
# pdf_gaussian_y2 = [gauss_y2(x) for x in xax_y2]
# pdf_gaussian_y2 ./= sum(pdf_gaussian_y2) * dx

# plot(plt_pdf_residual, xax_y2, pdf_gaussian_y2, label="Gaussian", xlabel="y2", ylabel="Density", title="KDE of Residual Signal", linewidth=2)
# μ = sum(kde_res_norm.density .* kde_res_norm.x) * step(kde_res_norm.x)
# σ² = sum((kde_res_norm.x .- μ).^2 .* kde_res_norm.density) * step(kde_res_norm.x)





using DifferentialEquations
# using Interpolations

# # === Parametri ===
# dt = 0.1
# τ_y2 = 1.0  # oppure il valore che hai già definito
# tspan = (0.0, dt * (n_steps - 1))

# # === Forzante sottocampionata ===
# forcing = y2_norm[1:100:end] 
# length(forcing) # coerente con dt=0.1
# n_steps = length(forcing)
# @assert length(forcing) == n_steps

# # === Interpolazione del forzante y2(t) ===
# itp = interpolate(forcing, BSpline(Linear()))
# forcing_interp = extrapolate(itp, Flat())

# function forcing_at_time(t::Float64)
#     idx = t / dt + 1
#     return forcing_interp(idx)
# end

# # === Definizione dell’equazione di Langevin ===
# function langevin_rhs!(du, u, p, t)
#     x = u[1]
#     ξ = forcing_at_time(t)
#     du[1] = score_clustered_xt([x], t)[1] + sigma_Langevin([x], t)[1] * ξ
# end

# # === Integrazione ===
# u0 = [0.0]
# prob = ODEProblem(langevin_rhs!, u0, tspan)
# sol = solve(prob, Rosenbrock23(autodiff=false), dt=dt, saveat=dt)

# traj_langevin = hcat(sol.u...)  # shape: (1, n_steps)
