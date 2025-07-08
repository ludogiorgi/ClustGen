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
Nsteps = 10000000
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
@show sizeof(inputs_targets)
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
Q = generator(labels; dt=dt)*0.16
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
score_clustered_xt(x) = Φ * score_clustered(x)

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
u0 = Z[:, rand(1:end)] #initial condition for Langevin
# traj_y2 = predict_with_model(u0, model_trained, tspan, t)
# Y2_series = traj_y2 #first component of delay embedding
# plot(Y2_series)  # primi 100k punti
# kde_y = kde(Y2_series)
# plt_compare = plot(kde_y.x, kde_y.density, label="PDF of y2(t)", xlabel="y2", ylabel="Density", title="Distribution of y2(t)", linewidth=2)

# μ, σ = mean(Y2_series), std(Y2_series)
# y2_norm = (Y2_series.- μ) ./ σ
# ked_y_norm = kde(y2_norm)
# plot!(plt_compare, ked_y_norm.x, ked_y_norm.density, label="Normalized PDF of y2(t)", xlabel="y2", ylabel="Density", title="Normalized Distribution of y2(t)", linewidth=2)

# autocov_y2 = autocovariance(y2_norm[1:100:end]; timesteps=100) 
# plt_autocov = plot(autocov_y_norm, label="y2 autocovariance", xlabel="Lag", ylabel="Autocovariance", title="ACF of y2")
# plot!(plt_autocov, autocov_y2, label="y2 NODE autocovariance", xlabel="Lag", ylabel="Autocovariance", title="ACF of y2")


# #compute decorrelation time of y2 for the effective diffusion coefficient
# function estimate_tau(y, dt; threshold=0.2)
#     y_centered = y .- mean(y)
#     acf = autocor(y_centered)
#     for i in 2:length(acf)
#         if abs(acf[i]) < threshold
#             return (i - 1) * dt, acf
#         end
#     end
#     return (length(acf) - 1) * dt, acf
# end
# # Applichiamolo alla terza variabile (y₂)
# τ_y2, acf_y2 = estimate_tau(Y2_series[1:100:end], dt) #NB subsampling of Y2_series to assure coherent dt 
# @show τ_y2
τ_y2 = 1


# Diffusion coefficient √(2Φ)
sigma_Lang = Σ / sqrt(2 * τ_y2) 


# Simulate Langevin dynamics
Nsamples = 1000000
dt = 0.01
t = collect(0:dt:dt*(Nsamples-1))  # esplicitamente un vettore
length(t)
tspan = (t[1], t[end])
u0 = rand(m+1) .* 0.01

traj_langevin = integrate_full_system(u0, score_clustered_xt, sigma_Lang, model_trained; tspan=tspan, t=t, m=m) 
length(traj_langevin[1,:])
kde_y2_test = kde(traj_langevin[2, :])
autocov_test = autocovariance(traj_langevin[2, 1:end]; timesteps=1)
plot(kde_y2_test.x, kde_y2_test.density, label="PDF of Langevin y2(t)", xlabel="y2", ylabel="Density", title="Distribution of Langevin y2(t)", linewidth=2)

plot(traj_langevin[1, 1:end], label="Langevin x", xlabel="Time", ylabel="x", title="Langevin x time series")
# #compute y2 trajectory
dt = 0.001f0
n_steps = 10000000
t = collect(0.0f0:dt:dt*(n_steps-1))
tspan = (t[1], t[end])
u0 = Z[:, rand(1:end)]
y2_generated_from_NODE = predict_with_model(u0, model_trained, tspan, t)

dt = 0.1
x_series = traj_langevin[1, :]
y2_series = traj_langevin[2, :]
var_y2_from_model = var(y2_generated_from_NODE)
var_y2_from_integrated_ODE = var(y2_series)

@show var_y2_from_model, var_y2_from_integrated_ODE
cor_xy = cor(x_series, y2_series)
@show cor_xy


using Interpolations, DifferentialEquations, ProgressMeter
Nsamples = 100000
dt = 0.01
t = 0:dt:(dt*(Nsamples-1))
function integrate_x_with_interpolated_y2(x0, t, y2_series, score, Φ, Σ, τ_y2)
    # Interpolazione di y2(t)
    itp_y2 = LinearInterpolation(t, y2_series, extrapolation_bc=Line())

    # Drift per x(t)
    function dxdt!(du, u, p, t)
        du[1] = Φ * score(u[1])[1] + Σ / sqrt(2 * τ_y2) * itp_y2(t)
    end

    tspan = (t[1], t[end])
    prob = ODEProblem(dxdt!, [x0], tspan)

    # ProgressBar
    prog = Progress(length(t), 1, "Integrating x(t)")
    function progress_cb(int)
        ProgressMeter.next!(prog)
        return nothing
    end
    cb = DiscreteCallback((u, t, int)->true, (int)->progress_cb(int))

    sol = solve(prob, Tsit5(), saveat=t, callback=cb)
    return hcat(sol.u...)  # shape: (1, length(t))
end
Y2 = y2_generated_from_NODE 
y2_series = Y2[1, 1:100000]  # la prima componente del delay embedding
x0 = 0.0
traj_x = integrate_x_with_interpolated_y2(x0, t, y2_series, score_clustered, Φ, Σ, τ_y2)
plot(traj_x)

# PDF of Langevin trajectory
kde_langevin = kde(trj_langevin_m[1, :])
#kde_langevin_test = kde(trj_langevin_test[1, :])

# Autocovariance of Langevin trajectory vs observed
auto_langevin = autocovariance(trj_langevin_m[1, 1:res:end]; timesteps=tsteps)
#auto_langevin_test = autocovariance(trj_langevin_test[1, 1:res:end]; timesteps=tsteps)
########## 7. Plotting ##########
Plots.default(fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10, legendfontsize=10)
plotlyjs()



#Plot PDF
p_pdf = plot(kde_obs.x, kde_obs.density, label="Observed", lw=2, color=:red)
plot!(p_pdf, kde_langevin.x, kde_langevin.density, label="Langevin", lw=2, color=:blue)
xlabel!("x"); ylabel!("Density"); title!("PDF comparison")
plot!(p_pdf, kde_langevin_test.x, kde_langevin_test.density; label="Langevin test", linewidth=2, color=:orange)
plot!(p_pdf, xax_2, pdf_true; label="PDF analytic", linewidth=2, linestyle=:dash, color=:lime)
# plot!(p_pdf, xax_2, pdf_interpolated_norm; label="PDF learned", linewidth=2,color=:cyan)


#Plot autocovariance
p_acf = plot(auto_obs, label="Observed", lw=2, color=:red)
xlabel!("Lag"); ylabel!("Autocorrelation"); title!("Autocorrelation: NN vs Observed")
plot!(p_acf, auto_langevin, label="Effective Langvein dynamics", lw=2, color=:blue)
xlabel!("Time steps"); ylabel!("Autocorrelation"); title!("Autocorrelation comparison")
plot!(p_acf, auto_langevin_test, label="Effective Langvein dynamics test", lw=2, color=:orange)


#Plot Score
p_score = scatter(centers_sorted, scores; color=:blue, alpha=0.2, label="Cluster centers",
    xlims=(-1.3, 1.3), ylims=(-5, 5), xlabel="𝑥", ylabel="Score(𝑥)", title="Score Function Estimate")
plot!(p_score, xax, interpolated_score; label="NN interpolation", linewidth=2, color=:red)
plot!(p_score, xax, true_score; label="Score analytic", linewidth=2, color=:lime)

display(p_score)
display(p_pdf) 
display(p_acf)

#TESTS
######### 6b. Compare y2 distribution with Gaussian ##########
y2_samples = obs_nn[3, :] #occhio a 200:end qui
kde_y2 = kde(y_norm)
μ_y2 = mean(y_norm)
σ_y2 = std(y_norm)
gauss_y2(x) = pdf(Normal(μ_y2, σ_y2), x)
xax_y2 = kde_y2.x
dx = xax_y2[2] - xax_y2[1]
pdf_kde = kde_y2.density ./ (sum(kde_y2.density) * dx)
pdf_gaussian_y2 = [gauss_y2(x) for x in xax_y2]
pdf_gaussian_y2 ./= sum(pdf_gaussian_y2) * dx
ked_y_norm = kde(y2_norm)
#plot pdf pdf_gaussian_y2
p_y2 = plot(kde_y2.x, pdf_kde; label="PDF of y2(t)", xlabel="y2", ylabel="Density", title="Distribution of y2(t)", linewidth=2)
plot!(p_y2, xax_y2, pdf_gaussian_y2; label="Gaussian fit", linewidth=2)
plot!(p_y2, ked_y_norm.x, ked_y_norm.density; label="Normalized PDF of y2(t)", xlabel="y2", ylabel="Density", title="Normalized Distribution of y2(t)", linewidth=2)

#compare autocovariance y2 NODE and y2 real
autocov_y2 = autocovariance(y2_norm[1:100:end]; timesteps=100) 
plt_autocov = plot(autocov_y_norm, label="y2 autocovariance", xlabel="Lag", ylabel="Autocovariance", title="ACF of y2")
plot!(plt_autocov, autocov_y2, label="y2 NODE autocovariance", xlabel="Lag", ylabel="Autocovariance", title="ACF of y2")

#compare time series y2 NODE and y2 real

plt_y2 = plot(t[1:10000],  Y2_series[1:100:1000000]; label="y2 NODE", xlabel="Time", ylabel="y2", title="Time series of y2 NODE")
plot!(plt_y2, t[1:10000], y_norm[1:10000]; label="y2 real", xlabel="Time", ylabel="y2", title="Time series of y2 real")

display(plt_y2)
########## 8. Save ##########
if save_figs
    base_path = "/Users/giuliodelfelice/Desktop/MIT"
    test_folder = joinpath(base_path, "TESTS")
    mkpath(test_folder)
    savefig(p_score, joinpath(test_folder, "Interpolation.pdf"))
    savefig(p_pdf, joinpath(test_folder, "PDFs.pdf"))
    savefig(p_loss, joinpath(test_folder, "loss_plot.pdf"))
    savefig(p_acf, joinpath(test_folder, "y2_pdf.pdf"))

else
    display(p_score)
    display(p_pdf)
    display(p_acf)
end


















# ########### VERIFICA ###########
# tsteps = 100
# res = 10

# acf_real = autocovariance(obs[1:res:end]; timesteps=tsteps)
# acf_sim = autocovariance(trj_langevin[1, 1:res:end]; timesteps=tsteps)

# acf_real_norm = acf_real ./ acf_real[1]
# acf_sim_norm = acf_sim ./ acf_sim[1]

# using LsqFit

# function exp_model(t, p)
#     A, τ = p
#     return A .* exp.(-t ./ τ)
# end

# lags = collect(0:tsteps-1)
# fit_result = curve_fit(exp_model, lags, acf_sim_norm, [1.0, 1.0])  # init: A=1, τ=1
# params = fit_result.param
# A_fit, τ_sim = params
# @show A_fit, τ_sim

# τ_real, _ = estimate_tau(obs_nn[1, :], dt)
# @show τ_real

# println("τ_real  = ", τ_real)
# println("τ_sim   = ", τ_sim)
# println("Φ_eff_real = ", 1 / τ_real)
# println("Φ_eff_sim  = ", 1 / τ_sim)


# #Plot for verification
# plot(lags, acf_real_norm, label="Real", lw=2, color=:blue)
# plot!(lags, acf_sim_norm, label="Simulated", lw=2, color=:red)
# plot!(lags, exp_model(lags, params), label="Fit A·exp(-t/τ)", lw=2, linestyle=:dash, color=:black)
# xlabel!("Lag")
# ylabel!("Autocorrelation")
# title!("ACF: real vs simulated")


                      
########## 6b. Compare y2 distribution with Gaussian ##########
# y2_samples = obs_nn[3, 200:end] #occhio a 200:end qui
# kde_y2 = kde(y2_samples)
# μ_y2 = mean(y2_samples)
# σ_y2 = std(y2_samples)
# gauss_y2(x) = pdf(Normal(μ_y2, σ_y2), x)
# xax_y2 = kde_y2.x
# dx = xax_y2[2] - xax_y2[1]
# pdf_kde = kde_y2.density ./ (sum(kde_y2.density) * dx)
# pdf_gaussian_y2 = [gauss_y2(x) for x in xax_y2]
# pdf_gaussian_y2 ./= sum(pdf_gaussian_y2) * dx


########## 6c. Compute ACF from Langvein equation dx = s(x) +  \sqrt(2)ξ(t) without Phi ##########
# score_only_xt(x, t) = score_clustered(x)
# sigma_plain(x, t) = [sqrt(2.0);]  

# trj_score_only = evolve([0.0], dt, Nsamples, score_only_xt, sigma_plain;
#                         timestepper=:euler, resolution=1)