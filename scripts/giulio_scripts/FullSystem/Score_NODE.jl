#TRAIN NODE WITH Residual
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

#====================ESTIMATE DECORRELATION TIME=====================#
function estimate_tau(y, dt; threshold=0.2)
    y_centered = y .- mean(y)
    acf = autocovariance(y_centered, timesteps=500)

    tau_found = false
    tau_value = 0.0

    for i in 2:length(acf)
        if abs(acf[i]) < threshold
            tau_found = true
            tau_value = i * dt
            break
        end
    end

    if tau_found
        return tau_value, acf
    else
        println("Warning: No decorrelation time found within the specified threshold.")
        return dt * length(acf), acf
    end
end




#==================== DELAY EMBEDDING ====================#
function delay_embedding(x; τ, m)
    q = round(Int, τ / dt)
    start_idx = 1 + (m - 1) * q
    Z = [ [x[i - j*q] for j in 0:m-1] for i in start_idx:length(x) ]
    Z = hcat(Z...)  # shape (m, T)
    return reverse(Z, dims=1)  # inverti l'ordine delle righe
end

#================= BATCH GENERATION ===================#

function gen_batches(x::Matrix, batch_len::Int, n_batch::Int)
    datasize = size(x, 2)
    r = rand(1:datasize - batch_len, n_batch)
    return [x[:, i+1:i+batch_len] for i in r]
end

#==================== NN MODEL ====================#
function create_nn(layers::Vector{Int}; activation_hidden=swish, activation_output=identity)
    layer_list = []
    for i in 1:(length(layers) - 2)
        push!(layer_list, Dense(layers[i], layers[i+1], activation_hidden))
    end
    push!(layer_list, Dense(layers[end-1], layers[end], activation_output))
    return Chain(layer_list...)
end

#==================== NODE ROLLOUT ====================#
function dudt_full!(du, u, p, t)
    score_vec = [score_clustered_xt(u_i, Φ) for u_i in u]
    residual = re(p)(u)                   # Vector{Float32} of length m
    du .= score_vec .+ residual           # m-dimensional ODE
end


#==================== PREDICTION FUNCTION ====================#
function predict_full_dynamics(u0, p, tspan, t)
    prob = ODEProblem(dudt_full!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=t, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    return hcat(sol.u...)
end

#==================== LOSS ====================#
function loss_neuralode(p)
    loss = 0.0f0
    for i in 1:100
        u = data_sample[rand(1:length(data_sample))]
        pred = predict_full_dynamics(u[:, 1], p, tspan, t)
        loss += sum(abs2, (u[:, 1:end-1] .- pred[:, 1:end]))
    end
    return loss / 100
end

#========== MAIN ==========#

# ------------------------
# 1. Model definition
# ------------------------

m = 16  # delay embedding dim
layers = [m, 256, 256, m]
activation_hidden = swish
activation_output = identity
model = create_nn(layers; activation_hidden=activation_hidden, activation_output=activation_output)

#extract parameters from the model
flat_p0, re = Flux.destructure(model)

# ------------------------
# 2. Data generation
# ------------------------
# Parameters
fix_initial_state = false
σ_value=0.08
ε=0.5
save_figs = false
dim = 4 # Number of dimensions in the system
dt = 0.001f0
n_steps = 100
t = collect(0.0f0:dt:dt*(n_steps - 1))
tspan = (t[1], t[end])
Nsteps = 10000000
t_full = collect(0:dt:(Nsteps-1)*dt)
f(x, t) = F(x, t, σ_value, ε)
obs_nn = evolve(randn(4), dt, Nsteps, f, sigma; timestepper=:rk4, resolution=1)
M = mean(obs_nn, dims=2)[1]
S = std(obs_nn, dims=2)[1]
obs = (obs_nn[1:1,:] .- M) ./ S
obs_signal = obs[1, :]
plotlyjs()
plot(obs_signal[1:1000000], title="Signal", xlabel="Time", ylabel="Amplitude", label="x(t)", lw=2, color=:blue)
autocov_obs_nn = zeros(4, 100)#

for i in 1:4
    autocov_obs_nn[i, :] = autocovariance(obs_nn[i, :]; timesteps=100)
end

D_eff = dt * (0.5 * autocov_obs_nn[3, 1] + sum(autocov_obs_nn[3, 2:end-1]) + 0.5 * autocov_obs_nn[3, end])
D_eff = 0.3
@show D_eff

#training and clustering parameters 
σ_value=0.05
prob=0.001
conv_param=0.02
n_epochs=5000
batch_size=16


########## 3. Clustering ##########
averages, centers, Nc, labels = f_tilde_labels(σ_value, obs[:,1:100:end]; prob=prob, do_print=false, conv_param=conv_param, normalization=false)
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

########## Phi calculation ##########
#rate matrix
Q = generator(labels; dt=dt)*0.01
P_steady = steady_state(Q)
#test if Q approximates well the dynamics
tsteps = 101
res = 100

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
using Zygote
score_clustered_safe(x) = Zygote.ignore() do
    score_clustered(x)
end

function score_clustered_xt(x, Φ)
    return Float32(Φ * score_clustered_safe(x)[1])
end



#==================== EMBEDDING SU x ====================#
dt=0.001
res=100
tau_opt, _ = estimate_tau(obs_signal[1:100:end], dt*res)
@info "Scelta ottimale di τ ≈ $(round(tau_opt, digits=4))"
auto_obs = autocovariance(obs_signal[1:100:end], timesteps=1000)
plotlyjs()
plot(auto_obs, title="Autocovariance of x(t)", xlabel="Lag", ylabel="ACF", label="ACF of x(t)", lw=2, color=:blue)
tau = 0.33 * tau_opt
Z = Float32.(delay_embedding(obs_signal; τ=tau, m=m))  # delay embedding sulla x
size(Z,2)
size(Z,1)
Z[:,1]
obs_signal[1]
#==================== GENERAZIONE BATCH x ====================#
batch_size = n_steps + 1
n_batches = 2000
data_sample = gen_batches(Z, batch_size, n_batches)

#==================== TRAINING ====================#
p = flat_p0
opt = Optimisers.Adam(0.01)
state = Optimisers.setup(opt, p)
n_epochs = 500
losses = []

using BSON: @save
save_every = 100 
u = data_sample[rand(1:end)]

# u[1,1]
# dt=0.0001
# dt
# M_y = mean(obs_nn[3,:])
# S_y = std(obs_nn[3,:])
# obs_y2 = (obs_nn[3, :] .- M_y) ./ S_y
# plot(obs_y2[1:1000000], title="Signal y2(t)", xlabel="Time", ylabel="Amplitude", label="y2(t)", lw=2, color=:blue)
# sigma_Langevin(x, t) = Σ/sqrt(2)
# n_steps = 1000000
# prova=evolve_chaos([0.0], dt, n_steps, score_clustered_xt, sigma_Langevin, obs_y2; timestepper=:euler, resolution=1)
# plot(prova[1, 1:1000000], title="Langevin y2(t)", xlabel="Time", ylabel="Amplitude", label="y2(t)", lw=2, color=:blue)

for epoch in ProgressBar(1:n_epochs)
    loss_val, back = Flux.withgradient(p) do p
        loss_neuralode(p)
    end
    state, p = Optimisers.update(state, p, back[1])
    push!(losses, loss_val)
    if epoch % save_every == 0
        @save "model_epoch_$(epoch).bson" p
    end
end

println("Final Loss: ", losses[end])
plotlyjs()
plt_loss = plot(losses, xlabel="Epoch", ylabel="Loss", label="Training loss")
plot(plt_loss, losses, title="Loss vs Epoch", xlabel="Epoch", ylabel="Loss", label="Training loss", lw=2)


# ------------------------
# 7. Plot predictions: NODE on x(t)
# ------------------------

@load "/Users/giuliodelfelice/Desktop/MIT/ClustGen/model_epoch_500.bson" p
model_trained = re(p)

# Predict full delay embedding trajectory from Langevin effective model
u0 = Z[:, 1]
n_long = 1000000

function dudt_full!(du, u)
    score_vec = [score_clustered_xt(u_i, Φ) for u_i in u]
    stochastic_vec = model_trained(u)  # also of length m
    du .= score_vec .+ stochastic_vec
end

t_long = collect(0.0f0:dt:dt*(n_long - 1))
tspan_long = (t_long[1], t_long[end])

prob = ODEProblem(dudt_full!, u0, tspan_long, p)
sol = solve(prob, Tsit5(), saveat=t_long)
x_pred_delay = hcat(sol.u...)  # shape (m, T)

# Extract x(t) = first component of delay embedding
x_pred_long = x_pred_delay[1, :]
x_true_long = obs_signal[1:length(x_pred_long)]

# PDF + ACF
kde_node = kde(x_pred_long)
kde_true = kde(x_true_long)

acf_node = autocovariance(x_pred_long, timesteps=1000)
acf_true = autocovariance(x_true_long, timesteps=1000)

# PDF teorica
function true_pdf_normalized(x)
    x_phys = x .* S[1] .+ M[1]
    U = .-0.5 .* x_phys.^2 .+ 0.25 .* x_phys.^4
    p = exp.(-2 .* U ./ D_eff)
    return p ./ S[1]
end

xax = [-1.25:0.005:1.25...]
xax_2 = [-1.6:0.02:1.6...]
pdf_true = true_pdf_normalized(xax_2)
scale_factor = maximum(kde_true.density) / maximum(pdf_true)
pdf_true .*= scale_factor

# === PLOT ===

plotlyjs()
# PDF
p_pdf = plot(kde_true.x, kde_true.density, label="Observed", lw=2, color=:red)
plot!(p_pdf, kde_node.x, kde_node.density, label="NODE", lw=2, color=:blue)
plot!(p_pdf, xax_2, pdf_true; label="PDF analytic", linewidth=2, linestyle=:dash, color=:lime)
xlabel!("x"); ylabel!("Density"); title!("PDF comparison")

# ACF
p_acf = plot(acf_true, label="Observed", lw=2, color=:red)
plot!(p_acf, acf_node, label="NODE", lw=2, color=:blue, title="Autocovariance function", xlabel="Lag", ylabel="ACF")

# Trajectory
t_plot = t_long[1:length(x_pred_long)]
p_traj = plot(t_plot, x_true_long, label="True x(t)", lw=2, color=:red)
plot!(p_traj, t_plot, x_pred_long, label="NODE x(t)", lw=2, color=:blue, ls=:dash, title="Trajectories")

# Style
Plots.default(fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10, legendfontsize=10)
display(p_pdf)
display(p_acf)
display(p_traj)












sigma_Langevin(x, t) = Σ 


# Simulate Langevin dynamics
Nsamples = 100000000
t = collect(0:dt:dt*(Nsamples-1))  # esplicitamente un vettore
length(t)
tspan = (t[1], t[end])
u0 = [randn()]

traj_langevin = evolve(u0, dt, Nsteps, score_clustered_xt, sigma_Langevin; timestepper=:euler, resolution=10)
size(traj_langevin)
length(traj_langevin[1,:])
kde_langevin = kde(traj_langevin[1,:])
auto_langevin = autocovariance(traj_langevin[1,:]; timesteps=tsteps)
# plot(kde_y2_test.x, kde_y2_test.density, label="PDF of Langevin y2(t)", xlabel="y2", ylabel="Density", title="Distribution of Langevin y2(t)", linewidth=2)

# plot(traj_langevin, label="Langevin x", xlabel="Time", ylabel="x", title="Langevin x time series")


function normalize_f(f, x, t, M, S)
    return f(x .* S .+ M, t) .* S
end


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

########## 7. Plotting ##########
Plots.default(fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10, legendfontsize=10)
plotlyjs()




# #Plot PDF
# p_pdf = plot(kde_obs.x, kde_obs.density, label="Observed", lw=2, color=:red)
# plot!(p_pdf, kde_langevin.x, kde_langevin.density, label="Langevin", lw=2, color=:blue)
# xlabel!("x"); ylabel!("Density"); title!("PDF comparison")
# plot!(p_pdf, xax_2, pdf_true; label="PDF analytic", linewidth=2, linestyle=:dash, color=:lime)
# # plot!(p_pdf, xax_2, pdf_interpolated_norm; label="PDF learned", linewidth=2,color=:cyan)

#Plot Score
p_score = scatter(centers_sorted, scores; color=:blue, alpha=0.2, label="Cluster centers",
    xlims=(-1.3, 1.3), ylims=(-5, 5), xlabel="𝑥", ylabel="Score(𝑥)", title="Score Function Estimate")
plot!(p_score, xax, interpolated_score; label="NN interpolation", linewidth=2, color=:red)
plot!(p_score, xax, true_score; label="Score analytic", linewidth=2, color=:lime)

display(p_score)
display(p_pdf) 