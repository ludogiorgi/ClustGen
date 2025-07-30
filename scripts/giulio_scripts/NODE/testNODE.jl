using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using MarkovChainHammer
using ClustGen
using KernelDensity
using HDF5
using BSON
using BSON: @save
using Plots
using LinearAlgebra
using Distributions
using DifferentialEquations
using Flux
using OrdinaryDiffEq
using Random
using Statistics
using StatsBase
using ProgressBars
using Optimisers
using SciMLSensitivity
#==================== LORENZ SYSTEM ====================#
function F(x, t, ε ; µ=10.0, ρ=28.0, β=8/3)
    dy1 = µ/ε^2 * (x[2] - x[1])
    dy2 = 1/ε^2 * (x[1] * (ρ - x[3]) - x[2])
    dy3 = 1/ε^2 * (x[1] * x[2] - β * x[3])
    return [dy1, dy2, dy3]
end

function sigma(x, t; noise = 0.0)
    sigma1 = noise
    sigma2 = noise
    sigma3 = noise
    return [sigma1, sigma2, sigma3]
end
#====================ESTIMATE DECORRELATION TIME=====================#
function estimate_decorrelation_time(y::AbstractVector; maxlag=200, threshold=0.2)
    y_centered = y .- mean(y)
    c = autocor(y_centered)
    for i in 2:min(maxlag, length(c))
        if abs(c[i]) < threshold
            return i, c
        end
    end
    return maxlag, c
end


#==================== DELAY EMBEDDING ====================#
function delay_embedding(x; τ, m)
    q = round(Int, τ / dt)
    start_idx = 1 + (m - 1) * q
    Z = [ [x[i - j*q] for j in 0:m-1] for i in start_idx:length(x) ]
    return hcat(Z...)
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
#==================== NODE ROLLOUT (Euler) ====================#
function dudt!(du, u, p, t)
    du .= re(p)(u)
end

function predict_neuralode(u0, p, tspan, t)
    prob = ODEProblem(dudt!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=t, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    return hcat(sol.u...)
end
#==================== LOSS E TRAIN ====================#

function loss_neuralode(p)
    loss = 0.0f0
    for i in 1:100
        u = data_sample[rand(1:length(data_sample))]
        pred = predict_neuralode(u[:, 1], p, tspan, t)
        loss += sum(abs2, (u[:, 2:end] .- pred[:, 1:end])* weights)
    end
    return loss / 100
end

#========== MAIN ==========#

# ------------------------
# 1. Model definition
# ------------------------

m = 10  # delay embedding dim
layers = [m, 256, 256, m]
activation_hidden = swish
activation_output = identity
model = create_nn(layers; activation_hidden=activation_hidden, activation_output=activation_output)

#extract parameters from the model
flat_p0, re = Flux.destructure(model)

# ------------------------
# 2. Data generation (cosine)
# ------------------------

dt = 0.001f0
n_steps = 50
t = collect(0.0f0:dt:dt*(n_steps - 1))
tspan = (t[1], t[end])
Nsteps = 1000000
t_full = collect(0:dt:(Nsteps-1)*dt)
ε = 0.5
# ====== TEST COSINE ====== # 
# t0 = 2π * rand()
# signal = cos.(t_full .+ t0)
# τ = 0.01
# Z = Float32.(delay_embedding(signal; τ=τ, m=m))#
# ====== end test ======= #

f = (x, t) -> F(x, t, ε)
obs_nn = evolve(randn(3), dt, Nsteps, f, sigma; timestepper=:rk4, resolution=1)
# normalize y before embedding
y = Float32.(obs_nn[2, :])
μ, σ = mean(y), std(y)
y_norm = (y .- μ) ./ σ

# ========= ESTIMATE OPTIMAL Tau ========= #
function estimate_tau(y, dt; threshold=0.2)
    y_centered = y .- mean(y)
    acf = autocor(y_centered)
    for i in 2:length(acf)
        if abs(acf[i]) < threshold
            return i * dt, acf
        end
    end
    return dt * length(acf), acf
end

τ_opt, acf = estimate_tau(obs_nn[2, :], dt)
@info "Scelta ottimale di τ ≈ $(round(τ_opt, digits=4))"
τ = 0.25* τ_opt

Z = Float32.(delay_embedding(y_norm; τ=τ, m=m))

# ------------------------
# 3. Batching
# ------------------------

batch_size = n_steps + 1
n_batches = 2000
data_sample = gen_batches(Z, batch_size, n_batches)

# ------------------------
# 4. Training the model
# ------------------------
p = flat_p0
opt = Optimisers.Adam(0.01)
state = Optimisers.setup(opt, p)
n_epochs = 500
losses = []

weights = exp.(LinRange(0.0f0, -1.0f0, n_steps))
size(weights)

using BSON: @save
save_every = 100  # Salva ogni 100 epoche


for epoch in ProgressBar(1:n_epochs)
    u = data_sample[rand(1:end)]
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
display(plt_loss)
# ------------------------
# 7. Plot predictions
# ------------------------
using BSON: @load
@load "/Users/giuliodelfelice/Desktop/MIT/ClustGen/model_epoch_500.bson" p
model_trained = re(p)

# First 100 steps prediction vs truth
u0 = Z[:, 1]
t_short = collect(0.0f0:dt:dt*99)
tspan_short = (t_short[1], t_short[end])

function predict_with_model(u0, model, tspan, t)
    function dudt!(du, u, _, t)
        du .= model(u)
    end
    prob = ODEProblem(dudt!, u0, tspan)
    sol = solve(prob, Tsit5(), saveat=t)
    return hcat(sol.u...)
end

pred_short = predict_with_model(u0, model_trained, tspan_short, t_short)
y_pred_short = pred_short[1, :]
y_true_short = Z[1, 1:100]

plotlyjs()
plt1 = plot(t_short, y_true_short, label="True y2(t)", lw=2)
plot!(plt1, t_short, y_pred_short, label="Predicted y2(t)", lw=2, ls=:dash, title="Prediction: 100 steps")
display(plt1)

# First n_long steps prediction vs truth
n_long = 1000000
t_long = collect(0.0f0:dt:dt*(n_long - 1))
tspan_long = (t_long[1], t_long[end])
pred_long = predict_with_model(u0, model_trained, tspan_long, t_long)
max_steps = min(size(pred_long, 2), size(Z, 2))
y_pred_long = pred_long[1, 1:max_steps]
y_true_long = Z[1, 1:max_steps]
t_plot = t_long[1:max_steps]

M_y, S_y = mean(y_pred_long), std(y_pred_long)
y_pred_long_norm = (y_pred_long .- M_y) ./ S_y

#Plot of the time series
plotlyjs()
plt2 = plot(t_plot[1:end], y_true_long[1:end], label="Observed y2(t)", lw=2, color=:red)
plot!(plt2, t_plot[1:end], y_pred_long[1:end], label="NODE y(t)", lw=2, color=:blue, title="Trajectories compared")
display(plt2)
#plot of the PDFs
plotlyjs()
kde_pred = kde(y_pred_long_norm)
kde_obs = kde(y_true_long)
plot_kde = plot(kde_pred.x, kde_pred.density; label = "NODE", lw=2, color = :blue)
plot!(plot_kde, kde_obs.x, kde_obs.density; label = "observations", lw=2, color = :red, title="PDF compared")
μ = mean(y_true_long)
σ = std(y_true_long)
x_gauss = range(minimum(kde_obs.x), stop=maximum(kde_obs.x), length=500)
pdf_gauss = pdf.(Normal(μ, σ), x_gauss)

plot!(plot_kde, x_gauss, pdf_gauss; label = "Gaussian", lw=2, color = :lime)

display(plot_kde)

#plot acf
σ_noise = std(y_true_long .- y_pred_long)   # oppure ≈ 0.01 × std
y_pred_noisy = y_pred_long .+ 1.5*σ_noise * randn(length(y_pred_long))

acf_NODE = autocovariance(y_pred_noisy, timesteps=2000)
acf_true = autocovariance(y_true_long, timesteps=2000)
plot(acf_true, lw=2, color=:red, label= "Observed", title="ACF")
plot!(acf_NODE, lw=2, color=:blue, label="NODE", xlabel="Lag", ylabel="ACF")


#=============== END MAIN ===============#










using Pkg
Pkg.activate(".")
Pkg.instantiate()

using ClustGen
using Distributions
using DifferentialEquations
using Zygote
using CUDA
using SciMLSensitivity
using Flux
using OrdinaryDiffEq
using Random
using Statistics
using StatsBase
using ProgressBars
using Optimisers
using ComponentArrays

#==================== LORENZ SYSTEM ====================#
function F(x, t, ε ; µ=10.0, ρ=28.0, β=8/3)
    dy1 = µ/ε^2 * (x[2] - x[1])
    dy2 = 1/ε^2 * (x[1] * (ρ - x[3]) - x[2])
    dy3 = 1/ε^2 * (x[1] * x[2] - β * x[3])
    return [dy1, dy2, dy3]
end

function sigma(x, t; noise = 0.0)
    sigma1 = noise
    sigma2 = noise
    sigma3 = noise
    return [sigma1, sigma2, sigma3]
end
#====================ESTIMATE DECORRELATION TIME=====================#
function estimate_decorrelation_time(y::AbstractVector; maxlag=200, threshold=0.2)
    y_centered = y .- mean(y)
    c = autocor(y_centered)
    for i in 2:min(maxlag, length(c))
        if abs(c[i]) < threshold
            return i, c
        end
    end
    return maxlag, c
end


#==================== DELAY EMBEDDING ====================#
function delay_embedding(x; τ, m)
    q = round(Int, τ / dt)
    start_idx = 1 + (m - 1) * q
    Z = [ [x[i - j*q] for j in 0:m-1] for i in start_idx:length(x) ]
    return hcat(Z...)
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
#==================== NODE ROLLOUT (Euler) ====================#
function dudt!(du, u, p, t)
    du .= re(p)(u)
end

function predict_neuralode(u0, p, tspan, t)
    prob = ODEProblem(dudt!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=t, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    return hcat(sol.u...)
end
#==================== LOSS E TRAIN ====================#

function loss_neuralode(p)
    loss = 0.0f0
    for i in 1:100
        u = data_sample[rand(1:length(data_sample))]
        pred = predict_neuralode(u[:, 1], p, tspan, t)
        loss += sum(abs2, u[:, 2:end] .- pred[:, 1:end])
    end
    return loss / 100
end

#========== MAIN ==========#

# ------------------------
# 1. Model definition
# ------------------------

m = 10  # delay embedding dim
layers = [m, 256, 256, m]
activation_hidden = swish
activation_output = identity
model = create_nn(layers; activation_hidden=activation_hidden, activation_output=activation_output)

#extract parameters from the model
flat_p0, re = Flux.destructure(model)

# ------------------------
# 2. Data generation (cosine)
# ------------------------

dt = 0.001f0
n_steps = 40
t = collect(0.0f0:dt:dt*(n_steps - 1))
tspan = (t[1], t[end])
Nsteps = 1000000
t_full = collect(0:dt:(Nsteps-1)*dt)
ε = 0.5
# ====== TEST COSINE ====== # 
# t0 = 2π * rand()
# signal = cos.(t_full .+ t0)
# τ = 0.01
# Z = Float32.(delay_embedding(signal; τ=τ, m=m))#
# ====== end test ======= #

f = (x, t) -> F(x, t, ε)
obs_nn = evolve(randn(3), dt, Nsteps, f, sigma; timestepper=:rk4, resolution=1)
# normalize y before embedding
y = Float32.(obs_nn[2, :])
μ, σ = mean(y), std(y)
y_norm = (y .- μ) ./ σ

# ========= ESTIMATE OPTIMAL Tau ========= #
function estimate_tau(y, dt; threshold=0.2)
    y_centered = y .- mean(y)
    acf = autocor(y_centered)
    for i in 2:length(acf)
        if abs(acf[i]) < threshold
            return i * dt, acf
        end
    end
    return dt * length(acf), acf
end

τ_opt, acf = estimate_tau(obs_nn[2, :], dt)
@info "Scelta ottimale di τ ≈ $(round(τ_opt, digits=4))"
τ = 0.25* τ_opt

Z = Float32.(delay_embedding(y_norm; τ=τ, m=m))

# ------------------------
# 3. Batching
# ------------------------

batch_size = n_steps + 1
n_batches = 2000
data_sample = gen_batches(Z, batch_size, n_batches)

# ------------------------
# 4. Training the model
# ------------------------
p = flat_p0
opt = Optimisers.Adam(0.01)
state = Optimisers.setup(opt, p)
n_epochs = 500
losses = []

for epoch in ProgressBar(1:n_epochs)
    u = data_sample[rand(1:end)]
    loss_val, back = Flux.withgradient(p) do p
        loss_neuralode(p)
    end
    state, p = Optimisers.update(state, p, back[1])
    push!(losses, loss_val)
end

println("Final Loss: ", losses[end])
plotlyjs()
plt_loss = plot(losses, xlabel="Epoch", ylabel="Loss", label="Training loss")
plot(plt_loss, losses, title="Loss vs Epoch", xlabel="Epoch", ylabel="Loss", label="Training loss")
display(plt_loss)
# ------------------------
# 7. Plot predictions
# ------------------------

model_trained = re(p)

# First 100 steps prediction vs truth
u0 = Z[:, 1]
t_short = collect(0.0f0:dt:dt*99)
tspan_short = (t_short[1], t_short[end])

function predict_with_model(u0, model, tspan, t)
    function dudt!(du, u, _, t)
        du .= model(u)
    end
    prob = ODEProblem(dudt!, u0, tspan)
    sol = solve(prob, Tsit5(), saveat=t)
    return hcat(sol.u...)
end

pred_short = predict_with_model(u0, model_trained, tspan_short, t_short)
y_pred_short = pred_short[1, :]
y_true_short = Z[1, 1:100]

plotlyjs()
plt1 = plot(t_short, y_true_short, label="True y2(t)", lw=2)
plot!(plt1, t_short, y_pred_short, label="Predicted y2(t)", lw=2, ls=:dash, title="Prediction: 100 steps")
display(plt1)

# First n_long steps prediction vs truth
n_long = 1000000
t_long = collect(0.0f0:dt:dt*(n_long - 1))
tspan_long = (t_long[1], t_long[end])
pred_long = predict_with_model(u0, model_trained, tspan_long, t_long)
max_steps = min(size(pred_long, 2), size(Z, 2))
y_pred_long = pred_long[1, 1:max_steps]
y_true_long = Z[1, 1:max_steps]
t_plot = t_long[1:max_steps]

#Plot of the time series
plotlyjs()
plt2 = plot(t_plot, y_true_long, label="True y2(t)", lw=2)
plot!(plt2, t_plot, y_pred_long, label="Predicted y(t)", lw=2, ls=:dash, title="$n_long steps with m= $m, n_steps = $n_steps and dt = $dt")
display(plt2)
#plot of the PDFs
plotlyjs()
kde_pred = kde(y_pred_long)
kde_obs = kde(y_true_long)
plot_kde = plot(kde_pred.x, kde_pred.density; label = "prediction", color = :red)
plot!(plot_kde, kde_obs.x, kde_obs.density; label = "observations", color = :blue)
display(plot_kde)

#=============== END MAIN ===============#