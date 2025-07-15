using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using MarkovChainHammer
using ClustGen
using KernelDensity
using HDF5
using BSON
using Plots
using LinearAlgebra
using Distributions
using CUDA
using Flux
using Statistics
using StatsBase
using ProgressBars
using Optimisers
using PlotlyJS

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

#==================== MANUAL EULER ROLLOUT ====================#
function rollout_manual_euler(u0, model, dt, n_steps)
    u = u0
    traj = CUDA.zeros(Float32, size(u0, 1), n_steps + 1)
    traj[:, 1] .= u
    for i in 1:n_steps
        du = model(u)
        u = u .+ dt .* du
        traj[:, i + 1] .= u
    end
    return traj
end

#==================== GPU LOSS FUNCTION PARALLELIZED ====================#
function loss_neuralode_gpu_spawn(p)
    model = re(p) |> gpu
    futures = CUDA.@sync [CUDA.@spawn begin
        u = data_sample[i]
        pred = rollout_manual_euler(u[:, 1], model, dt, n_steps)
        sum(abs2, u[:, 2:end] .- pred[:, 2:end])
    end for i in 1:length(data_sample)]
    
    losses = fetch.(futures)
    return sum(losses) / length(losses)
end


#========== MAIN ==========#
m = 10  # delay embedding dim
layers = [m, 256, 256, m]
activation_hidden = swish
activation_output = identity
model = create_nn(layers; activation_hidden=activation_hidden, activation_output=activation_output) |> gpu
flat_p0, re = Flux.destructure(model)

dt = 0.001f0
n_steps = 40
t = collect(0.0f0:dt:dt*(n_steps - 1))
Nsteps = 1000000
t_full = collect(0:dt:(Nsteps-1)*dt)
ε = 0.5
f = (x, t) -> F(x, t, ε)
obs_nn = evolve(randn(3), dt, Nsteps, f, sigma; timestepper=:rk4, resolution=1)

y = Float32.(obs_nn[2, :])
μ, σ = mean(y), std(y)
y_norm = (y .- μ) ./ σ

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
τ = 0.15 * τ_opt
Z = Float32.(delay_embedding(y_norm; τ=τ, m=m))

batch_size = n_steps + 1
n_batches = 2000
data_sample = gen_batches(Z, batch_size, n_batches)
data_sample = [cu(x) for x in data_sample]

p = flat_p0
opt = Optimisers.Adam(0.01)
state = Optimisers.setup(opt, p)
n_epochs = 500
losses = []

for epoch in ProgressBar(1:n_epochs)
    loss_val, back = Flux.withgradient(p) do p
        loss_neuralode_gpu_spawn(p)
    end
    state, p = Optimisers.update(state, p, back[1])
    push!(losses, loss_val)
end

println("Final Loss: ", losses[end])

model_trained = re(p)
model_cpu = model_trained |> cpu


plotlyjs()
plt_loss = plot(losses, xlabel="Epoch", ylabel="Loss", label="Training loss")
display(plt_loss)

u0 = Z[:, 1]
t_short = collect(0.0f0:dt:dt*99)

function predict_with_model(u0, model, dt, n_steps)
    u = u0
    traj = zeros(Float32, size(u0, 1), n_steps + 1)
    traj[:, 1] .= u
    for i in 1:n_steps
        du = model(u)
        u = u .+ dt .* du
        traj[:, i + 1] .= u
    end
    return traj
end

pred_short = predict_with_model(u0, model_cpu, dt, 99)
y_pred_short = pred_short[1, :]
y_true_short = Z[1, 1:100]
plt1 = plot(t_short, y_true_short, label="True y2(t)", lw=2)
plot!(plt1, t_short, y_pred_short, label="Predicted y2(t)", lw=2, ls=:dash, title="Prediction: 100 steps")
display(plt1)

n_long = 1000000
t_long = collect(0.0f0:dt:dt*(n_long - 1))
pred_long = predict_with_model(u0, model_cpu, dt, n_long - 1)
max_steps = min(size(pred_long, 2), size(Z, 2))
y_pred_long = pred_long[1, 1:max_steps]
y_true_long = Z[1, 1:max_steps]
t_plot = t_long[1:max_steps]

plt2 = plot(t_plot, y_true_long, label="True y2(t)", lw=2)
plot!(plt2, t_plot, y_pred_long, label="Predicted y(t)", lw=2, ls=:dash, title="$n_long steps with m= $m, n_steps = $n_steps and dt = $dt")
display(plt2)

kde_pred = kde(y_pred_long)
kde_obs = kde(y_true_long)
plot_kde = plot(kde_pred.x, kde_pred.density; label = "prediction", color = :red)
plot!(plot_kde, kde_obs.x, kde_obs.density; label = "observations", color = :blue)
display(plot_kde)


