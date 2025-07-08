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
    return fill(noise, 3)
end

#==================== DECORRELATION TIME ====================#
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

#==================== DELAY EMBEDDING ====================#
function delay_embedding(x; τ, m)
    q = round(Int, τ / dt)
    start_idx = 1 + (m - 1) * q
    Z = [ [x[i - j*q] for j in 0:m-1] for i in start_idx:length(x) ]
    return hcat(Z...)
end

#==================== NN MODEL ====================#
function create_nn(layers::Vector{Int})
    return Chain(
        Dense(layers[1], layers[2], tanh),
        LayerNorm(layers[2]),
        Dense(layers[2], layers[3], tanh),
        LayerNorm(layers[3]),
        Dense(layers[3], 1)  # output scalar y(t + dt)
    )
end

#==================== DATASET BUILDER ====================#
function build_dataset(y::Vector{Float32}, τ, m, dt)
    q = round(Int, τ / dt)
    start_idx = 1 + (m - 1) * q
    N = length(y) - start_idx - 1
    Z = [ [y[i - j*q] for j in 0:m-1] for i in start_idx:start_idx+N ]
    targets = y[start_idx+1:start_idx+N+1]
    return hcat(Z...), targets
end
#==================== LOSS ====================#
function loss_scalar(p)
    loss = 0.0f0
    for _ in 1:100
        i = rand(1:size(data_sample, 2))
        Z_t = data_sample[:, i]
        y_tpdt = target_values[i]
        y_pred = re(p)(Z_t)[1]  # scalar output
        loss += (y_pred - y_tpdt)^2
    end
    return loss / 100
end

#==================== MAIN ====================#
m = 10
layers = [m, 256, 256, 1]
model = create_nn(layers)
flat_p0, re = Flux.destructure(model)

dt = 0.001f0
n_steps = 40
Nsteps = 1000000

t_full = collect(0:dt:(Nsteps-1)*dt)
ε = 0.5

f = (x, t) -> F(x, t, ε)
obs_nn = evolve(randn(3), dt, Nsteps, f, sigma; timestepper=:rk4, resolution=1)

τ_opt, _ = estimate_tau(obs_nn[2, :], dt)
@info "Scelta ottimale di τ ≈ $(round(τ_opt, digits=4))"
τ = 0.25 * τ_opt  # corretto: τ_opt è già in unità di tempo

# normalize y before embedding
y = Float32.(obs_nn[2, :])
μ, σ = mean(y), std(y)
y_norm = (y .- μ) ./ σ

data_sample, target_values = build_dataset(y_norm, τ, m, dt)

#==================== TRAIN ====================#
p = flat_p0
opt = Optimisers.Adam(0.01)
state = Optimisers.setup(opt, p)
n_epochs = 500
losses = []

for epoch in ProgressBar(1:n_epochs)
    loss_val, back = Flux.withgradient(p) do p
        loss_scalar(p)
    end
    state, p = Optimisers.update(state, p, back[1])
    push!(losses, loss_val)
end

println("Final Loss: ", losses[end])
plotlyjs()
plt_loss = plot(losses, xlabel="Epoch", ylabel="Loss", label="Training loss", lw=2, title="Loss vs Epoch")
display(plt_loss)

#==================== PREDICTION ====================#
model_trained = re(p)

Z₀ = data_sample[:, 1]
n_pred = 10000
preds = zeros(Float32, n_pred)
preds[1] = model_trained(Z₀)[1]

for i in 2:n_pred
    Z₀ = vcat(preds[i-1:-1:max(i-m,1)], Z₀)[1:m]  # shift window
    preds[i] = model_trained(Z₀)[1]
end

true_vals = target_values[1:n_pred] .* σ .+ μ
pred_vals = preds .* σ .+ μ

t_plot = collect(0:dt:dt*(n_pred-1))
plt_pred = plot(t_plot, true_vals, label="True y₂(t)", lw=2)
plot!(plt_pred, t_plot, pred_vals, label="Predicted y(t)", lw=2, ls=:dash, title="Prediction: $n_pred steps")
display(plt_pred)

