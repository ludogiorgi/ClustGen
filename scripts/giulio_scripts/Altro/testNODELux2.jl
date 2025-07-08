using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using Lux
using Optimisers
using ComponentArrays
using OrdinaryDiffEq
using Zygote
using Random
using Plots
using ClustGen
using Statistics
using StatsBase
using CUDA
using ProgressBars: ProgressBar
using SciMLSensitivity: InterpolatingAdjoint, ZygoteVJP


#----------------------#
# 1. Model definition  #
#----------------------#

function create_nn(layers::Vector{Int}; activation_hidden=swish, activation_output=identity)
    layer_list = []
    for i in 1:(length(layers)-2)
        push!(layer_list, Lux.Dense(layers[i] => layers[i+1], activation_hidden))
    end
    push!(layer_list, Lux.Dense(layers[end-1] => layers[end], activation_output))
    return Lux.Chain(layer_list...)
end

m = 10 # delay embedding dimension
layers = [m, 128, 64, m]
activation_hidden = swish
activation_output = identity
nn = create_nn(layers; activation_hidden=activation_hidden, activation_output=activation_output)

# RNG and initialization
rng = Random.default_rng()
ps, st = Lux.setup(rng, nn)

#-------------------------#
# 2. Forward dynamics     #
#-------------------------#

dt = 0.001f0
n_steps = 100
t = collect(0.0f0:dt:dt*(n_steps - 1))
tspan = (t[1], t[end])


function dudt!(du, u, p, t)
    du .= first(nn(u, p, st))
end

function predict_neuralode(p, u0)
    prob = ODEProblem(dudt!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=t, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    return hcat(sol.u...)
end

#---------------------------#
# 3. Data generation        #
#---------------------------#

#---------------------------#
# 3.5. Facciamo un test con y= cos(t)#
#---------------------------#

Nsteps = 1000000
t_full = collect(0:dt:(Nsteps-1)*dt)

# cois(t) with random phase shift as initial condition
t0 = 2π * rand()  # random phase shift
signal = cos.(t_full .+ t0)

# Optional: aggiungi rumore se vuoi rendere il task più realistico
# signal .+= 0.01f0 * randn(length(signal))

# Delay embedding
function delay_embedding(x; τ, m)
    q = round(Int, τ / dt)
    start_idx = 1 + (m - 1) * q
    Z = [ [x[i - j*q] for j in 0:m-1] for i in start_idx:length(x) ]
    return hcat(Z...)
end

τ = 0.01  # delay time
Z = Float32.(delay_embedding(signal; τ=τ, m=m))
@show size(Z)
plot(Z[1, 1:10000], xlabel="y₁", label="Delay embedding", title="Delay embedding of cos(t)")

# dt = 0.001f0
batch_size = n_steps + 1
n_batches = 2000
n_epochs = 100
# Nsteps = 1000000
# ε = 0.5

# function F(x, t, ε ; µ=10.0, ρ=28.0, β=8/3)
#     dy1 = µ/ε^2 * (x[2] - x[1])
#     dy2 = 1/ε^2 * (x[1] * (ρ - x[3]) - x[2])
#     dy3 = 1/ε^2 * (x[1] * x[2] - β * x[3])
#     return [dy1, dy2, dy3]
# end

# function sigma(x, t; noise = 0.0)
#     sigma1 = noise
#     sigma2 = noise
#     sigma3 = noise
#     return [sigma1, sigma2, sigma3]
# end


# f = (x, t) -> F(x, t, ε)
# #obs_nn = evolve(randn(3), dt, Nsteps, f, sigma; timestepper=:rk4, resolution=100)
# println("➡️ Starting evolve...")
# obs_nn = evolve(randn(3), dt, Nsteps, f, sigma; timestepper=:rk4, resolution=100)
# println("✅ Finished evolve.")


# function estimate_tau(y, dt; threshold=0.2, maxlag=500)
#     y_centered = y .- mean(y)
#     acf = autocor(y_centered)
#     for i in 2:maxlag
#         if abs(acf[i]) < threshold
#             return i * dt, acf
#         end
#     end
#     return dt * maxlag, acf
# end

# τ_opt, _ = estimate_tau(obs_nn[2, :], dt)
# τ = τ_opt

# function delay_embedding(x; τ, m)
#     q = round(Int, τ / dt)
#     start_idx = 1 + (m - 1) * q
#     Z = [ [x[i - j*q] for j in 0:m-1] for i in start_idx:length(x) ]
#     return hcat(Z...)
# end

# Z = Float32.(delay_embedding(obs_nn[2, :]; τ=τ, m=m))

#---------------------------#
# 4. Data batching utils    #
#---------------------------#

function gen_batches(x::Matrix, batch_len::Int, n_batch::Int)
    datasize = size(x, 2)
    r = rand(1:datasize - batch_len, n_batch)
    return [x[:, i+1:i+batch_len] for i in r]
end

data_sample = gen_batches(Z, batch_size, n_batches)

#-----------------------------#
# 5. Loss + training loop     #
#-----------------------------#

function loss_neuralode(p)
    loss = 0.0f0
    for i in 1:100
        u = data_sample[rand(1:length(data_sample))]
        pred = predict_neuralode(p, u[:, 1])
        loss += sum(abs2, u[:, 2:end] .- pred[:, 1:end])
    end
    return loss / 100
end

pinit = ComponentArray(ps)
opt = Optimisers.Adam(0.01)
state = Optimisers.setup(opt, pinit)

losses = []

for epoch in ProgressBar(1:n_epochs)
    grad = Zygote.gradient(loss_neuralode, pinit)[1]
    state, pinit = Optimisers.update(state, pinit, grad)
    current_loss = loss_neuralode(pinit)
    push!(losses, current_loss)
end
println("Final Loss: ", losses[end])
plot(losses, xlabel="Epoch", ylabel="Loss", label="Training loss")







# Definizione: predizione per 5 steps (già addestrata così)
function predict_neuralode(p, u0)
    prob = ODEProblem(dudt!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=t, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    return hcat(sol.u...)
end

# Predizione
u0 = Z[:, 1]                             # condizione iniziale da Z
pred_y = predict_neuralode(pinit, u0)   # shape: (m, 5)

# Ground truth: i valori veri di y corrispondenti
true_y = Z[1, 1:1+length(t)]            # y(t₀), y(t₁), ..., y(t₅)  → 6 punti

# Costruzione asse temporale
t_pred = collect(t)                     # 0.0, 0.001, ..., 0.004  (5 steps dopo t₀)
t_full = [0.0; t_pred]                  # incluso t₀

# Include t₀ anche nella predizione
pred_y_full = hcat(u0, pred_y)[1, :]    # shape: (6,) — prima riga (y), colonne = tempi

# Plot
plt_pred = plot(t_full, true_y, label="True y(t)", lw=2)
plot!(plt_pred, t_full, pred_y_full, label="Predicted y(t)", lw=2, ls=:dash)
display(plt_pred)

function predict_neuralode2(p, u, tspan, t)
    prob = ODEProblem(dudt!, u, tspan, p)
    return hcat(solve(prob, Tsit5(), saveat=t,verbose=false).u...)
end
n_steps = 20000
tspan_long = (0.0f0, dt * (n_steps - 1))
t_long = 0.0f0:dt:(n_steps - 1)*dt

u0 = Z[:, 1]
pred_long = predict_neuralode2(pinit, u0, tspan_long, t_long)

max_steps = min(size(pred_long, 2), size(Z, 2))  # sicuro al 100%
y_true = Z[1, 1:max_steps]
y_pred = pred_long[1, 1:max_steps]
t_plot = t_long[1:max_steps]

plotlyjs()
plot(t_plot, y_true, label="True cos(t)", lw=2)
plot!(t_plot, y_pred, label="Predicted y(t)", lw=2, ls=:dash)














# Final trajectory prediction
tspan_pred = (0.0f0, 0.004f0)
t_pred = 0.0f0:0.001f0:0.005f0  # da t₀ a t₅

ind = 1
true_y_full = hcat(Z[:, ind], true_y)      # include t₀, t₁, ..., t₅
pred_y_full = hcat(Z[:, ind], pred_y)      # anche la predizione parte da t₀

plt1 = plot(t_pred, true_y_full[1, :], label="True y₂(t)", lw=2)
plot!(plt1, t_pred, pred_y_full[1, :], label="Predicted y₂(t)", lw=2, ls=:dash)

plt2 = plot(losses, xlabel="Epoch", ylabel="Loss", label="Training loss")

display(plt1)
function predict_neuralode2(p, u, tspan, t)
    prob = ODEProblem(dudt!, u, tspan, p)
    sol = solve(prob, Tsit5(), saveat=t, verbose=false, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    return hcat(sol.u...)
end

# Lungo intervallo di integrazione (100 steps)
t_pred_long = 0.0f0:0.001f0:0.099f0
tspan_long = (0.0f0, 0.099f0)

# Condizione iniziale
u0 = Z[:, 1]  # oppure qualunque colonna di Z

# Predizione
pred_long = predict_neuralode2(pinit, u0, tspan_long, t_pred_long)

n_pred = length(t_pred_long)
true_long = Z[1, 1:n_pred]


plt3 = plot(t_pred_long, true_long, label="True y(t)", lw=2)
plot!(plt3, t_pred_long, pred_long[1, :], label="Predicted y(t)", lw=2, ls=:dash)
display(plt3)