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

# function evolve_model_rk4(u0::AbstractVector, f::Function, steps::Int, dt::Float64)
#     dim = length(u0)
#     output = Matrix{eltype(u0)}(undef, dim, steps + 1)
#     output[:, 1] = u0
#     x = u0

#     for i in 1:steps
#         k1 = f(x)
#         k2 = f(x .+ dt/2 .* k1)
#         k3 = f(x .+ dt/2 .* k2)
#         k4 = f(x .+ dt .* k3)
#         x = x .+ dt/6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)
#         output[:, i+1] = x
#     end

#     return output
# end




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
    return loss / 50
end

# function loss_neuralode_rk4(p)
#     loss = 0.0f0
#     model = re(p)
#     f = x -> model(x) |> first
  # questa è compatibile con evolve_model_rk4


#     for i in 1:100
#         u = data_sample[rand(1:end)]
#         pred = evolve_model_rk4(u[:, 1], f, size(u, 2) - 1, Float64(dt))
#         loss += sum(abs2, u[:, 2:end] .- pred)
#     end

#     return loss / 100
# end


#========== MAIN ==========#

# ------------------------
# 1. Model definition
# ------------------------

m = 11  # delay embedding dim
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
n_steps = 55
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
plot(obs_signal[1:100:10000000])
# Estrai la terza variabile
signal_raw = obs_nn[3, :]

# Normalizza
M = mean(signal_raw)
S = std(signal_raw)
signal_norm = (signal_raw .- M) ./ S

# ========== MOVING AVERAGE AND RESIDUALS ========== #
# using ImageFiltering

# # Parametri
# window_radius = 20  # metà finestra → finestra totale = 2*radius + 1 = 99
# σ_gauss = window_radius / 3   # regola empirica
#        # deviazione standard della finestra gaussiana

# # Finestra gaussiana centrata
# g_kernel = exp.(-((-window_radius):window_radius).^2 ./ (2 * σ_gauss^2))
# g_kernel ./= sum(g_kernel)  # normalizza per avere somma 1

# # Applica la convoluzione gaussiana
# smoothed = imfilter(obs_signal, g_kernel, "reflect")
using ImageFiltering          # il pacchetto rimane lo stesso

# ----------------- parametri -----------------
window_radius = 3                    # metà finestra
window_size   = 2*window_radius + 1    # lunghezza totale (41)

# ----------- kernel uniforme (box) ----------
u_kernel = fill(1/window_size, window_size)   # somma = 1

# ----------- convoluzione -------------------
smoothed = imfilter(obs_signal, u_kernel, "reflect")


residual = obs_signal .- smoothed
mean_res = mean(residual)
std_res = std(residual)
res_norm = (residual .- mean_res) ./ std_res

plotlyjs()
plot(signal_norm[10:10000])
plot!(res_norm[10:10000])

# ========== PLOT VERIFICA ========== #
# #plot time series
# plotlyjs()
# plot(smoothed[1:10000000], label="Smoothed signal", xlabel="Time", ylabel="Amplitude", title="Moving Average of Observed Signal", linewidth=2)
# plot!(obs_signal[1:10000000])
# plot(residual[1:1000000], label="Residual", xlabel="Time", ylabel="Amplitude", title="Residual of Observed Signal", linewidth=2)
# plot!(obs_signal[1:1000000])

#plot pdf oscillations
kde_res = kde(residual)
kde_res_norm = kde(res_norm)
kde_real_y2 = kde(signal_norm)
plt_pdf_residual = plot(kde_real_y2.x, kde_real_y2.density, label="PDF Residual", xlabel="y2", ylabel="Density", title="PDF of Residual Signal", linewidth=2)
plot!(plt_pdf_residual, kde_res_norm.x, kde_res_norm.density, label="PDF Residual Normalized", xlabel="y2", ylabel="Density", title="PDF of Normalized Residual Signal", linewidth=2)
# autocov_res_norm = autocovariance(res_norm)
# acov = autocovariance(signal_norm; timesteps=500)

# plotlyjs()
# plot(autocov_res_norm)
# plot!(acov)

# plot(plt_pdf_residual, xax_y2, pdf_gaussian_y2, label="Gaussian", xlabel="y2", ylabel="Density", title="KDE of Residual Signal", linewidth=2)


#see if pdf oscillations is centered around 0 and has unit variance
μ = sum(kde_res_norm.density .* kde_res_norm.x) * step(kde_res_norm.x)
σ² = sum((kde_res_norm.x .- μ).^2 .* kde_res_norm.density) * step(kde_res_norm.x)

#plot pdf of smoothed signal
kde_smoothed = kde(smoothed)
plot(kde_smoothed.x, kde_smoothed.density, label="PDF Smoothed", xlabel="y2", ylabel="Density", title="PDF of Smoothed Signal", linewidth=2)

# ========= ESTIMATE OPTIMAL Tau ========= #
function estimate_tau(y, dt; threshold=0.2)
    y_centered = y .- mean(y)
    acf = autocovariance(y_centered, timesteps=500)
    for i in 2:length(acf)
        if abs(acf[i]) < threshold
            return i * dt, acf
        end
    end
    return dt * length(acf), acf
end

τ_opt, acf = estimate_tau(res_norm, dt)


@info "Scelta ottimale di τ ≈ $(round(τ_opt, digits=4))"
τ = τ_opt  

Z = Float32.(delay_embedding(res_norm; τ=τ, m=m))
kde1= kde(res_norm)
kde2 = kde(signal_norm)
plot(kde1.x, kde1.density)
plot!(kde2.x, kde2.density)
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

# ================= CONTINUA TRAINING DA EPOCA NSE PER CASO SI BLOCCA ================= #

# epoch_resume = 300  # <-- imposta qui il punto da cui riprendere
# @load "/Users/giuliodelfelice/Desktop/MIT/ClustGen/model_epoch_$(epoch_resume).bson" p # carica i pesi
# model_trained = re(p)  # rigenera il modello con pesi

# state = Optimisers.setup(opt, p)  # reinizializza l'ottimizzatore
# n_epochs_resume = 200  # nuove epoche da fare (o meno, come vuoi)

# # aggiorna losses array se vuoi mantenere continuità nel plot
# losses_resume = []

# for epoch in ProgressBar(1:n_epochs_resume)
#     u = data_sample[rand(1:end)]
#     loss_val, back = Flux.withgradient(p) do p
#         loss_neuralode(p)
#     end
#     state, p = Optimisers.update(state, p, back[1])
#     push!(losses_resume, loss_val)

#     global_epoch = epoch_resume + epoch
#     if global_epoch % save_every == 0
#         @save "model_epoch_$(global_epoch).bson" p
#     end
# end

# # unisci le due curve di loss (opzionale)
# append!(losses, losses_resume)

#========= FINE RESUME TRAINING =========#

# plot finale completo

println("Final Loss: ", losses[end])
plotlyjs()
plt_loss = plot(losses, xlabel="Epoch", ylabel="Loss", label="Training loss")
plot(plt_loss, losses, title="Loss vs Epoch", xlabel="Epoch", ylabel="Loss", label="Training loss", lw=2)


# ------------------------
# 7. Plot predictions
# ------------------------
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

# First n_long steps prediction vs truth
dt = 0.001
n_long = min(1000000, size(Z, 2))
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

#plot of the PDFs
plotlyjs()
kde_pred = kde(y_pred_long)
kde_obs = kde(y_true_long)
plot_kde = plot(kde_pred.x, kde_pred.density; label = "prediction", color = :red)
plot!(plot_kde, kde_res_norm.x, kde_res_norm.density; label = "observations", color = :blue)

# Display the plots

display(plt_loss)
display(plt1)
display(plt2)
display(plot_kde)


#=============== END MAIN ===============#








# TEST DEGLI IPERPARAMETRI SU 50 EPOCHE
# ============ SETUP ============ #

dt = 0.001f0
m = 13
τ = 0.4*τ_opt
n_steps = 45
η = 0.01
#n_epochs = 100


Z = Float32.(delay_embedding(res_norm; τ=τ, m=m))
batch_size = n_steps + 1
n_batches = 2000
data_sample = gen_batches(Z, batch_size, n_batches)
t = collect(0.0f0:dt:dt*(n_steps - 1))  # questo DEVE essere coerente con n_steps
tspan = (t[1], t[end])
layers = [m, 64, 64, m]
activation_hidden = swish
activation_output = identity
model = create_nn(layers; activation_hidden=activation_hidden, activation_output=activation_output)

#extract parameters from the model
flat_p0, re = Flux.destructure(model)


opt = Optimisers.Adam(η)
p = flat_p0  # punto di partenza
state = Optimisers.setup(opt, p)

n_epochs = 500
n_val = 200

Random.seed!(42)  # per consistenza
val_samples = data_sample[1:n_val]
train_samples = data_sample[(n_val+1):end]

losses_short = []

# ============ LOSS con input custom ============ #
function loss_neuralode_samples(p, samples)
    total = 0.0f0
    for u in samples
        pred = predict_neuralode(u[:, 1], p, tspan, t)
        total += sum(abs2, u[:, 2:end] .- pred[:, 1:end])
    end
    return total / length(samples)
end

# ============ TRAINING ============ #
@info "🔁 Inizio short training (50 epoche)"

for epoch in ProgressBar(1:n_epochs)
    # Mini-batch da 10 campioni casuali
    batch = [train_samples[rand(1:end)] for _ in 1:10]

    # Calcola gradienti su batch
    loss_val, back = Flux.withgradient(p) do p
        loss_neuralode_samples(p, batch)
    end

    # Aggiorna i pesi
    state, p = Optimisers.update(state, p, back[1])
    push!(losses_short, loss_val)
end

# ============ VALIDAZIONE ============ #
val_loss = loss_neuralode_samples(p, val_samples)

@info "✅ Fine short training"
@info "📉 Ultima training loss: $(losses_short[end])"
@info "📏 Validation loss: $val_loss"

# ============ PLOT (opzionale) ============ #
plt_loss = plot(losses_short, xlabel="Epoch", ylabel="Loss", label="Training loss (short run)", title="Quick Hyperparam Test")

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

# First n_long steps prediction vs truth
dt = 0.001
n_long = min(50000, size(Z, 2))
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
plot!(plt2, t_plot, y_pred_long, label="Predicted y(t)", lw=2, ls=:dash, title="$n_long steps with m= $m, n_steps = $n_steps dt = $dt and τ = $round(τ, digits = 3)")

#plot of the PDFs
plotlyjs()
kde_pred = kde(y_pred_long)
kde_obs = kde(y_true_long)
plot_kde = plot(kde_pred.x, kde_pred.density; label = "prediction", color = :red)
plot!(plot_kde, kde_res_norm.x, kde_res_norm.density; label = "observations", color = :blue)

# Display the plots

display(plt_loss)
display(plt1)
display(plt2)
display(plot_kde)

