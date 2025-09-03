#We now implement a NODE to learn x(t) directly, instead of separating the fast and slow dynamics. For results comparison we keep the same hyperparameters for the training and the preprocessing of data. m=10, n_steps=60, tau = 0.25*decorrelation time, n_epochs=1000


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
    sigma4 = noise
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
function delay_embedding(x; τ, m, dt_embed)
    q  = max(1, round(Int, τ/dt_embed))   # passi interi sulla griglia dt_embed
    τq = q * dt_embed                     # τ effettivo (quantizzato)
    start = 1 + (m-1)*q
    cols  = length(x) - start + 1
    Z = Matrix{eltype(x)}(undef, m, cols)
    @inbounds for c in 1:cols
        base = start + (c-1)
        for j in 0:m-1
            Z[m-j, c] = x[base - j*q]  # Invertito: m-j invece di j+1
        end
    end
    return Z, τq
end
#================= BATCH GENERATION ===================#

function gen_batches(x::Matrix, batch_len::Int, n_batch::Int)
    datasize = size(x, 2)
    r = rand(1:datasize - batch_len, n_batch)
    return [x[:, i+1:i+batch_len] for i in r]
end


function gen_batches_around_transitions(Z::AbstractMatrix, trans_idxs::AbstractVector{<:Integer}; pre_cols::Int, post_cols::Int)
    C = size(Z, 2)
    batches = Vector{SubArray{eltype(Z),2,typeof(Z),Tuple{Base.Slice{Base.OneTo{Int}},UnitRange{Int}},true}}()
    for idx in trans_idxs
        startc = idx - pre_cols
        endc   = idx + post_cols
        if 1 ≤ startc && endc ≤ C
            push!(batches, @view Z[:, startc:endc])  # vista, nessuna copia
        end
    end
    return batches
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
        loss += sum(abs2, (u[:, 1:end-1] .- pred[:, 1:end])* weights)
    end
    return loss / 100
end

#==================== NORMALIZE TIME SERIRES ====================#
function normalize_time_series(obs)
    mean_obs = mean(obs)
    sigma_obs = std(obs)
    return (obs .- mean_obs) ./ sigma_obs
end
#==================== DETECT CRITICAL TRANSITIONS ====================#
function detect_transitions(x::AbstractVector; threshold=0.8, window=50, min_spacing=100)
    transitions = []
    last = -min_spacing
    N = length(x)

    for i in 1:(N - window)
        # condizione: attraversamento dello 0
        if x[i] * x[i+1] < 0 && (i - last > min_spacing)
            # media successiva
            x_future = x[(i+1):(i+window)]
            m = mean(x_future)
            
            # media passata (per sapere in che minimo eravamo prima)
            x_past = x[max(1, i-window+1):i]
            m_past = mean(x_past)

            # se segni diversi ⇒ vero salto di minimo
            if abs(m) > threshold && abs(m_past) > threshold && sign(m) != sign(m_past)
                push!(transitions, i+1)
                last = i
            end
        end
    end

    return transitions
end
#========== MAIN ==========#

# ------------------------
# 1. Model definition
# ------------------------

m = 10  # delay embedding dim
layers = [m, 256, 256, m]
dt=0.001
activation_hidden = swish
activation_output = identity
model = create_nn(layers; activation_hidden=activation_hidden, activation_output=activation_output)

#extract parameters from the model
flat_p0, re = Flux.destructure(model)

# ------------------------
# 2. Data generation (cosine)
# ------------------------

n_steps = 100
dt_training = 0.001
t = collect(0.0f0:dt_training:dt_training*(n_steps - 1))
tspan = (t[1], t[end])
Nsteps = 100000000
t_full = collect(0:dt:(Nsteps-1)*dt)
ε = 0.5
σ_value=0.08
dim = 4 # Number of dimensions in the system
f(x, t) = F(x, t, σ_value, ε)
# ====== TEST COSINE ====== # 
# t0 = 2π * rand()
# signal = cos.(t_full .+ t0)
# τ = 0.01
# Z = Float32.(delay_embedding(signal; τ=τ, m=m))#
# ====== end test ======= #

obs_nn = evolve(randn(4), dt, Nsteps, f, sigma; timestepper=:rk4, resolution=10)
length(obs_nn[1,:])
# normalize y before embedding
obs = normalize_time_series(obs_nn[1,:])
y2 = normalize_time_series(obs_nn[3,:])
plotlyjs()
plot(y2[1:10000])
train_frac = 0.8
Ntrain = Int(floor(Nsteps/10 * train_frac))
#Nval = Nsteps - Ntrain

obs_train = obs[1:Ntrain]    #8M steps
obs_val = obs[Ntrain+1:end]  #2M steps

# ========= ESTIMATE OPTIMAL Tau ========= #

# dati subcampionati a 0.01
step = round(Int, dt_training/dt)
obs_train_sub = obs_train[1:step:end]
obs_val_sub   = obs_val[1:step:end]


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

# τ dalla stima su dt_fine, poi usato e quantizzato su dt_training
τ_opt, acf = estimate_tau(obs_nn[3,:], dt)
τ_opt = 0.073
τ = 0.25 * τ_opt

Z_train, τq = delay_embedding(obs_train_sub; τ=τ, m=m, dt_embed=dt_training)
Z_val,   _       = delay_embedding(obs_val_sub;   τ=τ, m=m, dt_embed=dt_training)

Z_val_obs, _  = delay_embedding(obs_val; τ=τ, m=m, dt_embed=dt)

q = round(Int, τ / dt_training)  # Aggiungi questa linea per definire q
@info "τ richiesto=$(round(τ, digits=5)) s  |  τ quantizzato=$(round(τq, digits=5)) s  |  q=$q"

trans_cols = detect_transitions(Z_train[1, :];
    threshold=0.8,
    window=round(Int, 0.5/dt_training),
    min_spacing=round(Int, 1.0/dt_training)
)



# ---------- Visualizza e sovrapponi l'ultimo vettore ---------- #
n_plot = min(11700, size(Z_train,2)-1)
j = rand(1:(size(Z_train,2)-n_plot))

ts = (0:n_plot-1) .* dt_training
x_series       = vec(Z_train[1, j : j+n_plot-1])        # prima riga = x(t) allineato alle colonne
last_embedding = Z_train[:, j+n_plot-1]
embedding_times = ts[end] .- (0:m-1) .* τq              # usa τ quantizzato!

plotlyjs()
plot(ts, x_series; lw=2, label="x(t)", xlabel="t", ylabel="x", legend=:topright)
scatter!(embedding_times, last_embedding; ms=6, label="Delay embedding (ultimo vettore)")

# ------------------------
# 3. Batching
# ------------------------


batch_size = n_steps + 1
n_batches = 2000
data_sample = gen_batches(Z_train, batch_size, n_batches)

# batch_size = n_steps + 1
# n_batches  = 2000
# post_cols  = 10;  pre_cols = batch_size - 1 - post_cols  # idx-pre : idx+post
# trans_cols = detect_transitions(Z_train[1,:]; threshold=0.8, window=round(Int,0.5/dt_training), min_spacing=round(Int,1.0/dt_training))
# trans_cols = Int.(trans_cols)
# near_pool  = gen_batches_around_transitions(Z_train, trans_cols; pre_cols=pre_cols, post_cols=post_cols)
# nb_near = Int(0.5 * n_batches)
# data_sample = vcat(near_pool[randperm(length(near_pool))[1:nb_near]], gen_batches(Z_train, batch_size, n_batches - nb_near)) |> x -> x[randperm(length(x))]






# ------------------------
# 4. Training the model
# ------------------------
p = flat_p0
opt = Optimisers.Adam(0.01)
state = Optimisers.setup(opt, p)
n_epochs = 1000
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
        mkpath("best_NODE_x")  # crea la cartella se non esiste
        @save joinpath("best_NODE_x", "model_epoch_$(epoch).bson") p
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
@load "/Users/giuliodelfelice/Desktop/MIT/ClustGen/best_NODE_x/model_epoch_1000.bson" p

model_trained = re(p)

# First 100 steps prediction vs truth
u0 = Z_val[:, 1]
dt = 0.001
t_short = collect(0.0f0:dt:dt*29)
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
x_pred_short = pred_short[1, :]
x_true_short = Z_val[1, 1:30]

plotlyjs()
plt1 = plot(t_short, x_true_short, label="True x(t)", lw=2)
plot!(plt1, t_short, x_pred_short, label="Predicted x(t)", lw=2, ls=:dash, title="Prediction: 100 steps")
display(plt1)

# First n_long steps prediction vs truth
n_long = 1000
t_long = collect(0.0f0:dt:dt*(n_long - 1))
tspan_long = (t_long[1], t_long[end])
pred_long = predict_with_model(u0, model_trained, tspan_long, t_long)
max_steps = min(size(pred_long, 2), size(Z_val, 2))
x_pred_long = pred_long[1, 1:max_steps]
x_true_long = Z_val[1, 1:max_steps]
t_plot = t_long[1:max_steps]

M_x, S_x = mean(x_pred_long), std(x_pred_long)
x_pred_long_norm = (x_pred_long .- M_x) ./ S_x

#Plot of the time series
plotlyjs()
plt2 = plot(t_plot[1:end], x_true_long[1:end], label="Observed y2(t)", lw=2, color=:red)
plot!(plt2, t_plot[1:end], x_pred_long[1:end], label="NODE y(t)", lw=2, color=:blue, title="Trajectories compared")
display(plt2)
#plot of the PDFs
plotlyjs()
kde_pred = kde(x_pred_long_norm)
kde_obs = kde(x_true_long)
plot_kde = plot(kde_pred.x, kde_pred.density; label = "NODE", lw=2, color = :blue)
plot!(plot_kde, kde_obs.x, kde_obs.density; label = "observations", lw=2, color = :red, title="PDF compared")
# μ = mean(x_true_long)
# σ = std(x_true_long)
# x_gauss = range(minimum(kde_obs.x), stop=maximum(kde_obs.x), length=500)
# pdf_gauss = pdf.(Normal(μ, σ), x_gauss)

#plot!(plot_kde, x_gauss, pdf_gauss; label = "Gaussian", lw=2, color = :lime)

display(plot_kde)


plotlyjs()


# === 2. Funzione di integrazione della NODE ===
function predict_with_model(u0, model, tspan, t)
    function dudt!(du, u, _, t)
        du .= model(u)
    end
    prob = ODEProblem(dudt!, u0, tspan)
    sol = solve(prob, Tsit5(), saveat=t)
    return hcat(sol.u...)  # colonne = stati nel tempo
end

# === 3. Parametri iniziali e serie osservata ===
# obs_val = Z_val[1, 1:10:end]        # osservabile x(t)
# Z_val_sub = Z_val[:, 1:10:end]      # delay embeddings sottocampionato
dt_sub = dt * 10                    # nuovo dt dopo subsampling
# === 4. Transizioni critiche ===
transitions = detect_transitions(Z_val[1, :]; threshold=0.7, window=800, min_spacing=100)
if isempty(transitions)
    error("Nessuna transizione trovata. Cambia threshold o window.")
else
    @show transitions[1:min(10, end)]
end
# === 5. Parametri di previsione ===
theta = [0.5, 0.8, 1.0, 1.8, 5.0, 10.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]  # in secondi
timesteps = round.(Int, theta ./ dt)


# === 6. Calcolo RMSE vs lag ===
rmses_vs_theta = Float64[]
var_rmses_x = Float64[]

for timestep in timesteps
    rmses = Float64[]
    for tau_n in transitions[20:end]
        u0 = obs_val[tau_n - timestep]  # delay embedding iniziale
        t = collect(0:dt:dt * (timestep + 10))
        tspan = (0.0, t[end])
        sol_matrix = predict_with_model(u0, model, tspan, t)
        pred = sol_matrix[1, :]  # prima riga: x(t)

        real = obs_val[(tau_n - timestep):(tau_n + 10)]
        @assert length(pred) == length(real)

        rmse = sqrt(mean((pred .- real).^2))
        push!(rmses, rmse)
    end
    push!(rmses_vs_theta, mean(rmses))
    push!(var_rmses_x, std(rmses))
end

# === 7. Plot RMSE vs lag ===
plt_rmse_node = plot(
    theta, rmses_vs_theta;
    ribbon=var_rmses_x,
    label="RMSE NODE",
    lw=2,
    xlabel="Prediction Lag (s)",
    ylabel="RMSE",
    title="RMSE vs Forecast Horizon (NODE)",
    legend=:topleft,
    size=(800, 300),
)

scatter!(plt_rmse_node, theta, rmses_vs_theta; marker=:circle, markersize=5, label="", color=:black)
scatter!(plt_rmse_node, theta, rmses_vs_theta; marker=:circle, markersize=4, markercolor=palette(:auto)[1], label="")
display(plt_rmse_node)


# === 8. Plot predizione vs realtà per alcune transizioni ===
indices_to_plot = [1, 2, 3, 4, 5]
colors = [:blue, :orange]
p_combined = plot(layout=(length(indices_to_plot), 1), size=(800, 1200))

for (k, idx) in enumerate(indices_to_plot)
    timestep = timesteps[idx]
    tau_n = transitions[6]  # puoi cambiare con un'altra transizione
    
    # Verifica bounds prima di procedere
    if tau_n - timestep < 1 || tau_n + 500 > length(obs_val)
        println("Skipping plot $k: out of bounds")
        continue
    end
    
    u0 = Z_val[:, tau_n - timestep]
    t = collect(0:dt:dt * (timestep + 500))
    tspan = (0.0, t[end])
    sol_matrix = predict_with_model(u0, model_trained, tspan, t)
    pred = sol_matrix[1, :]

    # Calcola gli indici corretti per il subsampling
    start_idx = tau_n - timestep
    end_idx = min(tau_n + 500, length(obs_val))
    real = Z_val[1, start_idx:end_idx]

    # Assicura che pred e real abbiano la stessa lunghezza
    min_length = min(length(pred), length(real))
    pred = pred[1:min_length]
    real = real[1:min_length]
    
    t_plot = dt .* (0:(min_length-1))

    plot!(p_combined[k], t_plot, real, legend=false, color=colors[1], lw=2)
    plot!(p_combined[k], t_plot, pred, legend=false, color=colors[2], lw=2)
    title!(p_combined[k], "θ = $(round(dt * timestep; digits=1)) s")
    xlabel!(p_combined[k], "")
    ylabel!(p_combined[k], "x(t)")
end

display(p_combined)

    # Assicura che pred e real abbiano la stessa lunghezza
    min_length = min(length(pred), length(real))
    pred = pred[1:min_length]
    real = real[1:min_length]
    
    t_plot = dt .* (0:(min_length-1))

    plot!(p_combined[k], t_plot, real, legend=false, color=colors[1], lw=2)
    plot!(p_combined[k], t_plot, pred, legend=false, color=colors[2], lw=2)
    title!(p_combined[k], "θ = $(round(dt * timestep; digits=1)) s")
    xlabel!(p_combined[k], "")
    ylabel!(p_combined[k], "x(t)")
end

display(p_combined)





