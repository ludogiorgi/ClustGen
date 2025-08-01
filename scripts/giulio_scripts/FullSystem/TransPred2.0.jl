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

#====================Smoothed derivative of x(t)=====================#
function smoothed_dx(x, dt)
    dx = similar(x)
    dx[1] = (x[2] - x[1]) / dt
    dx[end] = (x[end] - x[end-1]) / dt
    @inbounds for i in 2:length(x)-1
        dx[i] = (x[i+1] - x[i-1]) / (2*dt)
    end
    return dx
end

#====================ESTIMATE DECORRELATION TIME=====================#
function estimate_tau(y, dt; threshold=0.1)
    y_centered = y .- mean(y)
    acf = autocovariance(y_centered, timesteps=500)
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

#==================== ESTIMATE INITIAL z0 VECTOR ====================#

function full_initial_condition(input_vec, correction_net)
    x0, z_base, Y0 = input_vec
    correction = correction_net(Y0)  # adesso input ∈ ℝ^k
    z_corr = z_base .+ correction    # z_corr ∈ ℝ^m
    return vcat(x0, z_corr)          # full state ∈ ℝ^{1+m}
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
#==================== RECONSTRUCT MODELS FROM SINGLE PARAMETER VECTOR ====================#
function re_combined(p)
    N1 = length(flat_p0_NODE)
    p_node = p[1:N1]
    p_corr = p[N1+1:end]
    return re_NODE(p_node), re_corr(p_corr)
end

function split_state(s::AbstractVector)
    x = @view s[1:1]      # still view, but scalar inside Vector{Float32}
    z = @view s[2:end]
    return x, z
end

#==================== FULL DYNAMICS ====================#
function full_dynamics!(ds, s, p, t, g_theta)
    x, z = split_state(s)
    ds[1] = score_clustered_xt(x)[1] + z[1] #1st entry is score + y(t0)
    gz = g_theta(z)     #\dot(y) = g_theta(delay_embedding(y))
    ds[2] = gz[1]
    @inbounds for i in 3:m+1
        ds[i] = z[i-1]
    end
end

#==================== DYNAMICS WRAPPER ====================#
function dudt!(du, u, p, t)
    g_theta, _ = re_combined(p)
    full_dynamics!(du, u, p, t, g_theta)
end

#==================== INTEGRATION ====================#
function predict_neuralode(input_vec, p, tspan, tsteps)

    s0 = full_initial_condition(input_vec, correction_net)
    prob = ODEProblem(dudt!, s0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=tsteps,
                sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
                abstol=1e-6, reltol=1e-6)

    if sol.retcode != :Success
        return fill(1e6f0, length(tsteps))
    end
    return map(u -> u[1], sol.u)
end


#==================== LOSS FUNCTION ====================#
function loss_neuralode(p)
    loss = 0.0f0
    for _ in 1:100
        (x0, z0_base, z0_context), x_true = rand(data_sample)
        x_pred = predict_neuralode((x0, z0_base, z0_context), p, tspan, t)
        loss += sum((x_pred .- x_true[1:end-1]).^2)
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
function detect_transitions(x::Vector{Float64}; threshold=0.8, window=50, min_spacing=100)
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

m = 15  # delay embedding dim
# NODE model
layers_node = [m, 256, 256, 1]
g_theta = create_nn(layers_node; activation_hidden=swish, activation_output=identity)

# Correction NN
k = 8  
layers_corr = [k, 32, m]
correction_net = create_nn(layers_corr; activation_hidden=swish, activation_output=identity)




#extract parameters from the model
flat_p0_NODE, re_NODE = Flux.destructure(g_theta)
flat_p0_corr, re_corr = Flux.destructure(correction_net)
flat_p0 = vcat(flat_p0_NODE, flat_p0_corr)  # totale: N1 + N2 elementi

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
n_steps = 70
dt_training = 0.001f0

#generate data
t = collect(0.0f0:dt_training:dt_training*(n_steps - 1))
tspan = (t[1], t[end])
Nsteps = 10000000
t_full = collect(0:dt:(Nsteps-1)*dt)
f(x, t) = F(x, t, σ_value, ε)
obs_nn = evolve(randn(4), dt, Nsteps, f, sigma; timestepper=:rk4, resolution=1)

#normalize data
obs_signal = normalize_time_series(obs_nn[1,:])
kde_obs = kde(obs_signal)

y2_obs = Float32.(obs_nn[3, :])

# Normalizza
y2_obs_norm = normalize_time_series(y2_obs)


#ORA AL POSTO DI FARE LA MOVING AVERAGE ESTRAIAMO LA Y2 DA X E SCORE FUNCTION. E VEDIAMO COME VA
# ========== COMPUTE SCORE FUNCTION USING FIRST NN ========== #
autocov_obs_nn = zeros(4, 100)#

for i in 1:4
    autocov_obs_nn[i, :] = autocovariance(obs_nn[i, :]; timesteps=100)
end

D_eff = dt * (0.5 * autocov_obs_nn[3, 1] + sum(autocov_obs_nn[3, 2:end-1]) + 0.5 * autocov_obs_nn[3, end])
D_eff = 0.3

#divide data into training set and validation set
train_frac = 0.8
Ntrain = Int(floor(Nsteps * train_frac))
#Nval = Nsteps - Ntrain

obs_train = Float32.(obs_signal[1:Ntrain])   #8M steps
obs_val = Float32.(obs_signal[Ntrain+1:end])  #2M steps


#training and clustering parameters 
σ_value=0.05
prob=0.001
conv_param=0.02
n_epochs=5000
batch_size=16


# ------------------------
# 3. Clustering
# ------------------------
averages, centers, Nc, labels = f_tilde_labels(σ_value, reshape(obs_train[1:10:end], 1, :); prob=prob, do_print=false, conv_param=conv_param, normalization=false)
inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)

# ------------------------
# 4. Compute the score function
# ------------------------

#analytic score function
M = mean(obs_nn, dims=2)[1]
S = std(obs_nn, dims=2)[1]
f1(x,t) = x .- x.^3
score_true(x, t) = normalize_f(f1, x, t, M, S)

#learned score function
#kde_x = kde(obs_nn[1, :])
centers_sorted_indices = sortperm(centers[1, :])
centers_sorted = centers[:, centers_sorted_indices][:]
scores = .- averages[:, centers_sorted_indices][:] ./ σ_value

# ------------------------
# 5. Train NN
# ------------------------
@time nn, losses = train(inputs_targets, n_epochs, batch_size, [1, 50, 25, 1];
    opt=Flux.Adam(0.001), activation=swish, last_activation=identity,
    use_gpu=false)

nn_clustered_cpu = nn |> cpu
score_clustered(x) = .- nn_clustered_cpu(reshape(Float32[x...], :, 1))[:] ./ σ_value


# ------------------------
# 6. Check the score function
# ------------------------
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

#Plot Score
p_score = scatter(centers_sorted, scores; color=:blue, alpha=0.2, label="Cluster centers",
    xlims=(-1.3, 1.3), ylims=(-5, 5), xlabel="𝑥", ylabel="Score(𝑥)", title="Score Function Estimate")
plot!(p_score, xax, interpolated_score; label="NN interpolation", linewidth=2, color=:red)
plot!(p_score, xax, true_score; label="Score analytic", linewidth=2, color=:lime)

display(p_score)

# ------------------------
# 7. Phi calculation
# ------------------------
#rate matrix
dt = 0.001f0
Q = generator(labels; dt=dt)*0.1
P_steady = steady_state(Q)
#test if Q approximates well the dynamics
tsteps = 501
res = 10

auto_obs = autocovariance(obs_train[1:res:end]; timesteps=tsteps) 
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
autocov_y2 = autocovariance(y2_obs_norm; timesteps=1000)
plotlyjs()
plot(autocov_y2, label="Autocovariance of y2", xlabel="Lag", ylabel="Autocovariance",
    title="Autocovariance of the estimated y2 signal", linewidth=2)

########## Estimate y2 from the slow variable x ##########
dt=0.001
Σ_rescaled = Σ / sqrt(2*1.5)  # Rescale Σ for the score function


# ------------------------
# 8. Batching for NODE training                                                     (non è necessario costruire il delay embedding qui perché il delay embedding viene calcolato direttamente sulla y generata dalla x delle batch)
# ------------------------

acf_y2 = autocovariance(y2_obs_norm, timesteps=500)
plotlyjs()
plot(acf_y2)


tau_opt, _ = estimate_tau(y2_obs_norm, dt)
τ = 0.25*tau_opt
n_batches = 2000
#==================== DATASET CREATION (2000 batches) ====================#
q = round(Int, τ / dt)
start_idx = 1 + (m - 1) * q
last_valid = length(obs_train) - n_steps

# Reconstruct y(t)
dx = smoothed_dx(obs_train, dt)
scores = [score_clustered_xt([xi])[1] for xi in obs_train]
y_series = dx .- scores
y_series = normalize_time_series(y_series)

# Delay embedding of y
Z_all = delay_embedding(y_series; τ=τ, m=m)  # shape: (m, T_valid)
# Select 2000 random starting indices within valid range
valid_indices = collect(1:(last_valid - start_idx + 1))
selected_indices = sort(rand(valid_indices, n_batches))  # Optional sort: for deterministic order

# Build batches
batch_inputs = Vector{Tuple{Float32, Vector{Float32}, Vector{Float32}}}()
batch_targets = Vector{Vector{Float32}}()


for i in selected_indices
    # x0
    x0 = Float32(obs_train[start_idx + i - 1])

    # z0 base (m-dim delay embedding)
    z0_base = Float32.(Z_all[:, i])  # vettore ∈ ℝᵐ

    k = 8 
    if k > m
        error("k must be ≤ m")
    end
    z0_context = z0_base[1:k]  # vettore ∈ ℝᵏ

    # target: future trajectory of x
    x_target = Float32.(obs_train[start_idx + i - 1 : start_idx + i - 1 + n_steps])

    push!(batch_inputs, (x0, z0_base, z0_context))
    push!(batch_targets, x_target)
end

data_sample = collect(zip(batch_inputs, batch_targets))


# ------------------------
# 9. Training the model
# ------------------------
# weights_vec = exp.(LinRange(0.0f0, -1.0f0, n_steps))

p = flat_p0
opt = Optimisers.Adam(0.01)
state = Optimisers.setup(opt, p)
n_epochs = 500
losses = []

@info "Starting training..."
using BSON: @save
save_every = 100 

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



# plot finale completo

println("Final Loss: ", losses[end])
plotlyjs()
plt_loss = plot(losses, xlabel="Epoch", ylabel="Loss", label="Training loss")
plot(plt_loss, losses, title="Loss vs Epoch", xlabel="Epoch", ylabel="Loss", label="Training loss")

# ------------------------
# 10a. Check correction net training
# ------------------------
@load "/Users/giuliodelfelice/Desktop/MIT/ClustGen/model_epoch_500.bson" p
g_theta, correction_net = re_combined(p)


norms = Float64[]
z_bases = Matrix{Float64}(undef, m, length(data_sample))

for (i, (input_vec, _)) in enumerate(data_sample)
    z_base = input_vec[2:end]
    correction = correction_net(z_base)
    push!(norms, norm(correction))
    z_bases[:, i] .= z_base
end

plot(norms, xlabel="Training sample index", ylabel="‖correction‖", 
     label="Correction magnitude", lw=2, title="Correction size across dataset")



y_base = [input_vec[2] for (input_vec, _) in data_sample]
y_corr = [input_vec[2] + correction_net(input_vec[2:end])[1] for (input_vec, _) in data_sample]

plot(y_base, label="Base y₀", alpha=0.7)
plot!(y_corr, label="Corrected y₀", alpha=0.7)

# ------------------------
# 10b. Check NODE training: Plot y2 predictions
# ------------------------

dt=0.01
# === Funzione per stimare y₀ corretto === #
function estimate_z0_corrected(x_series, dt, correction_net; m, τ, k)
    dx = smoothed_dx(x_series, dt)
    scores = [score_clustered_xt([x])[1] for x in x_series]
    y_series = dx .- scores
    y_series = normalize_time_series(y_series)

    q = round(Int, τ / dt)

    # z_base = delay embedding completo di dimensione m
    z_base = delay_embedding(y_series; τ=τ, m=m)[:, 1]

    # y_input = primi k punti di y_series distanziati di τ (cioè ogni q step)
    y_input = [y_series[end - (i - 1)*q] for i in 1:k]  # y₀, y₋τ, ..., y₋(k−1)τ
    y_input = Float32.(reverse(y_input))  # Riordina da passato→presente

    correction = correction_net(y_input)  # ℝᵏ → ℝᵐ
    z_corr = z_base .+ correction
    return z_corr
end


# === Predict con NODE === #
function predict_y_trajectory(z0, model, tspan, t)
    function dudt!(du, u, _, t)
        du .= model(u)
    end
    prob = ODEProblem(dudt!, z0, tspan)
    sol = solve(prob, Tsit5(), saveat=t)
    return hcat(sol.u...)  # Each column is state z(t)
end

# === INPUT: segmento iniziale della x === #
window_length = (m - 1) * round(Int, τ / dt) + 1  # Lunghezza necessaria per delay embedding
x_segment = obs_signal[1:window_length]  # Oppure scegli un altro punto iniziale valido

# === Condizione iniziale z₀ corretta === #
z0 = estimate_z0_corrected(x_segment, dt, correction_net; m=m, τ=τ, k=k)
# === Predizione breve === #
t_short = 0.0f0:dt:dt*49
tspan_short = (t_short[1], t_short[end])
z_pred_short = predict_y_trajectory(z0, g_theta, tspan_short, t_short)
y_pred_short = z_pred_short[1,:]
y_true_short = y2_obs_norm[1:10:(50*10)]  # ground truth dalla serie di y già calcolata

plotlyjs()
plt1 = plot(t_short, y_true_short, label="True y(t)", lw=2, color=:red)
plot!(plt1, t_short, y_pred_short, label="Predicted y(t)", lw=2, color=:blue, title="Prediction: 100 steps")
display(plt1)

# === Predizione lunga === #
n_long = 1000
t_long = 0.0f0:dt:dt*(n_long - 1)
length(t_long)
tspan_long = (t_long[1], t_long[end])
z_pred_long = predict_y_trajectory(z0, g_theta, tspan_long, t_long)
size(z_pred_long, 2)

max_steps = min(size(z_pred_long, 2), length(y2_obs_norm))
y_pred_long = z_pred_long[1:max_steps]
y_true_long = y2_obs_norm[1:10:(max_steps*10)]
t_plot = t_long[1:max_steps]


y_pred_long_norm = normalize_time_series(y_pred_long)


# Plot finale
plt2 = plot(t_plot, y_true_long, label="True y(t)", lw=2, color=:red)
plot!(plt2, t_plot, y_pred_long, label="Predicted y(t)", lw=2, ls=:dash, color=:blue,
       title="Long-term y(t) Prediction")

kde_pred = kde(y_pred_long_norm)
kde_obs = kde(y_true_long)

plot_kde = plot(kde_pred.x, kde_pred.density; label = "NODE", lw=2, color = :blue)
plot!(plot_kde, kde_obs.x, kde_obs.density; label = "observations", lw=2, color = :red, title="PDF compared")

display(plot_kde)
display(plt2)





# ------------------------
# 10c. Plot predictions: Focus on short-term predictions for x
# ------------------------
@load "/Users/giuliodelfelice/Desktop/MIT/ClustGen/model_epoch_500.bson" p
g_theta, correction_net = re_combined(p)

# Useful functions
function estimate_y0_corrected(x_series, dt, correction_net; m, τ)
    dx = smoothed_dx(x_series, dt)
    scores = [score_clustered_xt([x])[1] for x in x_series]
    y_series = dx .- scores
    y_series = normalize_time_series(y_series)

    z_base = delay_embedding(y_series; τ=τ, m=m)[:, 1]  # vettore ∈ ℝᵐ
    correction = correction_net(z_base)                 # vettore ∈ ℝᵐ

    y0_corrected = z_base .+ correction           # SCALARE
    return y0_corrected
end


function predict_full_dynamics(x0, z0, p, tspan, tsteps)
    function dudt_wrapper!(du, u, p, t)
        g_theta, _ = re_combined(p)
        full_dynamics!(du, u, p, t, g_theta)
    end
    s0 = vcat(x0, z0)
    prob = ODEProblem(dudt_wrapper!, s0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=tsteps,
                sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    return map(u -> u[1], sol.u)
end
dt = 0.01
# transitions prediction RMSE
lags = [0.35, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
timesteps = round.(Int, lags ./ dt)

obs_val = obs_signal[Ntrain+1:end]
obs_val = obs_val[1:10:end]  # subsample se necessario

transitions = detect_transitions(obs_val; threshold=0.7, window=800, min_spacing=100)

rmses_vs_lags = Float64[]
var_rmses = Float64[]
for tau_steps in timesteps
    rmses = Float64[]
    for tau_n in transitions
        if tau_n ≤ tau_steps + (m - 1) * round(Int, τ/dt)
            continue  # serve spazio per delay embedding
        end

        x_segment = obs_val[(tau_n - tau_steps - (m-1)*round(Int, τ/dt)) : (tau_n - tau_steps)]
        x0 = Float32(obs_val[tau_n - tau_steps])

        z0 = estimate_y0_corrected(x_segment, dt, correction_net; m=m, τ=τ)

        pred = predict_full_dynamics(x0, z0, p, (0.0f0, dt*(tau_steps + 10)), 0.0f0:dt:(tau_steps + 10)*dt)
        real = obs_val[tau_n - tau_steps : tau_n + 10]

        if length(pred) != length(real)
            continue
        end

        rmse = sqrt(mean((pred .- real).^2))
        push!(rmses, rmse)
    end
    push!(rmses_vs_lags, mean(rmses))
    push!(var_rmses, std(rmses))
end 
@show rmses_vs_lags


default(markerstrokecolor=:black)

# --- RMSE vs Lag Plot ---
plt_rmse = plot(
    lags, rmses_vs_lags;
    ribbon = var_rmses,
    label = "RMSE model",
    lw = 2,
    xlabel = "Prediction Lag (s)",
    ylabel = "RMSE",
    title = "RMSE vs Forecast Horizon",
    legend = :topleft,
    size = (800, 300),
)

# Add black border-only circles
scatter!(
    plt_rmse, lags, rmses_vs_lags;
    marker = :circle,
    markersize = 5,
    label = "",
    color = :black,
)

# Add colored inner markers
scatter!(
    plt_rmse, lags, rmses_vs_lags;
    marker = :circle,
    markersize = 4,
    markercolor = palette(:auto)[1],
    label = "",
)

display(plt_rmse)


#transitions predictions exemple plots
indices_to_plot = [1, 2, 3, 4, 5]
colors = [:blue, :orange]
p_combined = plot(layout=(5,1), size=(800, 1200))

for (k, idx) in enumerate(indices_to_plot)
    τ_steps = timesteps[idx]
    τ_n = transitions[35]  # o altra transizione

    x_segment = obs_val[(τ_n - τ_steps - (m-1)*round(Int, τ/dt)) : (τ_n - τ_steps)]
    x0 = Float32(obs_val[τ_n - τ_steps])

    _, correction_net = re_combined(p)
    z0 = estimate_y0_corrected(x_segment, dt, correction_net; m=m, τ=τ)

    pred = predict_full_dynamics(x0, z0, p, (0.0f0, dt*500), 0.0f0:dt:dt*500)
    real = obs_val[τ_n - τ_steps : τ_n + 500]
    t_plot = dt .* (0:(length(pred)-1))

    plot!(p_combined[k], t_plot, real, color=colors[1], lw=2, legend=false)
    plot!(p_combined[k], t_plot, pred, color=colors[2], lw=2, legend=false)
    title!(p_combined[k], "Lag = $(lags[idx]) s")
end

display(p_combined)












plot(y2_obs_norm[1:10000])
# acfs_pred = Matrix{Float64}(undef, 100, 10)
# acfs_true = Matrix{Float64}(undef, 100, 10)

# for n in 1:10
#     # First 500 steps prediction vs truth
#     dt = 0.01
#     j = rand(1:size(Z_val, 2))
#     u0 = Z_val[:, j]
#     t_short = collect(0.0f0:dt:dt*9000)
#     tspan_short = (t_short[1], t_short[end])

#     function predict_with_model(u0, model, tspan, t)
#         function dudt!(du, u, _, t)
#             du .= model(u)
#         end
#         prob = ODEProblem(dudt!, u0, tspan)
#         sol = solve(prob, Tsit5(), saveat=t)
#         return hcat(sol.u...)
#     end

#     pred_short = predict_with_model(u0, model_trained, tspan_short, t_short)
#     y_pred_short = pred_short[1, :]
#     y_pred_short = normalize_time_series(y_pred_short)

#     y_true_short = Z_val[1, j:10:(j + 10*9000)]

#     acf_y_pred_short = autocovariance(y_pred_short, timesteps = 100)
#     acf_y_true_short = autocovariance(y_true_short, timesteps = 100)

#     # Inserisci nella colonna n-esima
#     acfs_pred[:, n] .= acf_y_pred_short
#     acfs_true[:, n] .= acf_y_true_short


# plotlyjs()

# plt1 = plot(t_short[1:250], y_true_short[1:250]; label="True y₂(t)", lw=2, color=:blue, markershape=:square, markerstrokewidth=1, markersize=3, line=:solid, marker=:auto)

# plot!(plt1, t_short[1:250], y_pred_short[1:250]; label="Predicted y₂(t)", lw=2, color=:orange, markershape=:square, markerstrokewidth=1, markersize=3, line=:solid, marker=:auto, title="Prediction: 500 steps", xlabel="t", ylabel="y₂(t)")
# display(plt1)
# end

# mean_acfs_pred = mean(acfs_pred, dims=2)[:]
# std_acfs_pred = std(acfs_pred, dims=2)[:]

# mean_acfs_true = mean(acfs_true, dims=2)[:]
# std_acfs_true = std(acfs_true, dims=2)[:]




# gr()  # Assicurati che il backend sia impostato

# t_plot = t_short[1:100]

# # Vettori 1D
# mean_acfs_pred_vec = mean_acfs_pred[:]
# std_acfs_pred_vec = std_acfs_pred[:]

# mean_acfs_true_vec = mean_acfs_true[:]
# std_acfs_true_vec = std_acfs_true[:]

# # Plot Predicted
# plt_acfs = plot(
#     t_plot, mean_acfs_pred_vec;
#     ribbon = std_acfs_pred_vec,
#     label = "Predicted",
#     lw = 2,
#     color = :orange,
#     line = :solid,
#     marker = :square,
#     markersize = 3,
#     markerstrokewidth = 1,
#     markercolor = :orange,
#     markerstrokecolor = :black
# )

# # Plot Observed
# plot!(
#     plt_acfs, t_plot, mean_acfs_true_vec;
#     ribbon = std_acfs_true_vec,
#     label = "Observed",
#     lw = 2,
#     color = :blue,
#     line = :solid,
#     marker = :circle,
#     markersize = 3,
#     markerstrokewidth = 1,
#     markercolor = :blue,
#     markerstrokecolor = :black
# )




# plotlyjs()
# kde_pred_short = kde(y_pred_short)
# kde_obs_y2_short = kde(y_true_short)


# plot_kde_short = plot(kde_pred_short.x, kde_pred_short.density; label = "prediction", lw=2, color = :orange)
# plot!(plot_kde_short, kde_obs_y2_short.x, kde_obs_y2_short.density; label = "observations", lw=2, color = :blue)

# # Numero di bin condiviso per confronto coerente
# nbins = 100

# plot_hist = Plots.histogram(
#     y_pred_short;
#     bins = nbins,
#     normalize = true,
#     label = "Prediction",
#     lw = 0.5,
#     opacity = 0.5,
#     color = :orange,
# )

# Plots.histogram!(
#     plot_hist, y_true_short;
#     bins = nbins,
#     normalize = true,
#     label = "Observations",
#     lw = 0.5,
#     opacity = 0.5,
#     color = :blue,
# )




# display(plt1)
# display(plot_kde_short)

# #evaluate accuracy for short term prediction of the fast variable y2
# N_traj = 200

# lags = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]  # Different lags to evaluate

# rmses_vs_lags = Float64[]  # Array to store RMSE for each lag
# var_rmses = Float64[]  # Array to store standard deviations of RMSEs
# for lag in lags

#     rmses_NODE = Float64[] #array to store squared differences
    
#     for n in 1:N_traj
#         j = rand(1:(size(Z_val, 2)))
#         if j + 500 > size(Z_val, 2)
#             println("Invalid value of index j")
#             continue
#         end
#         u0 = Z_val[:, j] #set initial condition
#         t = collect(0.0f0:dt:dt*lag)
#         tspan = (t[1], t[end])
#         pred = predict_with_model(u0, model_trained, tspan, t)
#         y_pred = pred[1, :]
#         y_true = Z_val[1, j:10:(j+(10*lag))]

#         #compute RMSE
#         rmse = sqrt(mean((y_pred .- y_true).^2))
#         push!(rmses_NODE, rmse)
#     end

#     push!(rmses_vs_lags, mean(rmses_NODE))
#     push!(var_rmses, std(rmses_NODE))

# end


# plotlyjs()

# plt = plot(
#     lags .* dt, rmses_vs_lags;
#     ribbon=var_rmses,
#     label="RMSE NODE",
#     lw=2,
#     xlabel="Prediction Lag",
#     ylabel="RMSE",
#     title="RMSE vs Forecast Horizon",
#     legend=:topleft,
#     size=(800, 300)
# )


# scatter!(
#     lags .* dt, rmses_vs_lags;
#     marker=:circle,
#     markersize=5,
#     markerstrokecolor=:black,
#     markerstrokewidth=1.5,
#     markercolor=:black,
#     label=""
# )


# scatter!(
#     lags .* dt, rmses_vs_lags;
#     marker=:circle,
#     markersize=3,
#     markercolor=palette(:auto)[1],  # usa il primo colore della palette attuale
#     label=""
# )

# display(plt)


