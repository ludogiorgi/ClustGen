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
using GLMakie

GLMakie.activate!()

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
# const λ_short = 5.0f0  # peso per il contributo short-term

# function loss_neuralode(p)
#     loss = 0.0f0
#     for i in 1:100
#         u = data_sample[rand(1:length(data_sample))]
#         pred = predict_neuralode(u[:, 1], p, tspan, t)

#         full_loss = sum(abs2, (u[:, 2:end] .- pred[:, 1:end]) * weights)

#         short_steps = min(40, size(pred, 2))
#         short_loss = sum(abs2, u[:, 2:short_steps+1] .- pred[:, 1:short_steps])

#         loss += full_loss + λ_short * short_loss
#     end
#     return loss / 100
# end


#==================== EXTRACT FAST SIGNAL FROM SLOW SIGNAL ====================#
function estimate_y2(x::Vector{Float64}, Φ::Float64, Σ::Float64, dt::Float64)
    N = length(x) - 1
    y2_estimated = zeros(Float64, N)
    for n in 1:N
        dx_dt = (x[n+1] - x[n]) / dt
        s = score_clustered_xt(x[n], n * dt)
        y2_estimated[n] = (dx_dt - s[1]) / Σ
    end
    return y2_estimated
end


#==================== NORMALIZE TIME SERIRES ====================#
function normalize_time_series(obs)
    mean_obs = mean(obs)
    sigma_obs = std(obs)
    return (obs .- mean_obs) ./ sigma_obs
end

#==================== DETECT CRITICAL TRANSITIONS ====================#
function detect_transitions(x::AbstractArray; threshold=0.8, window=50, min_spacing=100)
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
n_steps = 60
dt_training = 0.001f0
t = collect(0.0f0:dt_training:dt_training*(n_steps - 1))
tspan = (t[1], t[end])
Nsteps = 10000000
t_full = collect(0:dt:(Nsteps-1)*dt)
f(x, t) = F(x, t, σ_value, ε)
obs_nn = evolve(randn(4), dt, Nsteps, f, sigma; timestepper=:rk4, resolution=1)
M = mean(obs_nn, dims=2)[1]
S = std(obs_nn, dims=2)[1]
obs = (obs_nn[1:1,:] .- M) ./ S
obs_signal = obs[1, :]
kde_obs = kde(obs[1, :])

y2_obs = Float32.(obs_nn[3, :])

# Normalizza
M_y2_obs = mean(y2_obs)
S_y2_obs = std(y2_obs)
y2_obs_norm = (y2_obs .- M_y2_obs) ./ S_y2_obs


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

obs_train = obs_signal[1:Ntrain]    #8M steps
obs_val = obs_signal[Ntrain+1:end]  #2M steps
y2_obs_train_norm = y2_obs_norm[1:Ntrain]    #8M steps

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
plotlyjs()
p_score = Plots.scatter(centers_sorted, scores; color=:blue, alpha=0.2, label="Cluster centers",
    xlims=(-1.3, 1.3), ylims=(-5, 5), xlabel="𝑥", ylabel="Score(𝑥)", title="Score Function Estimate")
Plots.plot!(p_score, xax, interpolated_score; label="NN interpolation", linewidth=2, color=:red)
Plots.plot!(p_score, xax, true_score; label="Score analytic", linewidth=2, color=:lime)

display(p_score)

# ------------------------
# 7. Phi calculation
# ------------------------
#rate matrix
dt = 0.01
Q = generator(labels; dt=dt)*0.29
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
score_clustered_xt(x, t) = Φ * score_clustered(x)
autocov_y2 = autocovariance(y2_obs_norm; timesteps=1000)
plotlyjs()
Plots.plot(autocov_y2, label="Autocovariance of y2", xlabel="Lag", ylabel="Autocovariance",
    title="Autocovariance of the estimated y2 signal", linewidth=2)

########## Estimate y2 from the slow variable x ##########
dt=0.001
Σ_rescaled = Σ / sqrt(2*1.5)  # Rescale Σ for the score function
#estimate new safe values for x to extract y(t)
function poly_tail_score(x; D_eff=D_eff)
    return 2 * (x - x^3) / D_eff
end

function score_extended(x)
    if x ≥ -1.20 && x ≤ 1.08
        return score_clustered(x)
    else
        return poly_tail_score(x)
    end
end

score_extended_xt(x,t) = Φ * score_extended(x)

function estimate_y2(x::Vector{Float64}, Φ::Float64, Σ::Float64, dt::Float64)
    N = length(x) - 1
    y2_estimated = zeros(Float64, N)
    for n in 1:N
        dx_dt = (x[n+1] - x[n]) / dt
        s = score_extended_xt(x[n], n * dt)
        y2_estimated[n] = (dx_dt - s[1]) / Σ
    end
    return y2_estimated
end


#estimate y2 training set and y2 validation set from obs_train and obs_validation
y2_x_train = estimate_y2(obs_train, Φ, Σ_rescaled, dt)
y2_x_val = estimate_y2(obs_val, Φ, Σ_rescaled, dt)


#normalize training set and validation set
y2x_t_norm = normalize_time_series(y2_x_train)
y2x_v_norm = normalize_time_series(y2_x_val)

# Plot the estimated y2
plotlyjs()
plt_fast_sig = Plots.plot(y2x_t_norm[1:100000], label="Estimated y2", xlabel="Time Step", ylabel="Normalized y2",
    title="Estimated y2 from Score Function", lw=2, color=:red)
Plots.plot!(plt_fast_sig, y2_obs_train_norm[1:100000], label="Original y2", lw=2, color=:blue)

kde_fast_sig = kde(y2x_t_norm)
kde_signal_norm = kde(y2_obs_train_norm)
plotlyjs()
plt_kde = Plots.plot(kde_fast_sig.x, kde_fast_sig.density, label="Estimated y2 PDF", lw=2, color=:red)
Plots.plot!(plt_kde, kde_signal_norm.x, kde_signal_norm.density, label="Original y2 PDF", lw=2, color=:blue,
    title="PDF of Estimated y2 vs Original y2",
    xlabel="y2", ylabel="Density")

# ------------------------
# 8. Construct delay embedding
# ------------------------
acf_y2x_t_norm = autocovariance(y2x_t_norm; timesteps=1000)
#acf_y2x_v_norm = autocovariance(y2x_v_norm; timesteps=1000)
plotlyjs()
Plots.plot(acf_y2x_t_norm, label="Autocovariance of fast signal", xlabel="Lag", ylabel="Autocovariance",
    title="Autocovariance of the estimated y2 signal", linewidth=2)
Plots.plot!(autocov_y2, label="Autocovariance of y2", xlabel="Lag", ylabel="Autocovariance",
    title="Autocovariance of the estimated y2 signal", linewidth=2)

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

τ_opt, acf = estimate_tau(y2x_t_norm, dt_training)

@info "Scelta ottimale di τ ≈ $(round(τ_opt, digits=4))"
τ = 0.25*τ_opt  
q = round(Int, τ / dt)

Z_train = Float32.(delay_embedding(y2x_t_norm; τ=τ, m=m))
Y_embed = Float32.(delay_embedding(y2x_v_norm; τ=τ, m=m))  # (m, N)
X_cut = Float32.(obs_val[(2 + (m-1) * q):end])                   # (N,)
Z_val = vcat(X_cut', Y_embed)                              # (m+1, N)

# ------------------------
# 9. Batching for NODE training
# ------------------------

batch_size = n_steps + 1
n_batches = 2000
data_sample = gen_batches(Z_train, batch_size, n_batches)


# ------------------------
# 10. Training the model
# ------------------------
p = flat_p0
opt = Optimisers.Adam(0.01)
state = Optimisers.setup(opt, p)
n_epochs = 1000
losses = []

weights = exp.(LinRange(0.0f0, -1.0f0, n_steps))

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
        mkpath("best_NODE_y_10")  # crea la cartella se non esiste
        @save joinpath("best_NODE_y_10", "model_epoch_$(epoch).bson") p
    end
end


# plot finale completo

println("Final Loss: ", losses[end])
plotlyjs()
plt_loss = plot(losses, xlabel="Epoch", ylabel="Loss", label="Training loss")
plot(plt_loss, losses, title="Loss vs Epoch", xlabel="Epoch", ylabel="Loss", label="Training loss")

# ------------------------
# 7. Plot predictions: Focus on short-term predictions
# ------------------------
@load "/Users/giuliodelfelice/Desktop/MIT/MODELLO TRAINATO CHE ANDAVA ABBASTANZA BENE CON LA y2 estratta dalla x/model_epoch_500.bson" p

@load "/Users/giuliodelfelice/Desktop/MIT/ClustGen/ModelsTrained/best_NODE_y_13/model_epoch_1000.bson" p
model_trained = re(p)

acfs_pred = Matrix{Float64}(undef, 100, 10)
acfs_true = Matrix{Float64}(undef, 100, 10)

y_pred_short = nothing
y_true_short = nothing

dt = 0.01
function predict_with_model(u0, model, tspan, t)
    function dudt!(du, u, _, t)
        du .= model(u)
    end
    prob = ODEProblem(dudt!, u0, tspan)
    sol = solve(prob, Tsit5(), saveat=t)
    return hcat(sol.u...)
end

for n in 1:10
    # First 500 steps prediction vs truth
    j = rand(1:size(Y_embed, 2))
    u0 = Y_embed[:, j]
    t_short = collect(0.0f0:dt:dt*9000)
    tspan_short = (t_short[1], t_short[end])


    pred_short = predict_with_model(u0, model_trained, tspan_short, t_short)
    y_pred_short = pred_short[1, :]
    #y_pred_short = normalize_time_series(y_pred_short)

    y_true_short = Y_embed[1, j:10:(j + 10*9000)]

    acf_y_pred_short = autocovariance(y_pred_short, timesteps = 100)
    acf_y_true_short = autocovariance(y_true_short, timesteps = 100)

    # Inserisci nella colonna n-esima
    acfs_pred[:, n] .= acf_y_pred_short
    acfs_true[:, n] .= acf_y_true_short


    plotlyjs()

    plt1 = Plots.plot(t_short[1:250], y_true_short[1:250]; label="True y₂(t)", lw=2, color=:blue, markershape=:square, markerstrokewidth=1, markersize=3, line=:solid, marker=:auto)

    Plots.plot!(plt1, t_short[1:250], y_pred_short[1:250]; label="Predicted y₂(t)", lw=2, color=:orange, markershape=:square, markerstrokewidth=1, markersize=3, line=:solid, marker=:auto, title="Prediction: 500 steps", xlabel="t", ylabel="y₂(t)")
    display(plt1)

end


mean_acfs_pred = mean(acfs_pred, dims=2)[:]
std_acfs_pred = std(acfs_pred, dims=2)[:]

mean_acfs_true = mean(acfs_true, dims=2)[:]
std_acfs_true = std(acfs_true, dims=2)[:]




gr()  # Assicurati che il backend sia impostato



t_short = collect(0.0f0:dt:dt*10000)
tspan_short = (t_short[1], t_short[end])
t_plot = t_short[1:100]

max_j = size(Y_embed, 2) - 10 * 10000
j = rand(1:max_j)
u0 = Y_embed[:, j]
pred_short = predict_with_model(u0, model_trained, tspan_short, t_short)
y_pred_short = pred_short[1, :]
y_pred_hist = normalize_time_series(y_pred_short)
y_true_hist = Y_embed[1, j:10:(j + 10*10000)]


# Vettori 1D
mean_acfs_pred_vec = mean_acfs_pred[:]
std_acfs_pred_vec = std_acfs_pred[:]

mean_acfs_true_vec = mean_acfs_true[:]
std_acfs_true_vec = std_acfs_true[:]

# Plot Predicted
plt_acfs = Plots.plot(
    t_plot, mean_acfs_pred_vec;
    ribbon = std_acfs_pred_vec,
    label = "Predicted",
    lw = 2,
    color = :orange,
    line = :solid,
    marker = :square,
    markersize = 3,
    markerstrokewidth = 1,
    markercolor = :orange,
    markerstrokecolor = :black
)

# Plot Observed
Plots.plot!(
    plt_acfs, t_plot, mean_acfs_true_vec;
    ribbon = std_acfs_true_vec,
    label = "Observed",
    lw = 2,
    color = :blue,
    line = :solid,
    marker = :circle,
    markersize = 3,
    markerstrokewidth = 1,
    markercolor = :blue,
    markerstrokecolor = :black
)



plotlyjs()
kde_pred_short = kde(y_pred_short)
kde_obs_y2_short = kde(y_true_short)


plot_kde_short = Plots.plot(kde_pred_short.x, kde_pred_short.density; label = "prediction", lw=2, color = :orange)
Plots.plot!(plot_kde_short, kde_obs_y2_short.x, kde_obs_y2_short.density; label = "observations", lw=2, color = :blue)

# Numero di bin condiviso per confronto coerente - ridotto a metà per raddoppiare la dimensione
nbins = 50

# Calcola range comune per assicurare bins identici
range_min = min(minimum(y_pred_hist), minimum(y_true_hist))
range_max = max(maximum(y_pred_hist), maximum(y_true_hist))
bin_edges = range(range_min, range_max, length=nbins+1)

plot_hist = Plots.histogram(
    y_pred_hist;
    bins = bin_edges,
    normalize = :probability,  # usa :probability invece di true
    label = "Prediction",
    lw = 0.5,
    alpha = 0.5,  # usa alpha invece di opacity
    color = :orange,
)

Plots.histogram!(
    plot_hist, y_true_hist;
    bins = bin_edges,
    normalize = :probability,  # usa :probability invece di true
    label = "Observations",
    lw = 0.5,
    alpha = 0.5,  # usa alpha invece di opacity
    color = :blue,
)




#evaluate accuracy for short term prediction of the fast variable y2
N_traj = 200

lags = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]  # Different lags to evaluate

rmses_vs_lags = Float64[]  # Array to store RMSE for each lag
var_rmses = Float64[]  # Array to store standard deviations of RMSEs
for lag in lags

    rmses_NODE = Float64[] #array to store squared differences
    
    for n in 1:N_traj
        j = rand(1:(size(Y_embed, 2)))
        if j + 500 > size(Y_embed, 2)
            println("Invalid value of index j")
            continue
        end
        u0 = Y_embed[:, j] #set initial condition
        t = collect(0.0f0:dt:dt*lag)
        tspan = (t[1], t[end])
        pred = predict_with_model(u0, model_trained, tspan, t)
        y_pred = pred[1, :]
        y_true = Y_embed[1, j:10:(j+(10*lag))]

        #compute RMSE
        rmse = sqrt(mean((y_pred .- y_true).^2))
        push!(rmses_NODE, rmse)
    end

    push!(rmses_vs_lags, mean(rmses_NODE))
    push!(var_rmses, std(rmses_NODE))

end


plotlyjs()

plt = plot(
    lags .* dt, rmses_vs_lags;
    ribbon=var_rmses,
    label="RMSE NODE",
    lw=2,
    xlabel="Prediction Lag",
    ylabel="RMSE",
    title="RMSE vs Forecast Horizon",
    legend=:topleft,
    size=(800, 300)
)


scatter!(
    lags .* dt, rmses_vs_lags;
    marker=:circle,
    markersize=5,
    markerstrokecolor=:black,
    markerstrokewidth=1.5,
    markercolor=:black,
    label=""
)


scatter!(
    lags .* dt, rmses_vs_lags;
    marker=:circle,
    markersize=3,
    markercolor=:deepskyblue,
    label=""
)

display(plt)

# === score_extended universale ===
function score_extended(x::Real)
    if x ≥ -1.20 && x ≤ 1.08
        return score_clustered(x)
    else
        return 2 * (x - x^3) / D_eff
    end
end

function score_extended(x::AbstractVector)
    return score_extended(x[1])
end

# ------------------------
# short term prediction for slow variable 
# ------------------------
plotlyjs()

#detect critical transitions in the validation set
Z_val_subsampled = Z_val[:, 1:10:end]

# sigma_Langevin(x, t) = Σ / sqrt(2*1.02)
# x0 = obs_val[1] 
# traj_langevin_test = evolve_chaos([x0], dt, length(y_pred_short), score_clustered_xt, sigma_Langevin, y_pred_short; timestepper=:euler, resolution=1)
# plot(obs_val[1:length(y_pred_short)], label="obs_val", title="Validation Signal")
# plot!(traj_langevin_test[1,:], label= "Langevin")

transitions = detect_transitions(Z_val_subsampled[1,:]; threshold=0.7, window=800, min_spacing=100)
if !isempty(transitions)
    @show transitions[1:min(10, end)]
else
    println("No transition found. Change threshold or window.")
end

sigma_Langevin(x, t) = Σ / sqrt(2*1.5)
#define list of time horizons for predictions
theta = [0.5, 0.8, 1.0, 1.5, 1.8, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0] #in seconds
dt
#convert the time horizon into number of steps
timesteps = round.(Int, theta ./ dt)


#initialize array for mse as a function of theta
rmses_vs_theta = Float64[]
var_rmses_x = Float64[]

for timestep in timesteps
    rmses = Float64[]
    for tau_n in transitions
        t_start = tau_n - timestep

        # Skip se siamo troppo all'inizio
        if t_start < 1 || tau_n + 10 > length(obs_val)
            continue
        end

        # x0 = valore della x al tempo t_start
        x0 = Z_val_subsampled[1, t_start]

    
        embedding_y = Z_val_subsampled[2:end, t_start]

        # Simula y_pred_short partendo da embedding_y
        t_short_temp = collect(0.0f0:dt:dt * (timestep + 10))
        tspan_short_temp = (t_short_temp[1], t_short_temp[end])
        pred_short_temp = predict_with_model(embedding_y, model_trained, tspan_short_temp, t_short_temp)
        y_pred_short_temp = pred_short_temp[1, :]

        # Evolvi x con Langevin
        traj_langevin = evolve_chaos([x0], dt, timestep + 10, score_extended_xt, sigma_Langevin, y_pred_short_temp;
                                     timestepper = :euler, resolution = 1)
        pred = traj_langevin[1, :]

        # Dato reale
        real = Z_val_subsampled[1, t_start:(tau_n + 10)]

        @assert length(pred) == length(real)

        rmse = sqrt(mean((pred .- real).^2))
        push!(rmses, rmse)
    end
    push!(rmses_vs_theta, mean(rmses))
    push!(var_rmses_x, std(rmses))

end

#rmses_vs_theta[1] = 0.0  # Set RMSE for theta=0 to 0

plt_rmse_x = plot(
    theta, rmses_vs_theta;
    ribbon=var_rmses_x,
    label="RMSE langevin",
    lw=2,
    xlabel="Prediction Lag",
    ylabel="RMSE",
    title="RMSE vs Forecast Horizon",
    legend=:topleft,
    size=(800, 300),
)

# Secondo: aggiungi sopra i **soli marker** della curva centrale
scatter!(
    plt_rmse_x, theta, rmses_vs_theta;
    marker=:circle,
    markersize=5,
    label="",
    color=:black
)

scatter!(
    plt_rmse_x, theta, rmses_vs_theta;
    marker=:circle,
    markersize=4,
    markercolor=palette(:auto)[1],  # usa il primo colore della palette attuale
    label=""
)
# 
plotlyjs()


for k in 1:length(transitions)
    tau_n = transitions[k]
    t_start = tau_n - timestep

    # Skip se siamo troppo all'inizio
    if t_start < 1 || (tau_n + 500) > length(Z_val_subsampled[1,:])
        println("Skipping timestep $(k) — out of range")
        continue
    end

    x0 = Z_val_subsampled[1, t_start]
    z0 = Z_val_subsampled[2:end, t_start]

    t_plot_trans = collect(0.0f0:dt:dt * (timestep + 500))
    tspan_plot_trans = (t_plot_trans[1], t_plot_trans[end])
    T = length(t_plot_trans)

    #check dimension of data
    if t_start + T - 1 > size(Z_val_subsampled, 2)
        @warn "Skipping panel $k: out of range (t_start + T - 1 > length)"
        continue
    end

    # Simula ensemble x(t)
    trajs_x = Matrix{Float64}(undef, N_traj, T)
    for n in 1:N_traj
        rng = MersenneTwister(0xBEEF + 10_000*k + n)
        eta = σ0 .* randn(rng, eltype(z0), length(z0))
        z0n = z0 .+ eta

        #integrate NODE with perturbed initial condition
        pred_y = predict_with_model(z0n, model_trained, tspan_plot_trans, t_plot_trans)
        y_pred_trans = pred_y[1, :]
        traj_langevin = evolve_chaos([x0], dt, timestep + 500, score_extended_xt, sigma_Langevin, y_pred_trans;
                                     timestepper = :euler, resolution = 1)
        trajs_x[n, :] .= Float64.(traj_langevin[1, :])
    end

    # Statistics
    mean_traj = vec(mean(trajs_x, dims=1))
    std_traj  = vec(std(trajs_x,  dims=1))
    real_traj = Z_val_subsampled[1, t_start:(tau_n + 500)]
    t_plot_x  = dt .* (0:(length(real_traj) - 1))


    # Plot all trajectories with different colors
    ens_colors = palette(:tab20, N_traj)  # palette 
    for n in 1:N_traj
        plot!(p_ensemble[k], t_plot_x, trajs_x[n, :];
              seriestype=:path, lw=1.2, linealpha=0.35, color=ens_colors[n])
    end

    # Ribbon ±1σ 
    plot!(p_ensemble[k], t_plot_x, mean_traj;
          ribbon=std_traj, fillalpha=0.25, color=:red, lw=2)

    # Real trajectory vs ensemble average 
    plot!(p_ensemble[k], t_plot_x, real_traj, color = :blue, linewidth = 2.5, label = "Real")
    plot!(p_ensemble[k], t_plot_x, mean_traj_models, color = :red, linewidth = 2.5, label = "Ensemble Mean")

    if k == 1
        axislegend(p_ensemble[k], position = :rt)
    end
end

display(p_ensemble)



# Ensemble average usando diversi modelli trainati
using Random, Plots, Statistics
plotlyjs()

# Funzione per trovare l'ultimo file di una cartella
function get_latest_model_file(folder_path::String)
    files = readdir(folder_path)
    model_files = filter(f -> startswith(f, "model_epoch_") && endswith(f, ".bson"), files)
    if isempty(model_files)
        error("No model files found in $folder_path")
    end
    # Estrai i numeri delle epoche e trova il massimo
    epochs = [parse(Int, split(split(f, "_")[3], ".")[1]) for f in model_files]
    max_epoch_idx = argmax(epochs)
    return joinpath(folder_path, model_files[max_epoch_idx])
end

# Lista delle cartelle dei modelli
model_folders = ["best_NODE_y","best_NODE_y_2", "best_NODE_y_3", "best_NODE_y_4", "best_NODE_y_7", "best_NODE_y_9", "best_NODE_y_10", "best_NODE_y_11", "best_NODE_y_12", "best_NODE_y_15"]

base_path = "/Users/giuliodelfelice/Desktop/MIT/ClustGen/ModelsTrained"

# Carica tutti i modelli
trained_models = []
for folder in model_folders
    folder_path = joinpath(base_path, folder)
    if isdir(folder_path)
        try
            model_file = get_latest_model_file(folder_path)
            println("Loading model from: $model_file")
            @load model_file p
            model = re(p)
            push!(trained_models, model)
        catch e
            println("Error loading model from $folder: $e")
        end
    else
        println("Folder not found: $folder_path")
    end
end

println("Loaded $(length(trained_models)) models")


# --- GLMakie loop per le transizioni - SALVATAGGIO PNG ---
# Crea cartella per i plot sul Desktop
plot_folder = "/Users/giuliodelfelice/Desktop/Plot_Forecast_GLMakie"
mkpath(plot_folder)  # Crea la cartella se non esiste

for (transition_idx, transition) in enumerate(transitions)
    # parametri di plotting (stessi valori che usavi prima)
    theta_wanted = [0.5, 0.8, 1.0, 2.0, 5.0]
    idxs = map(tw -> findmin(abs.(theta .- tw))[2], theta_wanted)
    ts_to_plot = timesteps[idxs]
    theta_to_plot = theta[idxs]
    tau_n = transition

    N_models = length(trained_models)

    # Crea figura GLMakie con 5 pannelli verticali
    fig = Figure(resolution=(900, 1300))
    
    # Aggiungi titolo generale alla figura usando Label
    Label(fig[0, :], "Transition $(transition_idx) - Model Ensemble Predictions", 
          fontsize=16, tellwidth=false)
    
    for (k, (timestep, thetas)) in enumerate(zip(ts_to_plot, theta_to_plot))
        t_start = tau_n - timestep
        if t_start < 1 || (tau_n + 500) > length(Z_val_subsampled[1, :])
            println("Skipping timestep $(k) — out of range")
            continue
        end

        x0 = Z_val_subsampled[1, t_start]
        z0 = Z_val_subsampled[2:end, t_start]

        t_plot_trans = collect(0.0f0:dt:dt * (timestep + 500))
        tspan_plot_trans = (t_plot_trans[1], t_plot_trans[end])
        T = length(t_plot_trans)

        if t_start + T - 1 > size(Z_val_subsampled, 2)
            @warn "Skipping panel $k: out of range (t_start + T - 1 > length)"
            continue
        end

        # Simula ensemble con i diversi modelli
        trajs_x_models = Matrix{Float64}(undef, N_models, T)
        for n in 1:N_models
            pred_y = predict_with_model(z0, trained_models[n], tspan_plot_trans, t_plot_trans)
            y_pred_trans = pred_y[1, :]

            traj_langevin = evolve_chaos([x0], dt, timestep + 500, score_extended_xt, sigma_Langevin, y_pred_trans;
                                         timestepper = :euler, resolution = 1)
            trajs_x_models[n, :] .= Float64.(traj_langevin[1, :])
        end

        # Statistiche
        mean_traj_models = vec(mean(trajs_x_models, dims=1))
        std_traj_models = vec(std(trajs_x_models, dims=1))
        real_traj = Z_val_subsampled[1, t_start:(tau_n + 500)]
        t_plot_x = dt .* (0:(length(real_traj) - 1))


        # Subplot GLMakie
        ax = Axis(fig[k, 1],
                  title = "θ = $(thetas) s (Model Ensemble)",
                  xlabel = k == length(ts_to_plot) ? "Time (s)" : "",
                  ylabel = "x(t)")

        # colori predefiniti
        # disegna traiettorie individuali
        for n in 1:N_models
            plot!(ax, t_plot_x, trajs_x_models[n, :];
                  seriestype=:path, lw=1.2, linealpha=0.35, color=ens_colors[n])
        end

        # ribbon ±1σ (band)
        plot!(ax, t_plot_x, mean_traj;
          ribbon=std_traj, fillalpha=0.25, color=:red, lw=2)

        # traiettoria reale e media ensemble
        plot!(ax, t_plot_x, real_traj, color = :blue, linewidth = 2.5, label = "Real")
        plot!(ax, t_plot_x, mean_traj_models, color = :red, linewidth = 2.5, label = "Ensemble Mean")

        if k == 1
            axislegend(ax, position = :rt)
        end
    end
    
    # Salva la figura come PNG
    filename = "transition_$(transition_idx)_ensemble_forecast.png"
    filepath = joinpath(plot_folder, filename)
    save(filepath, fig)
    
    println("Saved figure for transition $(transition_idx) to: $(filepath)")
end

println("All plots saved to: $(plot_folder)")

# RMSE ensemble average sui modelli multipli - GLMakie version
GLMakie.activate!()

# Assicurati che i modelli siano caricati
if length(trained_models) == 0
    println("Warning: No models loaded. Skipping ensemble RMSE calculation.")
else
    println("Calculating ensemble RMSE with $(length(trained_models)) models")
    
    # Define time horizons and convert to timesteps
    theta_ensemble = [0.5, 0.8, 1.0, 1.5, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
    timesteps_ensemble = round.(Int, theta_ensemble ./ dt)
    
    # Initialize arrays for ensemble RMSE
    rmses_vs_theta_ensemble = Float64[]
    var_rmses_ensemble = Float64[]
    
    N_models = length(trained_models)
    
    for timestep in timesteps_ensemble
        # Array to store RMSE for each model
        rmses_all_models = Float64[]
        
        for model_idx in 1:N_models
            model = trained_models[model_idx]
            rmses_this_model = Float64[]
            
            # Calculate RMSE for this model across all transitions
            for tau_n in transitions
                t_start = tau_n - timestep
                
                # Skip if out of bounds
                if t_start < 1 || tau_n + 10 > length(Z_val_subsampled[1,:])
                    continue
                end
                
                # Initial conditions
                x0 = Z_val_subsampled[1, t_start]
                embedding_y = Z_val_subsampled[2:end, t_start]
                
                # Predict with this specific model
                t_short_temp = collect(0.0f0:dt:dt * (timestep + 10))
                tspan_short_temp = (t_short_temp[1], t_short_temp[end])
                
                try
                    pred_short_temp = predict_with_model(embedding_y, model, tspan_short_temp, t_short_temp)
                    y_pred_short_temp = pred_short_temp[1, :]
                    
                    # Evolve x using Langevin
                    traj_langevin = evolve_chaos([x0], dt, timestep + 10, score_extended_xt, sigma_Langevin, y_pred_short_temp;
                                                 timestepper = :euler, resolution = 1)
                    pred = traj_langevin[1, :]
                    
                    # Real trajectory
                    real = Z_val_subsampled[1, t_start:(tau_n + 10)]
                    
                    if length(pred) == length(real)
                        rmse = sqrt(mean((pred .- real).^2))
                        push!(rmses_this_model, rmse)
                    end
                catch e
                    println("Error with model $model_idx at transition $tau_n: $e")
                    continue
                end
            end
            
            # Average RMSE for this model across all transitions
            if !isempty(rmses_this_model)
                push!(rmses_all_models, mean(rmses_this_model))
            end
        end
        
        # Calculate ensemble statistics
        if !isempty(rmses_all_models)
            push!(rmses_vs_theta_ensemble, mean(rmses_all_models))
            push!(var_rmses_ensemble, std(rmses_all_models))
        else
            push!(rmses_vs_theta_ensemble, NaN)
            push!(var_rmses_ensemble, NaN)
        end
        
        println("Completed timestep $(timestep) (θ = $(timestep * dt) s): RMSE = $(rmses_vs_theta_ensemble[end])")
    end

end
    
    # GLMakie plot ensemble RMSE
    fig_rmse = Figure(resolution=(800, 400))
    ax_rmse = Axis(fig_rmse[1, 1],
                   xlabel="Prediction Lag (s)",
                   ylabel="RMSE",
                   title="Ensemble RMSE vs Forecast Horizon ($(N_models) models)")
    
    # Plot only first 6 points (same as original)
    x_data = theta_ensemble[1:7]
    y_data = rmses_vs_theta_ensemble[1:7]
    error_data = var_rmses_ensemble[1:7]
    
    # Ribbon (band) for ±std
    lower = y_data .- error_data
    upper = y_data .+ error_data
    band!(ax_rmse, x_data, lower, upper, color=(:deepskyblue, 0.3))
    
    # Main line
    lines!(ax_rmse, x_data, y_data, color=:deepskyblue, linewidth=2, 
           label="RMSE Ensemble ($(N_models) models)")
    
    # Scatter points - usa GLMakie.scatter! per essere esplicito
    GLMakie.scatter!(ax_rmse, x_data, y_data, color=:black, markersize=8, 
                     strokewidth=1.5, strokecolor=:black)
    GLMakie.scatter!(ax_rmse, x_data, y_data, color=:deepskyblue, markersize=5)
    
    # Legend
    axislegend(ax_rmse, position=:lt)
    
    # Save the figure
    rmse_filename = "ensemble_rmse_forecast.png"
    rmse_filepath = joinpath(plot_folder, rmse_filename)
    save(rmse_filepath, fig_rmse)
    
    println("Saved RMSE plot to: $(rmse_filepath)")
    
    # Print summary statistics
    println("\n=== Ensemble RMSE Summary ===")
    for (i, θ) in enumerate(theta_ensemble)
        if !isnan(rmses_vs_theta_ensemble[i])
            println("θ = $(θ) s: RMSE = $(round(rmses_vs_theta_ensemble[i], digits=4)) ± $(round(var_rmses_ensemble[i], digits=4))")
        end
    end


# Plot delle transizioni specifiche con GLMakie
GLMakie.activate!()

# Transizioni da plottare
transition_indices = [29, 30, 40]
transition_names = ["Transition A", "Transition B", "Transition C"]

# Crea figura con layout orizzontale (1 riga, 3 colonne)
fig_transitions = Figure(resolution=(1200, 400))

for (i, trans_idx) in enumerate(transition_indices)
    if trans_idx <= length(transitions)
        transition_point = transitions[trans_idx]
        
        # Definisci finestra di 500 timesteps a sinistra e destra della transizione
        t_start = max(1, transition_point - 1000)
        t_end = min(length(Z_val_subsampled[1,:]), transition_point + 1000)
        
        # Estrai i dati della finestra
        time_window = (t_start:t_end)
        x_data = Z_val_subsampled[1, time_window]
        time_axis = dt .* (time_window .- transition_point)  # Tempo relativo alla transizione
        
        # Crea subplot
        ax = Axis(fig_transitions[1, i],
                  title = transition_names[i],
                  xlabel = "Time relative to transition (s)",
                  ylabel = "x(t)")
        
        # Plot della serie temporale
        lines!(ax, time_axis, x_data, color = :blue, linewidth = 2)
        
        # Linea verticale tratteggiata alla transizione (t=0)
        vlines!(ax, [0.0], color = :orange, linewidth = 2, linestyle = :dash)
        
        # Imposta limiti degli assi per centrare la transizione - usa GLMakie
        GLMakie.xlims!(ax, (-1000*dt, 1000*dt))
        
    else
        println("Warning: Transition index $trans_idx is out of range")
    end
end

# Salva la figura
transitions_filename = "specific_transitions_plot.png"
transitions_filepath = joinpath(plot_folder, transitions_filename)
save(transitions_filepath, fig_transitions)

println("Saved transitions plot to: $(transitions_filepath)")
println("Saved transitions plot to: $(transitions_filepath)")



transitions


# Mega plot con 3 colonne delle transizioni specifiche con GLMakie
GLMakie.activate!()

# Transizioni da plottare
transition_indices = [29, 30, 40]
transition_names = ["Transition A", "Transition B", "Transition C"]

# Parametri per forecast ensemble
theta_wanted = [0.5, 0.8, 1.0, 2.0, 5.0]
idxs = map(tw -> findmin(abs.(theta .- tw))[2], theta_wanted)
ts_to_plot = timesteps[idxs]
theta_to_plot = theta[idxs]

# Crea figura ottimizzata per A4 (3 colonne x 6 righe)
fig_mega = Figure(resolution=(2970, 2100))  # 10x scale per alta qualità

# Loop attraverso le 3 transizioni (colonne)
for (col, trans_idx) in enumerate(transition_indices)
    if trans_idx <= length(transitions)
        transition_point = transitions[trans_idx]
        tau_n = transition_point
        
        # ===== ROW 1: Transizione con finestra di 2000 timesteps =====
        t_start_window = max(1, transition_point - 1000)
        t_end_window = min(length(Z_val_subsampled[1,:]), transition_point + 1000)
        
        time_window = (t_start_window:t_end_window)
        x_data = Z_val_subsampled[1, time_window]
        time_axis = dt .* (time_window .- transition_point)
        
        ax_trans = Axis(fig_mega[1, col],
                       title = transition_names[col],
                       xlabel = col == 2 ? "Time relative to transition (s)" : "",
                       ylabel = "x(t)")
        
        lines!(ax_trans, time_axis, x_data, color = :blue, linewidth = 2)
        vlines!(ax_trans, [0.0], color = :orange, linewidth = 3, linestyle = :dash)
        GLMakie.xlims!(ax_trans, (-1000*dt, 1000*dt))
        
        # ===== ROWS 2-6: Ensemble forecasts per diversi theta =====
        N_models = length(trained_models)
        
        for (row_idx, (timestep, thetas)) in enumerate(zip(ts_to_plot, theta_to_plot))
            t_start = tau_n - timestep
            
            if t_start < 1 || (tau_n + 500) > length(Z_val_subsampled[1,:])
                continue
            end
            
            x0 = Z_val_subsampled[1, t_start]
            z0 = Z_val_subsampled[2:end, t_start]
            
            t_plot_trans = collect(0.0f0:dt:dt * (timestep + 500))
            tspan_plot_trans = (t_plot_trans[1], t_plot_trans[end])
            T = length(t_plot_trans)
            
            if t_start + T - 1 > size(Z_val_subsampled, 2)
                continue
            end
            
            # Simula ensemble con i diversi modelli
            trajs_x_models = Matrix{Float64}(undef, N_models, T)
            for n in 1:N_models
                pred_y = predict_with_model(z0, trained_models[n], tspan_plot_trans, t_plot_trans)
                y_pred_trans = pred_y[1, :]

                traj_langevin = evolve_chaos([x0], dt, timestep + 500, score_extended_xt, sigma_Langevin, y_pred_trans;
                                             timestepper = :euler, resolution = 1)
                trajs_x_models[n, :] .= Float64.(traj_langevin[1, :])
            end
            
            # Statistiche
            mean_traj_models = vec(mean(trajs_x_models, dims=1))
            std_traj_models = vec(std(trajs_x_models, dims=1))
            real_traj = Z_val_subsampled[1, t_start:(tau_n + 500)]
            t_plot_x = dt .* (0:(length(real_traj) - 1))
            
            # Subplot per forecast (righe 2-6)
            ax_forecast = Axis(fig_mega[row_idx + 1, col],
                              title = row_idx == 1 && col == 2 ? "θ = $(thetas) s" : (col == 1 ? "θ = $(thetas) s" : ""),
                              xlabel = row_idx == 5 ? "Time (s)" : "",
                              ylabel = col == 1 ? "x(t)" : "")
            
            # Colori predefiniti per i modelli
            colors = Makie.wong_colors()
            
            # Disegna traiettorie individuali dei modelli
            for n in 1:N_models
                lines!(ax_forecast, t_plot_x, trajs_x_models[n, :],
                       color = (colors[mod1(n, length(colors))], 0.35),
                       linewidth = 1)
            end
            
            # Banda ±1σ
            lower = mean_traj_models .- std_traj_models
            upper = mean_traj_models .+ std_traj_models
            band!(ax_forecast, t_plot_x, lower, upper, color = (:red, 0.18))
            
            # Traiettoria reale e media ensemble
            lines!(ax_forecast, t_plot_x, real_traj, color = :blue, linewidth = 2.5, 
                   label = row_idx == 1 && col == 1 ? "Real" : "")
            lines!(ax_forecast, t_plot_x, mean_traj_models, color = :red, linewidth = 2.5,
                   label = row_idx == 1 && col == 1 ? "Ensemble Mean" : "")
            
            # Legenda solo nel primo pannello forecast
            if row_idx == 1 && col == 1
                axislegend(ax_forecast, position = :rt)
            end
        end
    end
end

# Ottimizza spaziatura per minimizzare spazio vuoto
rowgap!(fig_mega.layout, 15)  # Spazio verticale tra righe
colgap!(fig_mega.layout, 20)  # Spazio orizzontale tra colonne

# Salva la figura mega
mega_filename = "mega_transitions_ensemble_forecast_2.png"
mega_filepath = joinpath(plot_folder, mega_filename)
save(mega_filepath, fig_mega, px_per_unit = 3)  # Alta risoluzione per stampa A4

println("Saved mega plot to: $(mega_filepath)")






# ======== PLOT PER LA X ========= #

Z_x = Float32.(delay_embedding(obs_val, τ=τ, m=m))
Z_x_subsampled = Z_x[:, 1:10:end]
dt = 0.01
@load "/Users/giuliodelfelice/Desktop/MIT/ClustGen/best_NODE_x/model_epoch_1000.bson" p
model_trained_x = re(p)

# Mega plot con 3 colonne delle transizioni specifiche con GLMakie
GLMakie.activate!()

# Transizioni da plottare
transition_indices = [13, 21, 41]
transition_names = ["Transition A", "Transition B", "Transition C"]

# Parametri per forecast
theta_wanted_x = [0.5, 0.8, 1.0, 1.5, 2.0, 5.0, 10.0]
idxs_x = map(tw -> findmin(abs.(theta .- tw))[2], theta_wanted_x)
ts_to_plot_x = timesteps[idxs_x]
theta_to_plot_x = theta[idxs_x]

# Crea figura 2x3 (ma la seconda riga occupa tutto lo spazio)
fig_mega_x = Figure(resolution=(1800, 800))

# ===== PRIMA RIGA: Traiettorie per lag 0.5s =====
lag_idx = 1  # 0.5s
timestep_05 = ts_to_plot_x[lag_idx]

for (col, trans_idx) in enumerate(transition_indices)
    if trans_idx <= length(transitions)
        tau_n = transitions[trans_idx]
        t_start = tau_n - timestep_05
        
        if t_start < 1 || tau_n + 500 > length(obs_val)
            continue
        end
        
        # Predizione con NODE_x
        u0 = Z_x_subsampled[:, t_start]
        t = collect(0:dt:dt * (timestep_05 + 500))
        tspan = (0.0, t[end])
        sol_matrix = predict_with_model(u0, model_trained_x, tspan, t)
        pred = sol_matrix[1, :]
        
        # Dati reali
        start_idx = t_start
        end_idx = min(tau_n + 500, size(Z_x_subsampled, 2))
        real = Z_x_subsampled[1, start_idx:end_idx]
        
        # Sincronizza lunghezze
        min_length = min(length(pred), length(real))
        pred = pred[1:min_length]
        real = real[1:min_length]
        t_plot = dt .* (0:(min_length-1))
        
        # Plot prima riga
        ax = Axis(fig_mega_x[1, col],
                  title = transition_names[col],
                  xlabel = "",
                  ylabel = col == 1 ? "x(t)" : "")
        
        lines!(ax, t_plot, real, color = :blue, linewidth = 2, label = col == 1 ? "Real" : "")
        lines!(ax, t_plot, pred, color = :red, linewidth = 2, label = col == 1 ? "NODE_x" : "")
        
        if col == 1
            axislegend(ax, position = :rt)
        end
    end
end

# ===== SECONDA RIGA: RMSE vs lag mediato su tutte le transizioni =====
# Calcola RMSE per ogni lag mediato su tutte le transizioni
rmses_x_all = Float64[]
var_rmses_x_all = Float64[]

for timestep in ts_to_plot_x
    rmses_for_this_lag = Float64[]
    
    for trans_idx in 1:length(transitions)
        tau_n = transitions[trans_idx]
        t_start = tau_n - timestep
        
        if t_start < 1 || tau_n + 10 > length(obs_val)
            continue
        end
        
        # Predizione
        u0 = Z_x_subsampled[:, t_start]
        t = collect(0:dt:dt * (timestep + 10))
        tspan = (0.0, t[end])
        
        try
            sol_matrix = predict_with_model(u0, model_trained_x, tspan, t)
            pred = sol_matrix[1, :]
            
            # Reale
            start_idx = t_start
            end_idx = min(tau_n + 10, size(Z_x_subsampled, 2))
            real = Z_x_subsampled[1, start_idx:end_idx]

            if length(pred) == length(real)
                rmse = sqrt(mean((pred .- real).^2))
                push!(rmses_for_this_lag, rmse)
            end
        catch
            continue
        end
    end
    
    if !isempty(rmses_for_this_lag)
        push!(rmses_x_all, mean(rmses_for_this_lag))
        push!(var_rmses_x_all, std(rmses_for_this_lag))
    else
        push!(rmses_x_all, NaN)
        push!(var_rmses_x_all, NaN)
    end
end

# Plot seconda riga che occupa tutte e 3 le colonne
ax_rmse = Axis(fig_mega_x[2, 1:3],
               title = "RMSE vs Prediction Lag (averaged over all transitions)",
               xlabel = "Prediction Lag (s)",
               ylabel = "RMSE")

# Filtra valori validi
valid_idx = .!isnan.(rmses_x_all)
x_data = theta_to_plot_x[valid_idx]
y_data = rmses_x_all[valid_idx]
error_data = var_rmses_x_all[valid_idx]

if !isempty(y_data)
    # Ribbon per errore
    lower = y_data .- error_data
    upper = y_data .+ error_data
    band!(ax_rmse, x_data, lower, upper, color=(:deepskyblue, 0.3))
    
    # Linea principale
    lines!(ax_rmse, x_data, y_data, color = :deepskyblue, linewidth = 2)
    
    # Punti
    GLMakie.scatter!(ax_rmse, x_data, y_data, color = :black, markersize = 12)
    GLMakie.scatter!(ax_rmse, x_data, y_data, color = :deepskyblue, markersize = 9)
end

# Ottimizza spaziatura
rowgap!(fig_mega_x.layout, 20)
colgap!(fig_mega_x.layout, 15)

# Salva la figura
mega_x_filename = "mega_NODE_x_forecast.png"
mega_x_filepath = joinpath(plot_folder, mega_x_filename)
save(mega_x_filepath, fig_mega_x, px_per_unit = 3)

println("Saved NODE_x mega plot to: $(mega_x_filepath)")