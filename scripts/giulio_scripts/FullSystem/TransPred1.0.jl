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
score_clustered_xt(x, t) = Φ * score_clustered(x)
autocov_y2 = autocovariance(y2_obs_norm; timesteps=1000)
plotlyjs()
plot(autocov_y2, label="Autocovariance of y2", xlabel="Lag", ylabel="Autocovariance",
    title="Autocovariance of the estimated y2 signal", linewidth=2)

########## Estimate y2 from the slow variable x ##########
dt=0.001
Σ_rescaled = Σ / sqrt(2*1.5)  # Rescale Σ for the score function

#estimate y2 training set and y2 validation set from obs_train and obs_validation
y2_x_train = estimate_y2(obs_train, Φ, Σ_rescaled, dt)
y2_x_val = estimate_y2(obs_val, Φ, Σ_rescaled, dt)


#normalize training set and validation set
y2x_t_norm = normalize_time_series(y2_x_train)
y2x_v_norm = normalize_time_series(y2_x_val)

# Plot the estimated y2
plotlyjs()
plt_fast_sig = plot(y2x_t_norm[1:100000], label="Estimated y2", xlabel="Time Step", ylabel="Normalized y2",
    title="Estimated y2 from Score Function", lw=2, color=:red)
plot!(plt_fast_sig, y2_obs_train_norm[1:100000], label="Original y2", lw=2, color=:blue)

kde_fast_sig = kde(y2x_t_norm)
kde_signal_norm = kde(y2_obs_train_norm)
plotlyjs()
plt_kde = plot(kde_fast_sig.x, kde_fast_sig.density, label="Estimated y2 PDF", lw=2, color=:red)
plot!(plt_kde, kde_signal_norm.x, kde_signal_norm.density, label="Original y2 PDF", lw=2, color=:blue,
    title="PDF of Estimated y2 vs Original y2",
    xlabel="y2", ylabel="Density")

# ------------------------
# 8. Construct delay embedding
# ------------------------
acf_y2x_t_norm = autocovariance(y2x_t_norm; timesteps=1000)
#acf_y2x_v_norm = autocovariance(y2x_v_norm; timesteps=1000)
plotlyjs()
plot(acf_y2x_t_norm, label="Autocovariance of fast signal", xlabel="Lag", ylabel="Autocovariance",
    title="Autocovariance of the estimated y2 signal", linewidth=2)
plot!(autocov_y2, label="Autocovariance of y2", xlabel="Lag", ylabel="Autocovariance",
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

Z_train = Float32.(delay_embedding(y2x_t_norm; τ=τ, m=m))
Z_val = Float32.(delay_embedding(y2x_v_norm; τ=τ, m=m))
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
n_epochs = 500
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
        @save "model_epoch_$(epoch).bson" p
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

@load "/Users/giuliodelfelice/Desktop/MIT/ClustGen/model_epoch_500.bson" p
model_trained = re(p)

acfs_pred = Matrix{Float64}(undef, 100, 10)
acfs_true = Matrix{Float64}(undef, 100, 10)

for n in 1:10
    # First 500 steps prediction vs truth
    dt = 0.01
    j = rand(1:size(Z_val, 2))
    u0 = Z_val[:, j]
    t_short = collect(0.0f0:dt:dt*9000)
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
    y_pred_short = normalize_time_series(y_pred_short)

    y_true_short = Z_val[1, j:10:(j + 10*9000)]

    acf_y_pred_short = autocovariance(y_pred_short, timesteps = 100)
    acf_y_true_short = autocovariance(y_true_short, timesteps = 100)

    # Inserisci nella colonna n-esima
    acfs_pred[:, n] .= acf_y_pred_short
    acfs_true[:, n] .= acf_y_true_short


plotlyjs()

plt1 = plot(t_short[1:250], y_true_short[1:250]; label="True y₂(t)", lw=2, color=:blue, markershape=:square, markerstrokewidth=1, markersize=3, line=:solid, marker=:auto)

plot!(plt1, t_short[1:250], y_pred_short[1:250]; label="Predicted y₂(t)", lw=2, color=:orange, markershape=:square, markerstrokewidth=1, markersize=3, line=:solid, marker=:auto, title="Prediction: 500 steps", xlabel="t", ylabel="y₂(t)")
display(plt1)
end

mean_acfs_pred = mean(acfs_pred, dims=2)[:]
std_acfs_pred = std(acfs_pred, dims=2)[:]

mean_acfs_true = mean(acfs_true, dims=2)[:]
std_acfs_true = std(acfs_true, dims=2)[:]




gr()  # Assicurati che il backend sia impostato

t_plot = t_short[1:100]

# Vettori 1D
mean_acfs_pred_vec = mean_acfs_pred[:]
std_acfs_pred_vec = std_acfs_pred[:]

mean_acfs_true_vec = mean_acfs_true[:]
std_acfs_true_vec = std_acfs_true[:]

# Plot Predicted
plt_acfs = plot(
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
plot!(
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


plot_kde_short = plot(kde_pred_short.x, kde_pred_short.density; label = "prediction", lw=2, color = :orange)
plot!(plot_kde_short, kde_obs_y2_short.x, kde_obs_y2_short.density; label = "observations", lw=2, color = :blue)

# Numero di bin condiviso per confronto coerente
nbins = 100

plot_hist = Plots.histogram(
    y_pred_short;
    bins = nbins,
    normalize = true,
    label = "Prediction",
    lw = 0.5,
    opacity = 0.5,
    color = :orange,
)

Plots.histogram!(
    plot_hist, y_true_short;
    bins = nbins,
    normalize = true,
    label = "Observations",
    lw = 0.5,
    opacity = 0.5,
    color = :blue,
)




display(plt1)
display(plot_kde_short)

#evaluate accuracy for short term prediction of the fast variable y2
N_traj = 200

lags = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]  # Different lags to evaluate

rmses_vs_lags = Float64[]  # Array to store RMSE for each lag
var_rmses = Float64[]  # Array to store standard deviations of RMSEs
for lag in lags

    rmses_NODE = Float64[] #array to store squared differences
    
    for n in 1:N_traj
        j = rand(1:(size(Z_val, 2)))
        if j + 500 > size(Z_val, 2)
            println("Invalid value of index j")
            continue
        end
        u0 = Z_val[:, j] #set initial condition
        t = collect(0.0f0:dt:dt*lag)
        tspan = (t[1], t[end])
        pred = predict_with_model(u0, model_trained, tspan, t)
        y_pred = pred[1, :]
        y_true = Z_val[1, j:10:(j+(10*lag))]

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
    markercolor=palette(:auto)[1],  # usa il primo colore della palette attuale
    label=""
)

display(plt)





#short term prediction for slow variable 
plotlyjs()

#detect critical transitions in the validation set
obs_val = obs_signal[Ntrain+1:end]
obs_val = obs_val[1:10:end]
# sigma_Langevin(x, t) = Σ / sqrt(2*1.02)
# x0 = obs_val[1] 
# traj_langevin_test = evolve_chaos([x0], dt, length(y_pred_short), score_clustered_xt, sigma_Langevin, y_pred_short; timestepper=:euler, resolution=1)
# plot(obs_val[1:length(y_pred_short)], label="obs_val", title="Validation Signal")
# plot!(traj_langevin_test[1,:], label= "Langevin")

transitions = detect_transitions(obs_val; threshold=0.7, window=800, min_spacing=100)
if !isempty(transitions)
    @show transitions[1:min(10, end)]
else
    println("No transition found. Change threshold or window.")
end

sigma_Langevin(x, t) = Σ 
#define list of time horizons for predictions
theta = [0.35, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0] #in seconds

#convert the time horizon into number of steps
timesteps = round.(Int, theta ./ dt)

#initialize array for mse as a function of theta
rmses_vs_theta = Float64[]
var_rmses_x = Float64[]

for timestep in timesteps
    rmses = Float64[]
    for tau_n in transitions
        x0 = obs_val[tau_n - timestep]
        traj_langevin = evolve_chaos([x0], dt, timestep + 10, score_clustered_xt, sigma_Langevin, y_pred_short; timestepper=:euler, resolution=1)
        pred = traj_langevin[1, 1:end]  
        real = obs_val[(tau_n - timestep):(tau_n + 10)]

        # Check lengths
        @assert length(pred) == length(real)

        #Root Mean Squared Error
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
obs_val = obs_signal[Ntrain+1:end] 
obs_val = obs_val[1:10:end]

# Scegli gli indici di theta (i.e., timestep) che vuoi plottare
indices_to_plot = [1, 2, 3, 4, 5]
colors = [:blue, :orange]
# Creazione del layout verticale
sigma_Langevin(x, t) = Σ / sqrt(2*1.5)
p_combined = plot(layout=(5,1), size=(800, 1200))

for (k, idx) in enumerate(indices_to_plot)
    timestep = timesteps[idx]
    tau_n = transitions[35]  # prima transizione

    x0 = obs_val[tau_n - timestep]
    traj_langevin = evolve_chaos([x0], dt, timestep + 500, score_clustered_xt, sigma_Langevin, y_pred_short;
                                  timestepper=:euler, resolution=1)

    pred = traj_langevin[1, 1:end]
    real = obs_val[(tau_n - timestep):(tau_n + 500)]
    t_plot = dt .* (0:(length(pred)-1))

    plot!(p_combined[k], t_plot, real, legend=false, color=colors[1], lw=2, markersize=2)
    plot!(p_combined[k], t_plot, pred, legend=false, color=colors[2], lw=2, markersize=2)
    title!(p_combined[k], "θ = $(round(dt * timestep; digits=1)) s")
    xlabel!(p_combined[k], "")
    ylabel!(p_combined[k], "x(t)")
end

display(p_combined)






plots = []
rng = MersenneTwister()
t = collect(0:dt:dt*(timesteps - 1))
for i in 1:5
    j = rand(rng, 1:(length(obs_signal) - timesteps + 1))
    @show j 
    x0 = obs_val[j]
    @show x0
    traj_langevin = evolve_chaos([x0], dt, timesteps, score_clustered_xt, sigma_Langevin, y_pred_short;
                                  timestepper=:euler, resolution=1)
    p = plot(t, traj_langevin[1, 1:end-1], label="Langevin y₂(t)", lw=2, color=:blue,
             xlabel="t", ylabel="x(t)", title="Sample $i", legend= false)
    plot!(p, t, obs_val[j:100:(j+100*(timesteps-1))], label="Observed y₂(t)", lw=2, color=:red)
    
    push!(plots, p)
end
length(plots)
plot(plots..., layout=(5, 1), size=(800, 1200))


# ------------------------ END PREDICTION SHORT TERM ------------------------ #






sigma_Langevin(x, t) = Σ / sqrt(2)
#define list of time horizons for predictions
theta = [0.05, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0] #in seconds

#convert the time horizon into number of steps
timesteps = round.(Int, theta ./ dt)

#initialize array for mse as a function of theta
rmses_vs_theta = Float64[]
var_rmses_x = Float64[]

for timestep in timesteps
    rmses = Float64[]
    for tau_n in transitions
        x0 = obs_val[tau_n - timestep]
        traj_langevin = evolve_chaos([x0], dt, timestep + 10, score_clustered_xt, sigma_Langevin, y_pred_short; timestepper=:euler, resolution=1)
        pred = traj_langevin[1, 1:end]  
        real = obs_val[(tau_n - timestep):(tau_n + 10)]

        # Check lengths
        @assert length(pred) == length(real)

        #Root Mean Squared Error
        rmse = sqrt(mean((pred .- real).^2))
        push!(rmses, rmse)
    end
    push!(rmses_vs_theta, mean(rmses))
    push!(var_rmses_x, std(rmses))
end

#rmses_vs_theta[1] = 0.0  # Set RMSE for theta=0 to 0

plot!(plt_rmse_x,
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
    markercolor=palette(:auto)[2],  # usa il primo colore della palette attuale
    label=""
)







































# t= collect(0:dt:dt*49999)
# plot(t, obs_signal[1:50000])





# # First n_long steps prediction vs truth
# size(Z, 2)
# n_long = min(1000000, size(Z, 2))
# t_long = collect(0.0f0:dt:dt*(n_long - 1))
# tspan_long = (t_long[1], t_long[end])
# pred_long = predict_with_model(u0, model_trained, tspan_long, t_long)
# max_steps = min(size(pred_long, 2), size(Z, 2))
# y_pred_long = pred_long[1, 1:max_steps] .+ 0.23
# mu, sigy = mean(y_pred_long), std(y_pred_long)
# y_pred_long = (y_pred_long .- mu) ./ sigy  # Normalize the predicted signal

# y_true_long = Z[1, 1:max_steps]
# t_plot = t_long[1:max_steps]


# #Plot of the time series
# plotlyjs()
# plt2 = plot(t_plot[1:end], y_true_long[1:end], label="True y2(t)", lw=2)
# plot!(plt2, t_plot[1:end], y_pred_long[1:end], label="Predicted y(t)", lw=2, title="$n_long steps with m= $m, n_steps = $n_steps and dt = $dt")


# #plot of the PDFs

# plotlyjs()
# kde_pred = kde(y_pred_long)
# kde_obs_y2 = kde(y_true_long)
# plot_kde = plot(kde_pred.x, kde_pred.density; label = "prediction", color = :red)
# plot!(plot_kde, kde_obs_y2.x, kde_obs_y2.density; label = "observations", color = :blue)

# # Display the plots

# #display(plt_loss)
# display(plt1)
# display(plt2)
# display(plot_kde)
# #================== SIMULATE LANGEVIN DYNAMICS ==================#
# plot(y_true_long .- y_pred_long, label="Residual error")

# τ_y2 =2.0
# # Diffusion coefficient accounting for non zero decorellation time √(2Φ)
# sigma_Langevin(x, t) = Σ / sqrt(2 * τ_y2) 

# # Langevin dynamics
# size(y_pred_long)
# timesteps = 900000
# dt = 0.01
# traj_langevin = evolve_chaos([0.0], dt, timesteps, score_clustered_xt, sigma_Langevin, y_pred_long[1:10:end]; timestepper=:euler, resolution=1)
# traj_langevin_2 = evolve_chaos([0.0], dt, timesteps,score_clustered_xt, sigma_Langevin, y_true_long[1:10:end]; timestepper=:euler, resolution=1)




# M_langevin = mean(traj_langevin[1, :])[1]
# S_langevin = std(traj_langevin[1, :])[1]
# traj_langevin_norm = (traj_langevin[1:1, :] .- M_langevin) ./ S_langevin

# M_langevin_2 = mean(traj_langevin_2[1, :])[1]
# S_langevin_2 = std(traj_langevin_2[1, :])[1]
# traj_langevin_norm_2= (traj_langevin_2[1:1, :] .- M_langevin_2) ./ S_langevin_2

# size(traj_langevin_norm)
# compare_plt= plot(traj_langevin_norm[1, 1:end], label="Langevin y2(t)", xlabel="Time Step", ylabel="Normalized y2",
#     title="Langevin y2(t) time series", lw=2, color=:blue)
# plot!(compare_plt, traj_langevin_norm_2[1, 1:end], label="Observed y2(t)", lw=2, color=:red)
# plot!(compare_plt, obs[1, 1:10:timesteps*10], label="Real y2(t)", lw=2, color=:green)

# display(compare_plt)

# kde_langevin = kde(traj_langevin_norm[1,:])
# kde_obs = kde(obs[1, :])
# kde_langevin_2 = kde(traj_langevin_norm_2[1,:])

# function normalize_f(f, x, t, M, S)
#     return f(x .* S .+ M, t) .* S
# end


# function true_pdf_normalized(x)
#     x_phys = x .* S[1] .+ M[1]
#     U = .-0.5 .* x_phys.^2 .+ 0.25 .* x_phys.^4
#     p = exp.(-2 .* U ./ D_eff)
#     return p ./ S[1]
# end

# xax = [-1.25:0.005:1.25...]
# xax_2 = [-1.6:0.02:1.6...]
# interpolated_score = [score_clustered(xax[i])[1] for i in eachindex(xax)]
# true_score = [2 * score_true(xax[i], 0.0)[1] / D_eff for i in eachindex(xax)]
# pdf_interpolated_norm = compute_density_from_score(xax_2, score_clustered)
# pdf_true = true_pdf_normalized(xax_2)
# scale_factor = maximum(kde_obs.density) / maximum(pdf_true)
# pdf_true .*= scale_factor

# #Plot PDF
# p_pdf = plot(kde_obs.x, kde_obs.density, label="Observed", lw=2, color=:red)
# plot!(p_pdf, kde_langevin.x, kde_langevin.density, label="Langevin", lw=2, color=:blue)
# xlabel!("x"); ylabel!("Density"); title!("PDF comparison")
# plot!(p_pdf, xax_2, pdf_true; label="PDF analytic", linewidth=2, linestyle=:dash, color=:lime)
# # plot!(p_pdf, xax_2, pdf_interpolated_norm; label="PDF learned", linewidth=2,color=:cyan)

# #Plot Score
# p_score = scatter(centers_sorted, scores; color=:blue, alpha=0.2, label="Cluster centers",
#     xlims=(-1.3, 1.3), ylims=(-5, 5), xlabel="𝑥", ylabel="Score(𝑥)", title="Score Function Estimate")
# plot!(p_score, xax, interpolated_score; label="NN interpolation", linewidth=2, color=:red)
# plot!(p_score, xax, true_score; label="Score analytic", linewidth=2, color=:lime)

# display(p_score)
# display(p_pdf) 








# # CHECK OF TRAJECTORIES FROM TRANSFORMER AND REAL ONES #
# using BSON
# using Plots
# plotlyjs()
# data = BSON.load("fast_sig_norm.bson")
# println(keys(data))

# fast_signal = data[:fast_sig_norm]  
# typeof(fast_signal)
# size(fast_signal)
# # Load data
# data_2 = BSON.load("ensemble_preds.bson")
# ensemble = data_2[:ensemble_preds]  # shape: (100, 500)


# p1 = plot(fast_signal[1:10:5000], lw=2, xlabel="Time step", ylabel="y₂(t)",
#           title="True signal (slow)", color=:blue, legend=false)

# p2 = plot(y_pred_short[1:100:50000], lw=2, xlabel="Time step", ylabel="y₂(t)",
#           title="NODE prediction", color=:red, legend=false)

# p3 = plot(ensemble[75, :], lw=2, xlabel="Time step", ylabel="y₂(t)",
#           title="Transformer prediction (sample 75)", color=:orange, legend=false)


# plot(p1, p2, p3, layout=(3, 1), size=(800, 600))


# display(plot_traj)
# kde_fast_signal = kde(fast_signal[1:10:5000])
# kde_ensemble = kde(ensemble[75, :])
# plot_kde = plot(kde_fast_signal.x, kde_fast_signal.density; label = "Fast Signal PDF", color = :red)
# plot!(plot_kde, kde_ensemble.x, kde_ensemble.density; label = "Ensemble PDF", color = :blue,
#     title="PDF of Fast Signal vs Ensemble", xlabel="y2", ylabel="Density") 
# display(plot_kde)
#     U = .-0.5 .* x_phys.^2 .+ 0.25 .* x_phys.^4
#     p = exp.(-2 .* U ./ D_eff)
#     return p ./ S[1]
# end

# xax = [-1.25:0.005:1.25...]
# xax_2 = [-1.6:0.02:1.6...]
# interpolated_score = [score_clustered(xax[i])[1] for i in eachindex(xax)]
# true_score = [2 * score_true(xax[i], 0.0)[1] / D_eff for i in eachindex(xax)]
# pdf_interpolated_norm = compute_density_from_score(xax_2, score_clustered)
# pdf_true = true_pdf_normalized(xax_2)
# scale_factor = maximum(kde_obs.density) / maximum(pdf_true)
# pdf_true .*= scale_factor

# ########## 7. Plotting ##########
# Plots.default(fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10, legendfontsize=10)
# plotlyjs()




# #Plot PDF
# p_pdf = plot(kde_obs.x, kde_obs.density, label="Observed", lw=2, color=:red)
# plot!(p_pdf, kde_langevin.x, kde_langevin.density, label="Langevin", lw=2, color=:blue)
# xlabel!("x"); ylabel!("Density"); title!("PDF comparison")
# plot!(p_pdf, xax_2, pdf_true; label="PDF analytic", linewidth=2, linestyle=:dash, color=:lime)
# # plot!(p_pdf, xax_2, pdf_interpolated_norm; label="PDF learned", linewidth=2,color=:cyan)

# #Plot Score
# p_score = scatter(centers_sorted, scores; color=:blue, alpha=0.2, label="Cluster centers",
#     xlims=(-1.3, 1.3), ylims=(-5, 5), xlabel="𝑥", ylabel="Score(𝑥)", title="Score Function Estimate")
# plot!(p_score, xax, interpolated_score; label="NN interpolation", linewidth=2, color=:red)
# plot!(p_score, xax, true_score; label="Score analytic", linewidth=2, color=:lime)

# display(p_score)
# display(p_pdf) 








# # CHECK OF TRAJECTORIES FROM TRANSFORMER AND REAL ONES #
# using BSON
# using Plots
# plotlyjs()
# data = BSON.load("fast_sig_norm.bson")
# println(keys(data))

# fast_signal = data[:fast_sig_norm]  
# typeof(fast_signal)
# size(fast_signal)
# # Load data
# data_2 = BSON.load("ensemble_preds.bson")
# ensemble = data_2[:ensemble_preds]  # shape: (100, 500)


# p1 = plot(fast_signal[1:10:5000], lw=2, xlabel="Time step", ylabel="y₂(t)",
#           title="True signal (slow)", color=:blue, legend=false)

# p2 = plot(y_pred_short[1:100:50000], lw=2, xlabel="Time step", ylabel="y₂(t)",
#           title="NODE prediction", color=:red, legend=false)

# p3 = plot(ensemble[75, :], lw=2, xlabel="Time step", ylabel="y₂(t)",
#           title="Transformer prediction (sample 75)", color=:orange, legend=false)


# plot(p1, p2, p3, layout=(3, 1), size=(800, 600))


# display(plot_traj)
# kde_fast_signal = kde(fast_signal[1:10:5000])
# kde_ensemble = kde(ensemble[75, :])
# plot_kde = plot(kde_fast_signal.x, kde_fast_signal.density; label = "Fast Signal PDF", color = :red)
# plot!(plot_kde, kde_ensemble.x, kde_ensemble.density; label = "Ensemble PDF", color = :blue,
#     title="PDF of Fast Signal vs Ensemble", xlabel="y2", ylabel="Density") 
# display(plot_kde)




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