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

#==================== NORMALIZE TIME SERIRES ====================#
function normalize_time_series(obs)
    mean_obs = mean(obs)
    sigma_obs = std(obs)
    return (obs .- mean_obs) ./ sigma_obs
end

#========== MAIN ==========#

# ------------------------
# 1. Model definition
# ------------------------

m = 10  # delay embedding dim
layers = [m, 256, 256, m]
dt=0.01
activation_hidden = swish
activation_output = identity
model = create_nn(layers; activation_hidden=activation_hidden, activation_output=activation_output)

#extract parameters from the model
flat_p0, re = Flux.destructure(model)

# ------------------------
# 2. Data generation (cosine)
# ------------------------

n_steps = 60
t = collect(0.0f0:dt:dt*(n_steps - 1))
tspan = (t[1], t[end])
Nsteps = 1000000
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

obs_nn = evolve(randn(4), dt, Nsteps, f, sigma; timestepper=:rk4, resolution=1)
# normalize y before embedding
obs = normalize_time_series(obs_nn[1,:])

train_frac = 0.8
Ntrain = Int(floor(Nsteps * train_frac))
#Nval = Nsteps - Ntrain

obs_train = obs[1:Ntrain]    #8M steps
obs_val = obs[Ntrain+1:end]  #2M steps

# ========= ESTIMATE OPTIMAL Tau ========= #
function estimate_tau(y, dt; threshold=0.2)
    y_centered = y .- mean(y)
    acf = autocovariance(y_centered, timesteps=5000)
    for i in 2:length(acf)
        if abs(acf[i]) < threshold
            return i * dt, acf
        end
    end
    return dt * length(acf), acf
end

τ_opt, acf = estimate_tau(obs_nn[1, :], dt)
@info "Scelta ottimale di τ ≈ $(round(τ_opt, digits=4))"
τ = 0.25* τ_opt

Z_train = Float32.(delay_embedding(obs_train; τ=τ, m=m))
Z_val = Float32.(delay_embedding(obs_val; τ=τ, m=m))

# ------------------------
# 3. Batching
# ------------------------

batch_size = n_steps + 1
n_batches = 2000
data_sample = gen_batches(Z_train, batch_size, n_batches)

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
dt = 0.01
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
x_pred_short = pred_short[1, :]
x_true_short = Z_val[1, 1:100]

plotlyjs()
plt1 = plot(t_short, x_true_short, label="True x(t)", lw=2)
plot!(plt1, t_short, x_pred_short, label="Predicted x(t)", lw=2, ls=:dash, title="Prediction: 100 steps")
display(plt1)

# First n_long steps prediction vs truth
n_long = 10000
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


#=========== STATISTICAL INDICATORS FOR PREDICTION ACCURACY ==========#

#short term prediction for slow variable 
plotlyjs()

#detect critical transitions in the validation set
obs_val = obs_signal[Ntrain+1:end]
obs_val = obs_val[1:100:end]


transitions = detect_transitions(obs_val; threshold=0.7, window=800, min_spacing=100)
if !isempty(transitions)
    @show transitions[1:min(10, end)]
else
    println("No transition found. Change threshold or window.")
end

#define list of time horizons for predictions
theta = [0.35, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0] #in seconds

#convert the time horizon into number of steps
timesteps = round.(Int, theta ./ dt)
t_trans = collect(0:dt:dt*(timesteps+10))
tspan_trans = (t_trans[1], t_trans[end])
#initialize array for mse as a function of theta
rmses_vs_theta = Float64[]
var_rmses_x = Float64[]

for timestep in timesteps
    rmses = Float64[]
    for tau_n in transitions
        x0 = obs_val[tau_n - timestep]
        pred = predict_with_model(x0, model_trained, tspan_trans, t_trans)  
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

t_trans_long = collect(0:dt:dt*(timesteps+500))
tspan_trans_long = (t_trans_long[1], t_trans_long[end])
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
    pred = predict_with_model(x0, model_trained, tspan_trans_long, t_trans_long) 
    real = obs_val[(tau_n - timestep):(tau_n + 500)]
    t_plot = dt .* (0:(length(pred)-1))

    plot!(p_combined[k], t_plot, real, legend=false, color=colors[1], lw=2, markersize=2)
    plot!(p_combined[k], t_plot, pred, legend=false, color=colors[2], lw=2, markersize=2)
    title!(p_combined[k], "θ = $(round(dt * timestep; digits=1)) s")
    xlabel!(p_combined[k], "")
    ylabel!(p_combined[k], "x(t)")
end

display(p_combined)



#=============== END MAIN ===============#




