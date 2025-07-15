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

#==================== EXTRACT FAST SIGNAL FROM SLOW SIGNAL ====================#
function estimate_y2(x::Vector{Float64}, Σ::Float64, dt::Float64)
    N = length(x) - 1
    y2_estimated = zeros(Float64, N)
    for n in 1:N
        dx_dt = (x[n+1] - x[n]) / dt
        s = score_clustered_xt(x[n], n * dt)
        y2_estimated[n] = (dx_dt - s[1]) / Σ
    end
    return y2_estimated
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
n_steps = 100
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
kde_obs = kde(obs[1, :])

signal_raw = Float32.(obs_nn[3, :])

# Normalizza
M_2 = mean(signal_raw)
S_2 = std(signal_raw)
signal_norm = (signal_raw .- M_2) ./ S_2


#ORA AL POSTO DI FARE LA MOVING AVERAGE ESTRAIAMO LA Y2 DA X E SCORE FUNCTION. E VEDIAMO COME VA
# ========== COMPUTE SCORE FUNCTION USING FIRST NN ========== #
autocov_obs_nn = zeros(4, 100)#

for i in 1:4
    autocov_obs_nn[i, :] = autocovariance(obs_nn[i, :]; timesteps=100)
end

D_eff = dt * (0.5 * autocov_obs_nn[3, 1] + sum(autocov_obs_nn[3, 2:end-1]) + 0.5 * autocov_obs_nn[3, end])
D_eff = 0.3
@show D_eff

# plt_12 = plot(autocov_obs_nn[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of x")


#training and clustering parameters 
σ_value=0.05
prob=0.001
conv_param=0.02
n_epochs=5000
batch_size=16


########## 3. Clustering ##########
averages, centers, Nc, labels = f_tilde_labels(σ_value, obs[:,1:10:end]; prob=prob, do_print=false, conv_param=conv_param, normalization=false)
inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)

########## 4. Score Functions ##########

#analytic score function
f1(x,t) = x .- x.^3
score_true(x, t) = normalize_f(f1, x, t, M, S)

#learned score function
#kde_x = kde(obs_nn[1, :])
centers_sorted_indices = sortperm(centers[1, :])
centers_sorted = centers[:, centers_sorted_indices][:]
scores = .- averages[:, centers_sorted_indices][:] ./ σ_value

########## 5. Train NN ##########
@time nn, losses = train(inputs_targets, n_epochs, batch_size, [1, 50, 25, 1];
    opt=Flux.Adam(0.001), activation=swish, last_activation=identity,
    use_gpu=false)

nn_clustered_cpu = nn |> cpu
score_clustered(x) = .- nn_clustered_cpu(reshape(Float32[x...], :, 1))[:] ./ σ_value
score_clustered([0.1])


########## Phi calculation ##########
dt=0.01
#rate matrix
Q = generator(labels; dt=dt)*0.2
P_steady = steady_state(Q)
#test if Q approximates well the dynamics
tsteps = 51
res = 10

auto_obs = autocovariance(obs[1:res:end]; timesteps=tsteps) 
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
autocov_y2 = autocovariance(signal_norm; timesteps=1000)
plot(autocov_y2, label="Autocovariance of y2", xlabel="Lag", ylabel="Autocovariance",
    title="Autocovariance of the estimated y2 signal", linewidth=2)
dt
########## Estimate y2 from the slow variable x ##########
dt = 0.001
fast_sig = estimate_y2(obs[1,:], Σ, dt)

M_fast = mean(fast_sig)
S_fast = std(fast_sig)
fast_sig_norm = (fast_sig .- M_fast) ./ S_fast

# Plot the estimated y2
plotlyjs()
plt_fast_sig = plot(fast_sig_norm[1:10000], label="Estimated y2", xlabel="Time Step", ylabel="Normalized y2",
    title="Estimated y2 from Score Function", lw=2, color=:red)
plot!(plt_fast_sig, signal_norm[1:10000], label="Original y2", lw=2, color=:blue)

kde_fast_sig = kde(fast_sig_norm)
kde_signal_norm = kde(signal_norm)
plotlyjs()
plt_kde = plot(kde_fast_sig.x, kde_fast_sig.density, label="    Estimated y2 PDF", lw=2, color=:red)
plot!(plt_kde, kde_signal_norm.x, kde_signal_norm.density, label="  Original y2 PDF", lw=2, color=:blue,
    title="PDF Estimated vs Real",
    xlabel="y2", ylabel="Density")

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

τ_opt, acf = estimate_tau(fast_sig_norm, dt)


@info "Scelta ottimale di τ ≈ $(round(τ_opt, digits=4))"
τ = 0.3*τ_opt  

Z = Float32.(delay_embedding(fast_sig_norm; τ=τ, m=m))
Z[:,1]
# ------------------------
# 3. Batching for NODE training
# ------------------------

batch_size = n_steps + 1
n_batches = 2000
data_sample = gen_batches(Z, batch_size, n_batches)

acf_fast = autocovariance(fast_sig_norm; timesteps=1000)
plot(acf_fast, label="Autocovariance of fast signal", xlabel="Lag", ylabel="Autocovariance",
    title="Autocovariance of the estimated y2 signal", linewidth=2)

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
@load "/Users/giuliodelfelice/Desktop/MIT/MODELLO TRAINATO CHE ANDAVA ABBASTANZA BENE CON LA y2 estratta dalla x/model_epoch_500.bson" p
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
size(Z, 2)
n_long = min(10000000, size(Z, 2))
t_long = collect(0.0f0:dt:dt*(n_long - 1))
tspan_long = (t_long[1], t_long[end])
pred_long = predict_with_model(u0, model_trained, tspan_long, t_long)
max_steps = min(size(pred_long, 2), size(Z, 2))
y_pred_long = pred_long[1, 1:max_steps] 
mu, sigy = mean(y_pred_long), std(y_pred_long)
y_pred_long = (y_pred_long .- mu) ./ sigy  # Normalize the predicted signal

y_true_long = Z[1, 1:max_steps]
t_plot = t_long[1:max_steps]
y_true_long[1,1]
y_pred_long[1,1]
u0
pred_long[1,1]
using FFTW
using Plots

function plot_power_spectrum(signal::Vector{Float64}; dt=0.001, label::String="")
    N = length(signal)
    signal = signal .- mean(signal)  # remove DC offset
    yf = abs.(fft(signal))[1:div(N,2)]  # FFT, keep positive freqs
    psd = (1 / (N * dt)) * (yf .^ 2)
    freqs = (0:(N÷2 - 1)) / (N * dt)
    plot(freqs, psd; xscale=:log10, yscale=:log10, label=label,
         xlabel="Frequency [Hz]", ylabel="Power Spectral Density", lw=2)
end

# Convert to Float64 if needed
y_true_vec = Float64.(y_true_long)
y_pred_vec = Float64.(y_pred_long)

plotlyjs()
p = plot()
plot!(p, plot_power_spectrum(y_true_vec; dt=dt, label="True y₂"))
plot!(p, plot_power_spectrum(y_pred_vec; dt=dt, label="NODE prediction"))
display(p)


#Plot of the time series
plotlyjs()
plt2 = plot(t_plot[1:end], y_true_long[1:end], label="True y2(t)", lw=2)
plot!(plt2, t_plot[1:end], y_pred_long[1:end], label="Predicted y(t)", lw=2, title="$n_long steps with m= $m, n_steps = $n_steps and dt = $dt")

#plot of the PDFs

plotlyjs()
kde_pred = kde(y_pred_long)
kde_obs_y2 = kde(y_true_long)
plot_kde = plot(kde_pred.x, kde_pred.density; label = "prediction", color = :red)
plot!(plot_kde, kde_obs_y2.x, kde_obs_y2.density; label = "observations", color = :blue)

# Display the plots

#display(plt_loss)
display(plt1)
display(plt2)
display(plot_kde)
#================== SIMULATE LANGEVIN DYNAMICS ==================#
plot(y_true_long .- y_pred_long, label="Residual error")

τ_y2 =2.0
# Diffusion coefficient accounting for non zero decorellation time √(2Φ)
sigma_Langevin(x,t) = Σ / sqrt(20 * τ_y2) 

# Langevin dynamics
size(y_pred_long)
timesteps = 900000
dt = 0.01
traj_langevin = evolve_chaos([0.0], dt, timesteps, score_clustered_xt, sigma_Langevin, y_pred_long[1:10:end]; timestepper=:euler, resolution=1)
traj_langevin_2 = evolve_chaos([0.0], dt, timesteps,score_clustered_xt, sigma_Langevin, y_true_long[1:10:end]; timestepper=:euler, resolution=1)




M_langevin = mean(traj_langevin[1, :])[1]
S_langevin = std(traj_langevin[1, :])[1]
traj_langevin_norm = (traj_langevin[1:1, :] .- M_langevin) ./ S_langevin

M_langevin_2 = mean(traj_langevin_2[1, :])[1]
S_langevin_2 = std(traj_langevin_2[1, :])[1]
traj_langevin_norm_2= (traj_langevin_2[1:1, :] .- M_langevin_2) ./ S_langevin_2

size(traj_langevin_norm)
# compare_plt= plot(traj_langevin_norm[1, 1:end], label="Langevin y2(t)", xlabel="Time Step", ylabel="Normalized y2",
#     title="Langevin y2(t) time series", lw=2, color=:blue)
# plot!(compare_plt, traj_langevin_norm_2[1, 1:end], label="Observed y2(t)", lw=2, color=:red)
# plot!(compare_plt, obs[1, 1:10:timesteps*10], label="Real y2(t)", lw=2, color=:green)

# display(compare_plt)

kde_langevin = kde(traj_langevin_norm[1,:])
kde_obs = kde(obs[1, :])
kde_langevin_2 = kde(traj_langevin_norm_2[1,:])

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

p_pdf = plot(kde_obs.x, kde_obs.density, label="Observed", lw=2, color=:red)
plot!(p_pdf, xax_2, pdf_true; label="PDF analytic", linewidth=2, linestyle=:dash, color=:lime)
# plot!(p_pdf, xax_2, pdf_interpolated_norm; label="PDF learned", linewidth=2,color=:cyan)
plot!(p_pdf, kde_langevin.x, kde_langevin.density, label="PDF of Langevin y2(t)", xlabel="y2", ylabel="Density", title="PDF comparison", linewidth=2, color=:blue)
# plot!(p_pdf, kde_langevin_2.x, kde_langevin_2.density, label="PDF of Langevin", xlabel="y2", ylabel="Density", title="PDF comparison", linewidth=2, color=:green)
#autocov_langevin = autocovariance(traj_langevin_norm[1,1:100000]; timesteps=500)
#=============== END MAIN ===============#
























sigma_Langevin(x, t) = Σ 


# Simulate Langevin dynamics
Nsamples = 100000000
t = collect(0:dt:dt*(Nsamples-1))  # esplicitamente un vettore
length(t)
tspan = (t[1], t[end])
u0 = [randn()]

traj_langevin = evolve(u0, dt, Nsteps, score_clustered_xt, sigma_Langevin; timestepper=:euler, resolution=10)
size(traj_langevin)
length(traj_langevin[1,:])
kde_langevin = kde(traj_langevin[1,:])
auto_langevin = autocovariance(traj_langevin[1,:]; timesteps=tsteps)
# plot(kde_y2_test.x, kde_y2_test.density, label="PDF of Langevin y2(t)", xlabel="y2", ylabel="Density", title="Distribution of Langevin y2(t)", linewidth=2)

# plot(traj_langevin, label="Langevin x", xlabel="Time", ylabel="x", title="Langevin x time series")


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

########## 7. Plotting ##########
Plots.default(fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10, legendfontsize=10)
plotlyjs()




#Plot PDF
p_pdf = plot(kde_obs.x, kde_obs.density, label="Observed", lw=2, color=:red)
plot!(p_pdf, kde_langevin.x, kde_langevin.density, label="Langevin", lw=2, color=:blue)
xlabel!("x"); ylabel!("Density"); title!("PDF comparison")
plot!(p_pdf, xax_2, pdf_true; label="PDF analytic", linewidth=2, linestyle=:dash, color=:lime)
# plot!(p_pdf, xax_2, pdf_interpolated_norm; label="PDF learned", linewidth=2,color=:cyan)

#Plot Score
p_score = scatter(centers_sorted, scores; color=:blue, alpha=0.2, label="Cluster centers",
    xlims=(-1.3, 1.3), ylims=(-5, 5), xlabel="𝑥", ylabel="Score(𝑥)", title="Score Function Estimate")
plot!(p_score, xax, interpolated_score; label="NN interpolation", linewidth=2, color=:red)
plot!(p_score, xax, true_score; label="Score analytic", linewidth=2, color=:lime)

display(p_score)
display(p_pdf) 