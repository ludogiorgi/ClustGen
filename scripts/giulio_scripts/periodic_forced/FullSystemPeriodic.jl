using Pkg
Pkg.activate(".")
Pkg.instantiate()

#TRAIN NODE WITH Residual

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
using StaticArrays


#Parameters for the system
# External forcing on x
const A    = 0.1
const ω    = 2π * 0.002

# Coupling and OU-like relaxation for z
const σ   = 0.08

# Lorenz-63 fast driver for z (enters as y2/ε)
const ε    = 0.5

# Lorenz-63 parameters
const σL   = 10.0
const ρL   = 28.0
const βL   = 8/3

# -------------------- New potential U(x) and its derivative --------------------

# U(x) = (50/131) * (x^6/3 - (122/64)*x^4 + (79/29)*x^2)
@inline U(x) = (50/131) * ( (x^6)/3 - (122/64)*x^4 + (79/29)*x^2 ) + 0.02 * x^2

# U'(x) = (50/131) * ( 2*x^5 - (61/8)*x^3 + (158/29)*x )
@inline dUdx(x) = (50/131) * ( 2*x^5 - (61/8)*x^3 + (158/29)*x ) + 0.04 * x

# -------------------- Deterministic vector field (x, z, y1, y2, y3) --------------------

# Deterministic vector field without z; x is driven directly by y2 (Eq. 22)
function F_det(X, t)
    x, y1, y2, y3 = X

    # ẋ = -U'(x) + A cos(ω t) + (σ1/ε) * y2
    dx  = -dUdx(x) + A*cos(ω*t) + (σ/ε)*y2

    # Fast Lorenz-63, scaled by 1/ε^2
    dy1 = (σL/ε^2) * (y2 - y1)
    dy2 = (1/ε^2)  * (ρL*y1 - y2 - y1*y3)
    dy3 = (1/ε^2)  * (y1*y2 - βL*y3)

    return [dx, dy1, dy2, dy3]
end

# No diffusion: pure ODE in 4D
@inline function sigma_det(_, _)
    return zeros(4)
end

#====================BUILD AUGMENTED STATE=====================#
function augmented_state(x::AbstractVector, t::AbstractVector, ω)    
    n_points = length(t)
    augmented = Matrix{Float64}(undef, 3, n_points)
    
    for i in ProgressBar(1:n_points)
        augmented[1, i] = sin(ω * t[i])   # sin(ωt)
        augmented[2, i] = cos(ω * t[i])   # cos(ωt)
        augmented[3, i] = x[i]            # x(t)
    end
    
    return augmented
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
function delay_embedding(x; τ, m, dt_sample=dt)
    q = Int(round(Int, τ / dt_sample))
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




#==================== NORMALIZE TIME SERIRES ====================#
function normalize_time_series(obs)
    if ndims(obs) == 1
        # Caso 1D (vettore)
        mean_obs = mean(obs)
        sigma_obs = std(obs)
        return (obs .- mean_obs) ./ sigma_obs
    else
        # Caso nD (matrice), normalizza lungo ogni riga (dimensione)
        mean_obs = mean(obs, dims=2)
        sigma_obs = std(obs, dims=2)
        return (obs .- mean_obs) ./ sigma_obs
    end
end

#==================== DETECT CRITICAL TRANSITIONS ====================#
"""
    detect_transitions(x; thr=1.25/2, window=50, min_spacing=100)

Rileva transizioni tra i tre bacini usando crossing sui due confini ±thr
(con thr di default 1.25/2 = 0.625). Una transizione è registrata quando
la serie attraversa uno dei due confini e le medie su finestre passata e futura
si trovano su lati opposti rispetto al confine scelto. È inoltre applicato un
debounce temporale `min_spacing` per evitare conteggi ripetuti.

Argomenti:
- x::AbstractVector: serie temporale di x(t)
- thr::Real: metà distanza tra pozzetti esterni (default 0.625)
- window::Int: ampiezza della finestra per medie passata/futura
- min_spacing::Int: distanza minima tra due transizioni
"""
function detect_transitions(x::AbstractVector; thr::Real=1.25/2, window::Int=50, min_spacing::Int=100)
    transitions = Int[]
    last = -min_spacing
    N = length(x)

    # helper: verifica crossing rispetto a un confine c
    crosses_at(i, c) = (x[i] - c) * (x[i+1] - c) < 0

    for i in 1:(N - 1)
        if (i - last) <= min_spacing
            continue
        end

        # per stabilità, controlla che le medie passata/futura siano da parti opposte del confine
        past_lo = max(1, i - window + 1)
        past_hi = i
        future_lo = i + 1
        future_hi = min(N, i + window)

        m_past = mean(@view x[past_lo:past_hi])
        m_future = mean(@view x[future_lo:future_hi])

        # crossing su +thr
        if crosses_at(i, +thr) && (m_past - thr) * (m_future - thr) < 0
            push!(transitions, i + 1)
            last = i
            continue
        end

        # crossing su -thr
        if crosses_at(i, -thr) && (m_past + thr) * (m_future + thr) < 0
            push!(transitions, i + 1)
            last = i
            continue
        end
    end

    return transitions
end

#========== MAIN ==========#






# -------------------- Simulate --------------------

dim      = 1                # we will analyze only x = first component
dt_sim       = 0.001            # smaller timestep for stability
Nsteps   = 100_000_000          # fewer steps to avoid instability
res = 1000
dt = dt_sim * res

# Initial conditions (paper uses x(0)=0.1 and random z,y's)
rng = Random.default_rng()
x0  = 0.1
y10 = rand(rng, Uniform(-1, 1))    # smaller range for stability
y20 = rand(rng, Uniform(-1, 1))    # smaller range for stability
y30 = rand(rng, Uniform(-1, 1))    # smaller range for stability

X0 = [x0, y10, y20, y30]

# integrate deterministically with RK4
obs_nn = evolve(X0, dt_sim, Nsteps, F_det, sigma_det; timestepper=:euler, resolution=res)

Plots.plot(obs_nn[1,1:5000], markersize=2, label="",
              xlabel="time index", ylabel="x (normalized)",
              title="Observed Trajectory (first 10k steps)")
##

# Augment the state with a timeseries of sine and cosine with the same frequency as the forcing
# Create time array for the simulation
t_array = range(0, Nsteps * dt_sim, length=size(obs_nn, 2))


# Create 3D state: [sin, cos, x]
state_3d = augmented_state(obs_nn[1, :], t_array, ω)

# -------------------- Normalize & basic stats --------------------
M = mean(state_3d, dims=2)
S = std(state_3d, dims=2)
obs = (state_3d .- M) ./ S
x_obs = obs[3, :]


# Autocovariance (lag up to 300 on a subset to keep it light)
autocov_obs = zeros(1000)
autocov_obs = autocovariance(obs[3,1:1:end]; timesteps=1000)

# Create timeseries of t modulo the period of the forcing
forcing_period = 2π / ω
t_mod_period = mod.(t_array, forcing_period)

# Construct bivariate PDF using x and the timeseries of time values
# Use KernelDensity for 2D density estimation
kde_obs = kde(obs[3,1:1:end])
kde_2d = kde((obs[3,:], t_mod_period))

plt1 = Plots.plot(autocov_obs, label="x", xlabel="Lag",
                  ylabel="Autocovariance", title="Autocovariance of Normalized x(t)")
plt2 = Plots.plot(kde_obs.x, kde_obs.density, label="Observed",
                  xlabel="x", ylabel="Density", title="Observed PDF (normalized)")
# Plot the bivariate PDF
plt3 = Plots.heatmap(kde_2d.x, kde_2d.y, kde_2d.density', 
                     xlabel="x (normalized)", ylabel="t mod period", 
                     title="Bivariate PDF: x vs time")

Plots.plot(plt1, plt2, plt3, layout=(3, 1), size=(500, 1000))

## -------------------- (Optional) Clustering / KGMM scaffolding --------------------
# The original pipeline estimated score from small-noise identities tailored to SDEs.
# You can still run KGMM-style estimators on the *empirical* distribution of x,
# but a closed-form “true score” from dynamics is no longer applicable in this deterministic setup.

normalization = false
σ_value = 0.05

# This expects a (3×N) array for the 3D timeseries, so use obs_3d
obs_for_clust = obs
averages, centers, Nc, labels =
    f_tilde_labels(σ_value, obs_for_clust; prob=0.0005, do_print=false,
                conv_param=0.002, normalization=normalization)

inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)

# Example: quick visualization of the KGMM score estimate (scaled by σ_value)
centers_sorted_indices = sortperm(centers[1, :])
centers_sorted = centers[:, centers_sorted_indices][:]         # 1D centers
scores_est = .- averages[:, centers_sorted_indices][:] ./ σ_value

Plots.scatter(centers_sorted, scores_est, label="Estimated score (KGMM)",
              xlabel="x (normalized)", ylabel="score(x)", title="KGMM score estimate")

## -------------------- (Optional) NN training on clustering loss --------------------
@time nn_clustered, loss_clustered = train(inputs_targets, 5000, 32, [3, 100, 50, 3];
                                           use_gpu=false, activation=swish, last_activation=identity)
nn_clustered_cpu = nn_clustered |> cpu
score_clustered(x) = .- nn_clustered_cpu(Float32.(x))[:] ./ σ_value
Plots.plot(loss_clustered, xlabel="iter", ylabel="loss", title="Clustering loss")
##
dim = 3
function full_score(x_obs, t)
    # Augment the state with the normalized time variables (sin and cos)
    time_sin = (sin(ω * t) - M[1]) / S[1]
    time_cos = (cos(ω * t) - M[2]) / S[2]
    augmented_state = vcat(time_sin, time_cos, x_obs)
    
    # Return the full score vector (size: dim x 1)
    return score_clustered(augmented_state)
end

I_D = Matrix{Float64}(I, dim, dim)

drift_x_corrected(x_obs, t) = (I_D * full_score(x_obs, t))[3:end]

sigma_I(x,t) = I_D[3:end,:]

trj_clustered = evolve([0.0], 0.1*dt_sim, Int(Nsteps), drift_x_corrected, sigma_I; timestepper=:euler, resolution=res, boundary=[-10,10])

# Plot observed and generated trajectories in separate panels
plt_obs = Plots.plot(obs[3,1:5000], linewidth=1, linecolor=:blue, label="",
                     xlabel="time index", ylabel="x (normalized)",
                     title="Observed Trajectory (first 5000 steps)",
                     grid=true, legend=false)

plt_gen = Plots.plot(trj_clustered[1,1:5000], linewidth=1, linecolor=:red, label="",
                     xlabel="time index", ylabel="x (normalized)",
                     title="Generated Trajectory (first 5000 steps)",
                     grid=true, legend=false)

# Combine the two panels
Fig1 = Plots.plot(plt_obs, plt_gen, layout=(2,1), size=(800, 600))
display(Fig1)
##
t_array_clustered = range(0, Nsteps * dt_sim * 0.1, length=size(trj_clustered, 2))
t_mod_period_clustered = mod.(t_array_clustered, forcing_period)



kde_clustered = kde(trj_clustered[1,1:1:end])
kde_2d_clustered = kde((trj_clustered[1,:], t_mod_period_clustered[1:length(trj_clustered[1,:])]))

# Find common color range for heatmaps
density_min = min(minimum(kde_2d.density), minimum(kde_2d_clustered.density))
density_max = max(maximum(kde_2d.density), maximum(kde_2d_clustered.density))

# Create professional plots with consistent styling
plt1 = Plots.plot(kde_obs.x, kde_obs.density, 
                  linewidth=2, linecolor=:blue, label="Observed",
                  xlabel="x (normalized)", ylabel="Density", 
                  title="Probability Density Functions",
                  legend=:topleft, grid=true, 
                        titlefontsize=12, legendfontsize=10,
                        bottom_margin=15Plots.mm)
Plots.plot!(kde_clustered.x, kde_clustered.density, 
            linewidth=2, linecolor=:red, label="Generated",
            linestyle=:dash)

plt2 = Plots.heatmap(kde_2d.x, kde_2d.y, kde_2d.density', 
                     xlabel="x (normalized)", ylabel="t mod period", 
                     title="Observed Bivariate PDF",
                            color=:viridis, clim=(density_min, density_max),
                            titlefontsize=12,
                            bottom_margin=15Plots.mm)

plt3 = Plots.heatmap(kde_2d_clustered.x, kde_2d_clustered.y, kde_2d_clustered.density', 
                     xlabel="x (normalized)", ylabel="t mod period", 
                     title="Generated Bivariate PDF",
                            color=:viridis, clim=(density_min, density_max),
                            titlefontsize=12,
                            bottom_margin=15Plots.mm)

# Create professional layout
Fig2 = Plots.plot(plt1, plt2, plt3, 
           layout=(3, 1), 
           size=(800, 1200),
           plot_title="Comparison of Observed vs Generated Dynamics",
           plot_titlefontsize=14,
           margin=5Plots.mm)
display(Fig2)

##
autocov_obs = autocovariance(obs[3,1:end]; timesteps=700)
autocov_trj_clustered = autocovariance(trj_clustered[1,1:10:end]; timesteps=700)

plt_ac = Plots.plot(autocov_obs;
    color=:blue, linewidth=2, label="Observed",
    xlabel="Lag", ylabel="Autocovariance",
    title="Autocovariance: observed vs generated",
    titlefontsize=12, legendfontsize=10, legend=:topright)
Plots.plot!(plt_ac, autocov_trj_clustered; color=:red, linewidth=2, label="Generated")
display(plt_ac)

# ================= Score: NN (normalized) vs Analytic (normalized) =================

# Convenience: turn M,S into vectors
const Mv = vec(M);  const Sv = vec(S)

# Learned normalized x-score at a given phase t0, evaluated on normalized grid u
function s_norm_net_x(u_grid::AbstractVector, t0::Real)
    u1 = (sin(ω*t0) - Mv[1]) / Sv[1]
    u2 = (cos(ω*t0) - Mv[2]) / Sv[2]
    s_vals = similar(u_grid)
    @inbounds for i in eachindex(u_grid)
        u3 = u_grid[i]
        s = score_clustered(Float32.([u1, u2, u3]))  # score in normalized coords
        s_vals[i] = s[3]                              # x-component
    end
    return s_vals
end

# Analytic normalized x-score at phase t0 on the same normalized grid u:
# s_analytic_norm(u,t) = S3 * [ -U'(x) + A cos(ω t) ], with x = M3 + S3*u
function s_norm_analytic_x(u_grid::AbstractVector, t0::Real)
    s_vals = similar(u_grid)
    @inbounds for i in eachindex(u_grid)
        x = Mv[3] + Sv[3]*u_grid[i]
        s_vals[i] = Sv[3] * ( -dUdx(x) + A*cos(ω*t0) )
    end
    return s_vals
end

# Phases and labels
T = forcing_period
phases = [0.0, 0.25T, 0.5T, 0.75T]
legenda = ["t = 0", "t = T/4", "t = T/2", "t = 3T/4"]

# Normalized x-grid (cover typical range)
u_grid = range(-3.0, 3.0, length=600)

# Build 2x2 panel comparing NN vs Analytic at the four phases
score_plots = Plots.Plot[]
for (j, t0) in enumerate(phases)
    s_net  = s_norm_net_x(u_grid, t0)
    s_an = s_norm_analytic_x(u_grid, t0)

    # Etichette solo nell'ultima subplot, vuote nelle altre
    label_an = (j == length(phases) ? "Analytic (normalized)" : "")
    label_nn = (j == length(phases) ? "NN (normalized)" : "")

    p = Plots.plot(u_grid, s_an;
                   label=label_an,
                   xlabel="x (normalized)", ylabel="score",
                   title="$(legenda[j])",
                   linewidth=2, xlims=(-1.5, 1.5), ylims=(-3, 3), color=:lime,
                   legend=false,
                   titlefontsize=12, legendfontsize=10,
                   bottom_margin=15Plots.mm)
    Plots.plot!(p, u_grid, s_net ./ 7;
                label=label_nn, linewidth=2,
                xlims=(-1.5, 1.5), ylims=(-3, 3), color=:red)
    push!(score_plots, p)
end

Fig3 = Plots.plot(score_plots..., layout=(2,2), size=(900,700),
           plot_title="Phase-conditioned normalized Score Function",
           legend=:outerbottomright, legendfontsize=8)
display(Fig3)
# ================================================================================

# ------------------------
# 7. Phi calculation
# ------------------------
#rate matrix
dt = 1.0
Q = generator(labels; dt=dt)
P_steady = steady_state(Q)
#test if Q approximates well the dynamics
tsteps = 50
res = 1


auto_obs = autocovariance(obs[3, 1:res:end]; timesteps=tsteps) 
auto_Q = autocovariance(centers[3,:], Q, [0:dt*res:Int(res * (tsteps-1) * dt)...])


plt = Plots.plot(auto_obs, color=:blue)
plt = Plots.plot!(auto_Q, color=:red)

#compute the score function
gradLogp = - averages ./ σ_value


#compute Phi and Σ
M_Q = centers * Q * (centers *Diagonal(P_steady))'
V_Q = gradLogp * (centers * Diagonal(P_steady))'
Φ = (M_Q * inv(V_Q))
Σ = cholesky(0.5 * (Φ .+ Φ')).L 
println("Σ = ", Σ)



########## Test effective dynamics ##########

res = 100

# === SDE 1D per x(t) con score aumentata ===
# Requisiti: hai già definito Φ, Σ, ω, M_val, S_val, score_clustered(..)
#            (dove M_val,S_val sono le statistiche usate nel training per normalizzare sin,cos,x)

# coefficiente di rumore equivalente (solo x)
sigma_lang(x, t)= Matrix(Σ)

# drift 1D: usa il tempo reale per costruire lo stato aumentato (normalizzato)
function drift_x_only(u, t)
    x = u[1]
    svec = full_score(x, t)          # restituisce vettore 3D (sin, cos, x)
    return [dot(Φ[3, :], svec)]      # scala: vettore 1x1 per evolve
end

# diffusione 1D (scalare): niente rumore su sin/cos perché non li integriamo

# integrazione
u0 = [0.0]
traj_x = evolve(u0, 0.1*dt_sim, Nsteps, drift_x_only, sigma_lang;
                timestepper=:euler, resolution=res, boundary=false, scalar_prod=true)

x_generated = traj_x[1, :] 


# (opzionale) se vuoi già il segnale normalizzato come in training:
x_generated_norm = normalize_time_series(x_generated) 
# plot(x_generated_norm[1:1000000])

# Confronta le distribuzioni
kde_x_gen_phi = kde(x_generated_norm)
plot(kde_x_gen_phi.x, kde_x_gen_phi.density; label="Generated x(t)", title="PDFs Compared", lw=2, color=:blue)
plot!(kde_obs.x, kde_obs.density; label="Original x(t)", lw=2, color=:red)


autocov_obs = autocovariance(obs[3,1:end]; timesteps=300)
autocov_x_gen_phi = autocovariance(x_generated_norm[1:100:end]; timesteps=300)
plot(autocov_x_gen_phi; label="Generated x(t)", title="Autocovariance of Generated x(t)", lw=2, color=:blue)
plot!(autocov_obs; label="Original x(t)", lw=2, color=:red)

# ------------------------------------------------------------------
# 8. Extrapolate fast dynamics from x(t) knowing the score function
# ------------------------------------------------------------------
dt=dt_sim
function estimate_zt(x::Vector{Float64}, Σ::Float64, dt, Sx::Real)
    N = length(x) - 1
    zt_estimated = zeros(Float64, N)
    Σ_norm = Σ / Sx 
    for n in 1:N
        dx_dt = (x[n+1] - x[n]) / dt
        s = drift_x_only(x[n], n * dt)
        zt_estimated[n] = (dx_dt - s[1]) / Σ_norm
    end
    return zt_estimated
end
Nsteps = 10000000
#divide obs_aug[3,:] in training set and validation set
# integrate deterministically with RK4
obs_nn_for_z = evolve(X0, dt_sim, Nsteps, F_det, sigma_det; timestepper=:euler, resolution=1, scalar_prod=false)

# Augment the state with a timeseries of sine and cosine with the same frequency as the forcing
# Create time array for the simulation
t_array = range(0, Nsteps * dt_sim, length=size(obs_nn_for_z, 2))


# Create 3D state: [sin, cos, x]
println("Creating augmented state...")
state_3d = augmented_state(obs_nn_for_z[1, :], t_array, ω)

# -------------------- Normalize & basic stats --------------------
M = mean(state_3d, dims=2)
S = std(state_3d, dims=2)
obs_for_z = (state_3d .- M) ./ S
x_obs_for_z = obs_for_z[3, :]

train_frac = 0.8
Ntrain = Int(floor(length(obs_for_z[3,:]) * train_frac))
#Nval = Nsteps - Ntrain

obs_train = obs_for_z[3, 1:Ntrain]    #8M steps
obs_val = obs_for_z[3, Ntrain+1:end]  #2M steps

println("Estimating z(t) from score function...")
zt_est_train = estimate_zt(obs_train, Σ[3,3], dt, S[3])
zt_est_val = estimate_zt(obs_val, Σ[3,3], dt, S[3])
    
zt_est_train_norm = normalize_time_series(zt_est_train)
zt_est_val_norm = normalize_time_series(zt_est_val)

#real time series of z(t) for comparison
zt_obs_norm = normalize_time_series(obs_nn_for_z[3,:])
zt_obs_train_norm = zt_obs_norm[1:Ntrain]

# Plot the estimated z(t)
plotlyjs()
plt_fast_sig = Plots.plot(zt_est_train_norm[1:100000], label="Estimated z(t)", xlabel="Time Step", ylabel="Normalized z(t)",
    title="Estimated z(t) from Score Function", lw=2, color=:red)
Plots.plot!(plt_fast_sig, zt_obs_train_norm[1:100000], label="Original z(t)", lw=2, color=:blue)

kde_fast_sig = kde(zt_est_train_norm)
kde_signal_norm = kde(zt_obs_train_norm)
plotlyjs()
plt_kde = Plots.plot(kde_fast_sig.x, kde_fast_sig.density, label="Estimated z(t) PDF", lw=2, color=:red)
Plots.plot!(plt_kde, kde_signal_norm.x, kde_signal_norm.density, label="Original z(t) PDF", lw=2, color=:blue,
    title="PDF of Estimated z(t) vs Original z(t)",
    xlabel="z(t)", ylabel="Density")



# ------------------------
# 8. Construct delay embedding
# ------------------------
dt_training = 0.001f0

acf_y2x_t_norm = autocovariance(zt_est_train; timesteps=1000)
autocov_y2 = autocovariance(obs_nn_for_z[3,1:Ntrain]; timesteps=1000)
#acf_y2x_v_norm = autocovariance(zt_est_val_norm; timesteps=1000)
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

τ_opt, acf = estimate_tau(zt_est_train_norm, dt_training)

@info "Scelta ottimale di τ ≈ $(round(τ_opt, digits=4))"
τ = 0.25*0.073 
q = round(Int, τ / dt)

Z_train = Float32.(delay_embedding(zt_est_train_norm; τ=τ, m=m, dt_sample=dt_training))
Y_embed = Float32.(delay_embedding(zt_est_val_norm; τ=τ, m=m, dt_sample=dt_training))  # (m, N)
# Allinea x(t) del validation set con le colonne di Y_embed
x_val = obs_val  # x normalizzato sul validation set
start_idx = 1 + (m - 1) * q
Ny = size(Y_embed, 2)
X_cut = Float32.(x_val[start_idx : (start_idx + Ny - 1)])  # x(t) allineato alla prima riga di Y_embed (y2(t))
Z_val = vcat(X_cut', Y_embed)  # (m+1, Ny)


# ------------------------
# 1. NODE Model definition
# ------------------------

m = 10  # delay embedding dim
layers = [m, 256, 256, m]
activation_hidden = swish
activation_output = identity
model = create_nn(layers; activation_hidden=activation_hidden, activation_output=activation_output)
n_steps = 60
dt_training = 0.001f0
t = collect(0.0f0:dt_training:dt_training*(n_steps - 1))
tspan = (t[1], t[end])
#extract parameters from the model
flat_p0, re = Flux.destructure(model)
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



@load "/Users/giuliodelfelice/Desktop/MIT/ClustGen/best_NODE_y_10/model_epoch_1000.bson" p
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

    plt1 = plot(t_short[1:250], y_true_short[1:250]; label="True y₂(t)", lw=2, color=:blue, markershape=:square, markerstrokewidth=1, markersize=3, line=:solid, marker=:auto)

    plot!(plt1, t_short[1:250], y_pred_short[1:250]; label="Predicted y₂(t)", lw=2, color=:orange, markershape=:square, markerstrokewidth=1, markersize=3, line=:solid, marker=:auto, title="Prediction: 500 steps", xlabel="t", ylabel="y₂(t)")
    display(plt1)

end


mean_acfs_pred = mean(acfs_pred, dims=2)[:]
std_acfs_pred = std(acfs_pred, dims=2)[:]

mean_acfs_true = mean(acfs_true, dims=2)[:]
std_acfs_true = std(acfs_true, dims=2)[:]




gr()  # Assicurati che il backend sia impostato



t_short = collect(0.0f0:dt:dt*100000)
tspan_short = (t_short[1], t_short[end])
t_plot = t_short[1:100]

max_j = size(Y_embed, 2) - 10 * 100000
j = rand(1:max_j)
u0 = Y_embed[:, j]
pred_short = predict_with_model(u0, model_trained, tspan_short, t_short)
y_pred_short = pred_short[1, :]
y_pred_hist = normalize_time_series(y_pred_short)
y_true_hist = Y_embed[1, j:10:(j + 10*100000)]


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
    markercolor=:cyan,
    label=""
)

display(plt)


#detect critical transitions in the validation set
Z_val_subsampled = Z_val[:, 1:10:end]

# sigma_Langevin(x, t) = Σ / sqrt(2*1.02)
# x0 = obs_val[1] 
# traj_langevin_test = evolve_chaos([x0], dt, length(y_pred_short), score_clustered_xt, sigma_Langevin, y_pred_short; timestepper=:euler, resolution=1)
# plot(obs_val[1:length(y_pred_short)], label="obs_val", title="Validation Signal")
# plot!(traj_langevin_test[1,:], label= "Langevin")

transitions = detect_transitions(Z_val_subsampled[1,:]; thr=1.25/2, window=800, min_spacing=500)
if !isempty(transitions)
    @show transitions[1:min(10, end)]
else
    println("No transition found. Change threshold or window.")
end
plt_trans = plot(Z_val_subsampled[1,:], label="Validation Signal")
Plots.scatter!(plt_trans, transitions, Z_val_subsampled[1, transitions]; color=:red, label="Transitions")
sigma_Langevin(x, t) = Σ[3,3]
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
        y_pred_short_temp = pred_short_temp[1, :]  # prima componente

        # Salva la traiettoria simulata della y
        #saved_y_preds[timestep] = y_pred_short_temp

        # Evolvi x con Langevin
        traj_langevin = evolve_chaos([x0], dt, timestep + 10, drift_x_only, sigma_Langevin, y_pred_short_temp;
                                     timestepper = :euler, resolution = 1)
        pred = traj_langevin[1, 1:end]

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

Z_val_subsampled = Z_val[:, 1:10:end]

sigma_Langevin(x, t) = Σ[3,3] / sqrt(2*1.5)

indices_to_plot = [1, 3, 4, 6, 7]
colors = [:blue, :orange]

p_combined = plot(layout=(5, 1), size=(800, 1200))

tau_n = transitions[10]  # transizione fissa

# Dizionario per salvare le traiettorie y_pred_trans
saved_y_preds = Dict{Int, Vector{Float32}}()

for (k, idx) in enumerate(indices_to_plot)
    timestep = timesteps[idx]
    t_start = tau_n - timestep

    # Skip se fuori range
    if t_start < 1 || (tau_n + 400) > length(Z_val_subsampled[1,:])
        println("Skipping plot $(k) — indice fuori range")
        continue
    end

    # x0 nel tempo t_start (bassa risoluzione)
    x0 = Z_val_subsampled[1, t_start]

    # embedding_y in Z_val ad alta risoluzione
    embedding_y = Z_val_subsampled[2:end, t_start]

    # Simula y_pred a partire da embedding_y
    t_plot_trans = collect(0.0f0:dt:dt * (timestep + 400))
    tspan_plot_trans = (t_plot_trans[1], t_plot_trans[end])
    pred_y = predict_with_model(embedding_y, model_trained, tspan_plot_trans, t_plot_trans)
    y_pred_trans = pred_y[1, :]

    # Salva la traiettoria simulata della y
    saved_y_preds[timestep] = y_pred_trans

    # Evolvi x con Langevin
    traj_langevin = evolve_chaos([x0], dt, timestep + 400, drift_x_only, sigma_Langevin, y_pred_trans;
                                  timestepper = :euler, resolution = 1)
    pred = traj_langevin[1, :]
    real = Z_val_subsampled[1, t_start:(tau_n + 400)]

    @assert length(pred) == length(real)
    t_plot_x = dt .* (0:(length(pred) - 1))

    plot!(p_combined[k], t_plot_x, real, legend = false, color = colors[1], lw = 2)
    plot!(p_combined[k], t_plot_x, pred, legend = false, color = colors[2], lw = 2)
    title!(p_combined[k], "θ = $(round(dt * timestep; digits=1)) s")
    xlabel!(p_combined[k], "")
    ylabel!(p_combined[k], "x(t)")
end

display(p_combined)


