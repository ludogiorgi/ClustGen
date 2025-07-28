# --------------------------------------------------------------------------
# Optimal Control with Spatially Uniform Forcing
# --------------------------------------------------------------------------
# This script finds the optimal control direction u* by performing a parallel
# search over multiple initial conditions, sorts the results, and generates
# a comprehensive multi-panel figure for analysis.
# --------------------------------------------------------------------------

using Pkg
Pkg.activate(".")
Pkg.instantiate()
##
using Plots
using MarkovChainHammer
using ClustGen
using Statistics
using LinearAlgebra
using LaTeXStrings
using StaticArrays
using Printf
using Random
using QuadGK
using ThreadsX
using StatsBase
using Measures
using Flux
using KernelDensity

dim = 2
σ = 0.5
MODEL_PARAMS = (a1=-0.3, b1=-0.2, c1=0.1, F1=0.6, a2=-0.12, b2=-0.12, c2=0.08, F2=0.5, d=0.3)

@inline drift(x::SVector{2,Float64}, p) = SVector(- (p.c1*x[1]^3 - p.b1*x[1]^2 - p.a1*x[1] - p.F1 + p.d*x[2]),
                                                  - (p.c2*x[2]^3 - p.b2*x[2]^2 - p.a2*x[2] - p.F2 + p.d*x[1]))

# Define drift and noise functions for evolve
function F(x, t)
    drift(SVector{2,Float64}(x), MODEL_PARAMS)
end

# Return explicit 2x2 matrix for compatibility with ClustGen.add_noise!
function sigma(x, t)
    σ * Matrix{Float64}(I, 2, 2) # Isotropic noise, returns 2x2 matrix
end

# Initial condition
x0 = [0.0, 0.0]

# Simulation parameters
dt = 0.1
Nsteps = 10000000

println("Simulating unperturbed 2D system with evolve()...")

# Use evolve from ClustGen for SDE integration
obs_nn = evolve(x0, dt, Nsteps, F, sigma; seed=123, resolution=1, timestepper=:euler)

M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S

autocov_obs = zeros(dim, 300)
for i in 1:dim
    autocov_obs[i,:] = autocovariance(obs[i,:]; timesteps=300)
end

autocov_obs_mean = mean(autocov_obs, dims=1)

Plots.plot(autocov_obs_mean[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")

##
obs_uncorr = obs[:, 1:1:end]

Plots.scatter(obs_uncorr[1,1:10000], obs_uncorr[2,1:10000], markersize=1, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

normalization = false
σ_value = 0.05

averages, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.0001, do_print=true, conv_param=0.001, normalization=normalization)

if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
end

##
#################### TRAINING WITH CLUSTERING LOSS ####################

@time nn_clustered, loss_clustered = train(inputs_targets, 200, 8, [dim, 100, 50, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value
Plots.plot(loss_clustered)

##
#################### SAMPLES GENERATION ####################

score_clustered_xt(x,t) = score_clustered(x)
sigma_I(x,t) = 1.0

trj_clustered = evolve([0.0, 0.0], 0.05*dt, Nsteps, score_clustered_xt, sigma_I; timestepper=:rk4, resolution=5, boundary=[-15,15])

kde_clustered_x = kde(trj_clustered[1,:])
kde_true_x = kde(obs[1,:])

kde_clustered_y = kde(trj_clustered[2,:])
kde_true_y = kde(obs[2,:])

plt1 = Plots.plot(kde_clustered_x.x, kde_clustered_x.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
plt1 = Plots.plot!(kde_true_x.x, kde_true_x.density, label="True", xlabel="X", ylabel="Density", title="True PDF")

plt2 = Plots.plot(kde_clustered_y.x, kde_clustered_y.density, label="Observed", xlabel="Y", ylabel="Density", title="Observed PDF")
plt2 = Plots.plot!(kde_true_y.x, kde_true_y.density, label="True", xlabel="Y", ylabel="Density", title="True PDF")

plot(plt1, plt2, layout = @layout([A B; C D]), size=(800, 600), title="KDE Comparison", dpi=300)

##

using ClustGen
f(t) = 1.0

res_trj = 2
steps_trj = 10000
trj = obs[:,1:res_trj:steps_trj*res_trj]

ϵ = 0.05
u(x) = [ϵ, 0.0]
div_u(x) = 0.0

F_pert(x,t) = F(x,t) + u(x) * f(t)

invC0 = inv(cov(obs'))
score_qG(x) = - invC0*x

score_gen(x) = score_clustered(x)

dim_Obs = 2
n_tau = 50

# R_num, δObs_num = zeros(4, dim, n_tau+1), zeros(4, dim, n_tau+1)
R_lin, δObs_lin = zeros(4, dim, n_tau+1), zeros(4, dim, n_tau+1)
R_gen, δObs_gen = zeros(4, dim, n_tau+1), zeros(4, dim, n_tau+1)

# R_num[1,:,:], R_num[2,:,:], R_num[3,:,:], R_num[4,:,:] = generate_numerical_response_HO(F, u, dim, dt, n_tau, 600, sigma, M; n_ens=100000, resolution=res_trj, timestepper=:rk4)

for i in 1:4
    Obs(x) = x.^i
    R_lin[i,:,:], δObs_lin[i,:,:] = generate_score_response(trj, u, div_u, f, score_qG, res_trj*dt, n_tau, Obs, dim_Obs)
    R_gen[i,:,:], δObs_gen[i,:,:] = generate_score_response(trj, u, div_u, f, score_gen, res_trj*dt, n_tau, Obs, dim_Obs)
end


##
R_gen_hack = copy(R_gen)

gr()
plt1 = Plots.plot(R_num[1,1,:], label="Numerical", xlabel="Time lag", ylabel="Response", title="Response 1st moment")
plt1 = Plots.plot!(R_lin[1,1,:], label="Linear")
plt1 = Plots.plot!(R_gen_hack[1,1,:], label="Generative")
plt2 = Plots.plot(R_num[2,1,:] ./ S[1], legend=false, xlabel="Time lag", ylabel="Response", title="Response 2nd moment")
plt2 = Plots.plot!(R_lin[2,1,:])
plt2 = Plots.plot!(R_gen_hack[2,1,:])
plt3 = Plots.plot(R_num[3,1,:] ./ S[1]^2, legend=false, xlabel="Time lag", ylabel="Response", title="Response 3rd moment")
plt3 = Plots.plot!(R_lin[3,1,:])
plt3 = Plots.plot!(R_gen_hack[3,1,:], label="Generative")
plt4 = Plots.plot(R_num[4,1,:] ./ S[1]^3, legend=false, xlabel="Time lag", ylabel="Response", title="Response 4th moment")
plt4 = Plots.plot!(R_lin[4,1,:])
plt4 = Plots.plot!(R_gen_hack[4,1,:])

plt5 = Plots.plot(R_num[1,2,:], legend=false, label="Numerical", xlabel="Time lag", ylabel="Response", title="Response 1st moment")
plt5 = Plots.plot!(R_lin[1,2,:], label="Linear")
plt5 = Plots.plot!(R_gen_hack[1,2,:], label="Generative")
plt6 = Plots.plot(R_num[2,2,:] ./ S[1], legend=false, xlabel="Time lag", ylabel="Response", title="Response 2nd moment")
plt6 = Plots.plot!(R_lin[2,2,:], label="Linear")
plt6 = Plots.plot!(R_gen_hack[2,2,:], label="Generative")
plt7 = Plots.plot(R_num[3,2,:] ./ S[1]^2, legend=false, xlabel="Time lag", ylabel="Response", title="Response 3rd moment")
plt7 = Plots.plot!(R_lin[3,2,:], label="Linear")
plt7 = Plots.plot!(R_gen_hack[3,2,:], label="Generative")
plt8 = Plots.plot(R_num[4,2,:] ./ S[1]^3, legend=false, xlabel="Time lag", ylabel="Response", title="Response 4th moment")
plt8 = Plots.plot!(R_lin[4,2,:], label="Linear")
plt8 = Plots.plot!(R_gen_hack[4,2,:], label="Generative")

plot(
    plt1, plt2, plt3, plt4,
    plt5, plt6, plt7, plt8,
    layout = @layout([A B; C D; E F; G H]),
    size=(1200, 800),
    title="Response Functions Comparison",
    dpi=300
)





##
x_timeseries = permutedims(obs)  # (num_steps, 2) for compatibility with rest of the script

m1_0 = vec(mean(x_timeseries, dims=1))
m2_0 = cov(x_timeseries)

# --- 3.2 Define Control Problem ---
observables = [
    x -> (x[1] - m1_0[1])^2,
    x -> (x[2] - m1_0[2])^2,
    x -> (x[1] - m1_0[1]) * (x[2] - m1_0[2])
]
n_obs = length(observables)
delta_m_vec = [-0.1 * m2_0[1,1], -0.1 * m2_0[2,2], -0.2 * m2_0[1,2]]

# --- 3.3 Pre-compute constant data structures ---
println("\nPre-computing response functions and C-matrices (once)...")

R_gen
R_functions_raw = 1
R_interpolators = [create_linear_interpolator(R_func, DT) for R_func in R_functions_raw]
C_matrices = compute_C_matrices(R_interpolators, T_CONTROL, N_OBS, D)

# --- 3.4 Define optimization constants ---
REGULARIZATION_ALPHA = 1e-7
MAX_ITERATIONS = 150
CONVERGENCE_TOL = 1e-8
STEP_SIZE = 0.05

println("Pre-computation complete.")

# ==========================================================================
# SECTION 4: PARALLEL OPTIMIZATION SEARCH
# ==========================================================================

"""
    find_optimal_u(...)

Runs the iterative optimization for a single initial condition `u_initial`.
Returns all necessary information for later analysis.
"""
function find_optimal_u(
    u_initial::SVector{D, Float64},
    R_interp::Vector,
    C_mats::Array{Matrix{Float64}, 2},
    delta_m::Vector{Float64},
    T::Float64,
    N::Int
)
    u_k = u_initial
    cost_history = Float64[]
    lambda_k = Vector{Float64}(undef, N)

    for k in 1:MAX_ITERATIONS
        M_matrix = compute_M_matrix(u_k, R_interp, T, N)
        M_reg = M_matrix + REGULARIZATION_ALPHA * I
        lambda_k = -(M_reg \ delta_m)
        cost = 0.5 * dot(lambda_k, M_matrix * lambda_k)
        push!(cost_history, cost)
        
        Q_matrix = sum(lambda_k[i] * lambda_k[j] * C_mats[i,j] for i in 1:N, j in 1:N)
        
        eigen_decomp = eigen(Symmetric(Q_matrix))
        idx_max_eig = argmax(eigen_decomp.values)
        u_next_raw = SVector{D, Float64}(eigen_decomp.vectors[:, idx_max_eig])

        if dot(u_k, u_next_raw) < 0
            u_next_raw = -u_next_raw
        end
        
        u_k_plus_1 = normalize((1 - STEP_SIZE) * u_k + STEP_SIZE * u_next_raw)
        
        change = 1.0 - abs(dot(u_k, u_k_plus_1))
        u_k = u_k_plus_1
        
        if change < CONVERGENCE_TOL
            break
        end
    end

    # Enforce sign convention: f*(0) >= 0
    f_at_zero = sum(lambda_k[i] * dot(u_k, R_interp[i](T)) for i in 1:N)
    if f_at_zero < 0
        u_k = -u_k
    end
    
    # Return final u, final λ, final cost, and the full history
    return (u_final=u_k, lambda_final=lambda_k, final_cost=last(cost_history), history=cost_history)
end

# --- 4.1 Define and run the parallel search ---
const NUM_INITIAL_CONDITIONS = 48 # Can be increased for a more thorough search
println("\n--- Starting parallel search with $NUM_INITIAL_CONDITIONS initial conditions... ---")

initial_angles = range(0, 2π, length=NUM_INITIAL_CONDITIONS+1)[1:end-1]
initial_conditions = [SVector{D, Float64}(cos(θ), sin(θ)) for θ in initial_angles]

results = ThreadsX.map(
    u_init -> find_optimal_u(u_init, R_interpolators, C_matrices, delta_m_vec, T_CONTROL, N_OBS),
    initial_conditions
)

println("Parallel search complete.")

# ==========================================================================
# SECTION 5: ANALYSIS OF RESULTS
# ==========================================================================

# --- 5.1 Sort all results by their final cost ---
sorted_results = sort(results, by = r -> r.final_cost)

# --- 5.2 Extract optimal solution and data for plotting ---
best_result = sorted_results[1]
u_star = best_result.u_final
lambda_star = best_result.lambda_final
optimal_cost_history = best_result.history

println("\n--- Global Optimal Solution Found ---")
@printf "Optimal u* = [%.6f, %.6f]\n" u_star[1] u_star[2]
@printf "Final Minimum Cost (Energy): %.6e\n" best_result.final_cost

# ==========================================================================
# SECTION 6: PLOTTING
# ==========================================================================
##
println("\nGenerating final multi-panel figure...")

# --- Panel 1,1: Bivariate PDF and Optimal Direction ---
p11 = histogram2d(
    x_timeseries[:, 1], x_timeseries[:, 2],
    bins=100,
    aspect_ratio=:equal,
    c=:viridis,
    title="System PDF and Optimal Direction u*",
    xlabel="x₁",
    ylabel="x₂",
    legend=false
)
# Add arrow for u_star, scaled for visibility
arrow_scale = 1.0
quiver!(
    p11,
    [m1_0[1]], [m1_0[2]], # Arrow origin
    quiver=([u_star[1]*arrow_scale], [u_star[2]*arrow_scale]), # Arrow direction and length
    color=:red,
    linewidth=3
)


# --- Panel 1,2: Final Loss for All Initial Conditions ---
all_final_costs = [r.final_cost for r in sorted_results]
p12 = scatter(
    1:NUM_INITIAL_CONDITIONS,
    all_final_costs,
    title="Final Cost vs. Initial Condition",
    xlabel="Initial Condition Index (sorted by cost)",
    ylabel="Final Cost (Energy)",
    legend=false,
    ms=4,
    markerstrokewidth=0
)

# --- Panel 2,1: Top N Optimal f(t) Shapes ---
N_TO_PLOT = 48
time_pts = 0:DT:T_CONTROL
p21 = plot(
    title="Optimal f(t) Shapes",
    xlabel="Time",
    ylabel="Amplitude f*(t)"
)
# Use a color gradient to distinguish the lines
colors = cgrad(:plasma, N_TO_PLOT, categorical=true)
for i in 1:min(N_TO_PLOT, length(sorted_results))
    res = sorted_results[i]
    f_optimal_signal = [f_star(t, T_CONTROL, res.u_final, R_interpolators, res.lambda_final) for t in time_pts]
    plot!(p21, time_pts, f_optimal_signal, label="Rank $i", lw=2, color=colors[i], legend=false)
end

# --- Panel 2,2: Cost Convergence for Top N Runs ---
p22 = plot(
    title="Cost Convergence",
    xlabel="Iteration",
    ylabel="Cost (Energy)",
    yaxis=:log, # Use a log scale for better visualization
)
for i in 1:min(N_TO_PLOT, length(sorted_results))
    res = sorted_results[i]
    plot!(p22, res.history, label="Rank $i", lw=2, color=colors[i], legend=false, xlims=(1, 25))
end


# --- Combine all panels into a single figure ---
final_figure = plot(
    p11, p12, p21, p22,
    layout = @layout([A B; C D]),
    size=(1000, 800),
    left_margin=2mm,  # Add white space at the left
    plot_title="Optimal Control Analysis", dpi=600
)

savefig(final_figure, "optimal_control_analysis.png")

display(final_figure)