# -----------------------------------------------------
# --- Environment Setup and System Definition
# -----------------------------------------------------
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using ClustGen
using MarkovChainHammer
using Statistics
using LinearAlgebra
using Interpolations
using QuadGK
using Plots
using KernelDensity
using StatsBase

##
# Define the unperturbed system dynamics from your script
a = -0.0222
b = -0.2
c = 0.0494
F_tilde = 0.6
s = 0.7071

# Store original parameters in a vector
params_original = [a, b, c, F_tilde]

# Langevin equation force and noise functions
F(x, t; p = params_original) = [p[4] + p[1] * x[1] + p[2] * x[1]^2 - p[3] * x[1]^3]
sigma(x, t) = s

# --- Generate data from the unperturbed system ---
dt_sim = 0.01
res = 2
dt = dt_sim * res
Nsteps = 20_000_000 # Use a long trajectory for accurate statistics
println("Generating unperturbed time series...")
obs = evolve([0.0], dt_sim, Nsteps, F, sigma; timestepper=:euler, resolution=res)
plot(obs[1, end-1000:end], label="Unperturbed System", xlabel="Time", ylabel="Observable", title="Unperturbed Langevin Dynamics")
##
# Normalize the data (work in standardized coordinates)
M = mean(obs, dims=2)[1]
S = std(obs, dims=2)[1]
println("Unperturbed System: Mean = $(round(mean(obs), digits=4)), Variance = $(round(var(obs), digits=4))")

# -----------------------------------------------------
# --- Problem Formulation (Applying Manuscript's Theory)
# -----------------------------------------------------

# 1. Define Target Moment Changes (Δμ)
# We want to perturb the first two central moments (mean and variance).
# Let's set a target: increase the mean by 0.05 and decrease the variance by 0.1.
# M = 2 observables (mean, variance)
# N = 4 parameters (a, b, c, F_tilde)
# Target moment changes (mean, variance)

println("Testing with smaller target changes to validate linear response theory...")

# 2. Define Observables (A_k) and Forcing Fields (f_i)
# Observables (for standardized data where mean=0, var=1):
A_1(x) = (x - M)       # Observable for the mean
A_2(x) = (x - M).^2       # Observable for the variance (since mean is 0)
observables = [A_1, A_2]

# The forcing term is F = F_tilde*f_F + a*f_a + b*f_b - c*f_c
# Corresponds to c = [a, b, c, F_tilde]
# And forcing fields f_i are the terms multiplying each parameter in the *un-normalized* space.
f_a(x) = x
f_b(x) = x^2
f_c(x) = -x^3
f_F(x) = 1.0
forcing_fields = [f_a, f_b, f_c, f_F]


# 3. Define Conjugate Observables (B_fi) using GFDT
# The score function s(x) = ∇ln(p_ss) = 2F(x)/σ² for this system
# We work with normalized coordinates, so we must transform the score and forcing fields.
# score_norm(x_norm) = s_unnorm(x_unnorm) * S
# f_norm(x_norm) = f_unnorm(x_unnorm) / S
function score(x; p=params_original)
    score_val = 2 * (p[4] + p[1]*x + p[2]*x^2 - p[3]*x^3) / s^2
    # Clamp extreme values to prevent numerical issues
    return score_val
end

# B_fi(x) = - [∇⋅f_i(x) + f_i(x)⋅s(x)]
# For normalized coordinates: B_fi_norm = -[(1/S) * ∇f_i_unnorm + (f_i_unnorm/S) * (s_norm/S)]
# where evaluation is at x_unnorm = x*S+M. After simplifying:
# B_fi(x_norm) = -[∇f_i_unnorm(x*S+M) + f_i_unnorm(x*S+M) * score_unnorm(x*S+M)]

function B_a(x)
    score_val = score(x)
    result = -(1 + x * score_val)
    return result
end

function B_b(x)
    score_val = score(x)
    result = -(2*x + x^2 * score_val)
    return result
end

function B_c(x)
    score_val = score(x)
    result = -(-3*x^2 - x^3 * score_val)
    return result
end

function B_F(x)
    score_val = score(x)
    result = -(0 + 1 * score_val)
    return result
end

conjugate_observables = [B_a, B_b, B_c, B_F]

# ----------------------------------------------------------------------
# --- 4. Compute the Response Matrix (R) 
# ----------------------------------------------------------------------

println("\nCalculating 2x2 Response Matrix R (Optimized)...")
N_moments = length(observables)
N_params = length(conjugate_observables)
R = zeros(N_moments, N_params)
max_lag = 1500  # Number of time steps for correlation integral
lags = 0:dt:max_lag*dt
lag_indices = 0:max_lag  # Integer indices for crosscov

# Storage for plotting the correlation functions, same as before
Rt = zeros(N_moments, N_params, max_lag + 1)

# --- Performance Improvements ---
# 1. Pre-calculate all time series to avoid redundant work inside the loops.
# 2. Use multithreading to parallelize the independent calculations for each R[k, i].
# 3. Replace the manual cross-correlation function with the highly optimized `StatsBase.crosscov`.

# Pre-calculate all observable and conjugate observable time series
println("Pre-calculating time series...")
all_ts_A = [observables[k].(obs[1, :]) for k in 1:N_moments]
all_ts_B = [conjugate_observables[i].(obs[1, :]) for i in 1:N_params]

# Pre-calculate means needed to convert covariance to correlation
mean_A = mean.(all_ts_A)
mean_B = mean.(all_ts_B)

println("Computing response matrix elements in parallel...")
# Use multithreading to compute the R matrix elements.
# This provides a significant speedup on multi-core machines.
Threads.@threads for k in 1:N_moments
    for i in 1:N_params
        # Use the pre-calculated time series
        ts_A = all_ts_A[k]
        ts_B = all_ts_B[i]
        
        # Use StatsBase.crosscov for a fast, optimized calculation.
        # It computes Cov(A(t+τ), B(t)) using integer lag indices.
        cov_series = crosscov(ts_A, ts_B, lag_indices, demean=true)
        
        # Convert covariance to the raw cross-correlation using E[XY] = Cov(X,Y) + E[X]E[Y]
        corr_series = cov_series .+ (mean_A[k] * mean_B[i])
        
        # Store the correlation function for later plotting
        Rt[k, i, :] = corr_series

        # Integrate using the trapezoidal rule. The time step is `dt`.
        # The sum is equivalent to sum(corr[1:end-1] .+ corr[2:end]) / 2 but can be more stable.
        integral_val = dt * (sum(corr_series) - 0.5 * (corr_series[1] + corr_series[end]))
        
        R[k, i] = integral_val
    end
end

# --- Display Results ---
for k=1:N_moments, i=1:N_params
    println("Done. R[$k, $i] = $(round(R[k,i], digits=4))")
end
println("\nComputed Response Matrix R:")
display(R)

plot(lags, Rt[1,4,:], label="R[1,1]", xlabel="Lag", ylabel="Correlation", title="Response Matrix Cross-Correlations")




##
# ----------------------------------------------------------------
# --- Recursive Iterative Algorithm for Moment Control (using Analytic PDF)
# ----------------------------------------------------------------
println("\n--- STARTING RECURSIVE MOMENT CONTROL ---")

# --- Analytic PDF Helper Functions ---
# For a 1D Langevin system, p(x) = N * exp(-U(x)/D) where D = σ²/2
const D_eff = s^2/2 # Effective diffusion coefficient D = σ²/2

# The potential U(x) is the negative integral of the force F(x)
# U(x) = -∫(p4 + p1*y + p2*y² - p3*y³)dy
potential(x, p) = -(p[4]*x + p[1]*x^2/2 + p[2]*x^3/3 - p[3]*x^4/4)

# Unnormalized PDF
p_unnormalized(x, p) = exp(-potential(x, p) / D_eff)

# --- Configuration ---
final_Δμ = [0.5, -0.4]  # Desired final changes: e.g., increase mean by 0.5, decrease variance by 0.5
base_step_Δμ = 0.2 * normalize(final_Δμ)  # Base step size
max_iters = 2500  # Maximum iterations to prevent infinite loop
tolerance = 1e-4  # Stop when actual changes are within this of final_Δμ

# Initial moments
initial_moments = [M, S^2]

# --- Storage for Results ---
predicted_changes = []
actual_changes = []
params_history = [copy(params_original)]
iter = 0

# --- Optimization Configuration ---
Γ = I(N_moments)
λ = 0.01
A = λ * Diagonal(1 ./ abs.(params_original))

# --- Initialize current state ---
curr_params = copy(params_original)
curr_moments = initial_moments
curr_change = [0.0, 0.0]  # Current change from initial

# --- Recursive Loop ---
while iter < max_iters
    iter += 1
    
    # 1. Compute remaining target change
    remaining_Δμ = final_Δμ - curr_change
    
    # Check if target is reached
    if all(abs.(remaining_Δμ) .< tolerance)
        println("Target reached within tolerance.")
        break
    end
    
    # 2. Dynamic step size calculation based on distance to target
    # Scale step size by the fraction of remaining distance, with a minimum step
    distance_fraction = norm(remaining_Δμ) / norm(final_Δμ)
    adaptive_factor = max(0.1, min(1.0, distance_fraction))  # Keep between 0.1 and 1.0
    step_Δμ = adaptive_factor * base_step_Δμ
    
    # Determine this step's Δμ_k (clip to avoid overshooting)
    Δμ_k = sign.(remaining_Δμ) .* min.(abs.(remaining_Δμ), abs.(step_Δμ))
    
    # 3. Calculate the Optimal Parameter Perturbation
    matrix_to_invert = R' * Γ * R + A
    δc_opt = matrix_to_invert \ (R' * Γ * Δμ_k)
    
    # 4. Update parameters
    curr_params .+= δc_opt
    push!(params_history, copy(curr_params))
    
    # 5. Predict and Store the Moment Change (for plotting)
    predicted_change_k = R * δc_opt
    push!(predicted_changes, predicted_change_k + curr_change)  # Cumulative
    
    # Define integrands for the new parameter set
    integrand_norm(x) = p_unnormalized(x, curr_params)
    integrand_mean(x) = x * integrand_norm(x)
    integrand_var(x) = x^2 * integrand_norm(x)  # Use <x^2> and compute variance as <x^2> - mean^2
    
    # Define integration bounds (should be wide enough to capture the PDF)
    int_bounds = (-20.0, 20.0)
    
    # Numerically integrate to find normalization constant, mean, and <x^2>
    norm_const, _ = quadgk(integrand_norm, int_bounds...)
    mean_integral, _ = quadgk(integrand_mean, int_bounds...)
    x2_integral, _ = quadgk(integrand_var, int_bounds...)
    
    # Calculate the moments from the integrals
    mean_new = mean_integral / norm_const
    x2_new = x2_integral / norm_const
    variance_new = x2_new - mean_new^2  # Correct variance calculation
    
    # Update current moments and change
    curr_moments = [mean_new, variance_new]
    curr_change = curr_moments - initial_moments
    push!(actual_changes, copy(curr_change))
end

if iter == max_iters
    println("Warning: Maximum iterations reached without converging to target.")
end

println("\n--- RECURSIVE MOMENT CONTROL COMPLETE ---")
println("Final parameters: $(round.(curr_params, digits=5))")

# -----------------------------------------------------
# --- Plotting Results
# -----------------------------------------------------
println("Generating plots...")

iters = 1:length(actual_changes)

# Set publication-ready defaults
default(fontfamily="Times", 
        linewidth=2, 
        framestyle=:box, 
        grid=true, 
        gridwidth=1, 
        gridcolor=:lightgray, 
        gridalpha=0.3,
        dpi=300)

# Define colors for consistency
color_predicted = :blue
color_actual = :red
color_target = :darkgreen

# --- Panel 1: Mean Change ---
p1 = plot(
    iters, 
    getindex.(predicted_changes, 1),
    label="Linear Response Theory",
    xlabel="Iteration",
    ylabel="Δμ₁ (Change in Mean)",
    legend=:bottomright,
    lw=2.5,
    color=color_predicted,
    markershape=:circle,
    markersize=3,
    markerstrokewidth=0,
    size=(400, 300)
)
plot!(
    p1,
    iters,
    getindex.(actual_changes, 1),
    label="Numerical Integration",
    lw=2.5,
    color=color_actual,
    linestyle=:dash,
    markershape=:square,
    markersize=3,
    markerstrokewidth=0
)
hline!(p1, [final_Δμ[1]], 
       color=color_target, 
       ls=:dot, 
       lw=2, 
       label="Target (μ₁)",
       alpha=0.8)

# Add subtle shading around target
target_band = 0.02 * abs(final_Δμ[1])
plot!(p1, [0, maximum(iters)], [final_Δμ[1] - target_band, final_Δμ[1] - target_band], 
      fillrange=[final_Δμ[1] + target_band, final_Δμ[1] + target_band], 
      alpha=0.1, color=color_target, label="")

# --- Panel 2: Variance Change ---
p2 = plot(
    iters, 
    getindex.(predicted_changes, 2),
    label="Linear Response Theory",
    xlabel="Iteration",
    ylabel="Δμ₂ (Change in Variance)",
    legend=:topright,
    lw=2.5,
    color=color_predicted,
    markershape=:circle,
    markersize=3,
    markerstrokewidth=0,
    size=(400, 300)
)
plot!(
    p2,
    iters,
    getindex.(actual_changes, 2),
    label="Numerical Integration",
    lw=2.5,
    color=color_actual,
    linestyle=:dash,
    markershape=:square,
    markersize=3,
    markerstrokewidth=0
)
hline!(p2, [final_Δμ[2]], 
       color=color_target, 
       ls=:dot, 
       lw=2, 
       label="Target (μ₂)",
       alpha=0.8)

# Add subtle shading around target
target_band = 0.02 * abs(final_Δμ[2])
plot!(p2, [0, maximum(iters)], [final_Δμ[2] - target_band, final_Δμ[2] - target_band], 
      fillrange=[final_Δμ[2] + target_band, final_Δμ[2] + target_band], 
      alpha=0.1, color=color_target, label="")

# --- Create comprehensive title with subtitle ---
main_title = "Recursive Moment Control via Linear Response Theory\n(λ = $λ, Target: Δμ = $(final_Δμ))"

# --- Combine into one professional figure ---
final_plot = plot(p1, p2, 
                 layout=(2, 1), 
                 size=(800, 600),
                 plot_title=main_title,
                 plot_titlefontsize=12,
                 plot_titlefontweight=:bold,
                 margin=5Plots.mm,
                 left_margin=8Plots.mm,
                 bottom_margin=6Plots.mm,
                 top_margin=15Plots.mm)

# Optional: Save high-resolution figure
# savefig(final_plot, "recursive_moment_control.pdf")
# savefig(final_plot, "recursive_moment_control2.png")

display(final_plot)


R1 = copy(R)