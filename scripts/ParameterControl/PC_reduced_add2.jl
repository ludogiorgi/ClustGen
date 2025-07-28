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
D_eff = s^2 / 2

# Store original parameters in a vector
params_original = [a, b, c, F_tilde]
# params_initial = [a, F_tilde]

# Langevin equation force and noise functions
F(x, t; p = params_original) = [p[4] + p[1] * x[1] + p[2] * x[1]^2 - p[3] * x[1]^3]
sigma(x, t) = s / √2

# --- Generate data from the unperturbed system ---
dt_sim = 0.01
res = 2
dt = dt_sim * res
Nsteps = 20_000_000 # Use a long trajectory for accurate statistics
println("Generating unperturbed time series...")
obs = evolve([0.0], dt_sim, Nsteps, F, sigma; timestepper=:euler, resolution=res)
plot(obs[1, end-2000:end], label="Unperturbed System", xlabel="Time", ylabel="Observable", title="Unperturbed Langevin Dynamics")
##
M = mean(obs, dims=2)[1]
S = std(obs, dims=2)[1]
#Skew = mean((obs[1, :] .- M) .^ 3)  # Remove dims parameter and use flattened array
println("Unperturbed System: Mean = $(round(mean(obs), digits=4)), Variance = $(round(var(obs), digits=4))")


println("Testing with smaller target changes to validate linear response theory...")

# 2. Define Observables (A_k) and Forcing Fields (f_i)
# Observables (for standardized data where mean=0, var=1):
A_1(x) = (x)       # Observable for the mean
A_2(x) = (x - M).^2       # Observable for the variance (since mean is 0)
#A_3(x) = (x - M).^3       # Observable for the skewness (not used in this example)
observables = [A_1, A_2]

function score(x; p=params_original)
    # Handle both scalar and vector inputs
    x_val = x isa Vector ? x[1] : x
    score_val = 2 * (p[4] + p[1]*x_val + p[2]*x_val^2 - p[3]*x_val^3) / s^2
    
    return score_val
end

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

params_initial = [a, F_tilde]
conjugate_observables = [B_a, B_F]
#conjugate_observables = [B_a, B_F]

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
        cov_series = crosscov(ts_A, ts_B, lag_indices; demean=true)
        
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


plot(lags, Rt[2,2,:], label="R[1,1]", xlabel="Lag", ylabel="Correlation", title="Response Matrix Cross-Correlations")


# Function to compute moments via numeric integration
function compute_moments(p)
    integrand_norm(x) = p_unnormalized(x, p)
    int_bounds = (-20.0, 20.0)
    norm_const, _ = quadgk(integrand_norm, int_bounds...)
    
    integrand_mean(x) = x * integrand_norm(x)
    mean_integral, _ = quadgk(integrand_mean, int_bounds...)
    mean_val = mean_integral / norm_const

    integrand_x2(x) = (x)^2 * integrand_norm(x)
    x2_integral, _ = quadgk(integrand_x2, int_bounds...)
    x2_val = x2_integral / norm_const
    var_val = x2_val - mean_val^2
    
    return [mean_val, var_val]
end


potential(x, p) = -(p[4] * x + p[1] * x^2 / 2 + p[2] * x^3 / 3 - p[3] * x^4 / 4)
p_unnormalized(x, p) = exp(-potential(x, p) / D_eff)

function compute_jacobian(p; ε=1e-5)
    J = zeros(N_moments, N_params)
    base_moments = compute_moments(p)
    
    for i in 1:N_params
        # Positive perturbation
        p_pos = copy(p)
        p_pos[i] += ε
        moments_pos = compute_moments(p_pos)
        
        # Negative perturbation
        p_neg = copy(p)
        p_neg[i] -= ε
        moments_neg = compute_moments(p_neg)
        
        # Central difference
        J[:, i] = (moments_pos - moments_neg) / (2 * ε)
    end
    return J
end

J = compute_jacobian(params_original; ε=1e-5)
R
##
# ----------------------------------------------------------------
# --- Sensitivity Analysis Loop (using Analytic PDF)
# ----------------------------------------------------------------
println("\n--- STARTING SENSITIVITY ANALYSIS (using Analytic PDF) ---")

# --- Analytic PDF Helper Functions ---
# For a 1D Langevin system, p(x) = N * exp(-U(x)/D) where D = σ²/2
D_eff = s^2/2 # Effective diffusion coefficient D = σ²/2

# The potential U(x) is the negative integral of the force F(x)
# U(x) = -∫(p4 + p1*y + p2*y² - p3*y³)dy
potential(x, p) = -(p[4]*x + p[1]*x^2/2 + p[2]*x^3/3 - p[3]*x^4/4)

# Unnormalized PDF
p_unnormalized(x, p) = exp(-potential(x, p) / D_eff)

# --- Loop Configuration ---
n_control = 100 # Number of control steps to run
base_Δμ = [0.01, 0.01] # Base target change for k=1

# --- Storage for Results ---
predicted_changes = zeros(N_moments, n_control)
actual_changes = zeros(N_moments, n_control)
control_steps = 0:n_control-1
# --- Optimization Configuration ---
Γ = I(N_moments)
λ = 0.1
A = λ * Diagonal(1 ./ abs.(params_initial))
# A[3,3] = 10.0
# A[2,2] = 10.0
RJ = copy(R)
#R = R1[:, [1,4]]
# --- Main Loop ---
for k in control_steps
    
    # 1. Scale the target moment change
    Δμ_k = k * base_Δμ
    
    # 2. Calculate the Optimal Parameter Perturbation
    matrix_to_invert = RJ' * Γ * RJ + A
    δc_opt = matrix_to_invert \ (RJ' * Γ * Δμ_k)

    
    # 3. Predict and Store the Moment Change
    predicted_change_k = RJ * δc_opt
    predicted_changes[:, k+1] = predicted_change_k
    
    # Create the new full parameter vector (need all 4 parameters for potential function)
    params_new_full = copy(params_original)  # Start with original [a, b, c, F_tilde]
    # Only update the parameters we're optimizing over (indices corresponding to params_initial)
    params_new_full[1] += δc_opt[1]  # Update 'a' parameter
    params_new_full[4] += δc_opt[2]  # Update 'F_tilde' parameter
    
    # Define integrands for the new parameter set
    integrand_norm(x) = p_unnormalized(x, params_new_full)
    integrand_mean(x) = (x) * integrand_norm(x)
    integrand_var(x) = (x - M)^2 * integrand_norm(x)
    #integrand_skew(x) = (x - M)^3 * integrand_norm(x)
    
    # Define integration bounds (should be wide enough to capture the PDF)
    int_bounds = (-20.0, 20.0)
    
    # Numerically integrate to find normalization constant, mean, and variance
    norm_const, _ = quadgk(integrand_norm, int_bounds...)
    mean_integral, _ = quadgk(integrand_mean, int_bounds...)
    var_integral, _ = quadgk(integrand_var, int_bounds...)
    #skew_integral, _ = quadgk(integrand_skew, int_bounds...)
    
    # Calculate the moments from the integrals
    mean_new = mean_integral / norm_const
    variance_new = var_integral / norm_const  # <x²>_new
    #skewness_new = skew_integral / norm_const  # <x³>_new
    actual_change_k = [mean_new - M, variance_new - S^2]
    # actual_change_k = [mean_new - M]
    actual_changes[:, k+1] = actual_change_k
    println("Actual change (from PDF): $(round.(actual_change_k, digits=5))")
end

println("\n--- SENSITIVITY ANALYSIS COMPLETE ---")

# -----------------------------------------------------
# --- Plotting Results
# -----------------------------------------------------
println("Generating plots...")

# --- Panel 1: Mean Change ---
p1 = plot(
    control_steps .* base_Δμ[1], 
    predicted_changes[1, :],
    label="Predicted",
    xlabel="Δμ₁",
    ylabel="Change in Mean",
    title="v=$(round.(normalize(base_Δμ), digits=2))," * " λ=$λ",
    legend=:topleft,
    lw=2
)
plot!(
    p1,
    control_steps .* base_Δμ[1],
    actual_changes[1, :],
    label="Actual",
    lw=2,
    ls=:dash,
    marker=:xcross,
    markersize=5
)

# --- Panel 2: Variance Change ---
p2 = plot(
    control_steps .* base_Δμ[2],
    predicted_changes[2, :],
    legend = false,
    xlabel="Δμ₂",
    ylabel="Change in Variance",
    lw=2
)
plot!(
    p2,
    control_steps .* base_Δμ[2],
    actual_changes[2, :],
    lw=2,
    ls=:dash,
    marker=:xcross,
    markersize=5
)

# --- Combine into one figure and display ---
final_plot = plot(p1, p2, layout=(2, 1), size=(800, 1200))
#savefig(final_plot, "sensitivity_analysis_results.png")
display(final_plot)


