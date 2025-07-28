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

# Function to compute moments via numeric integration
function compute_moments(p)
    integrand_norm(x) = p_unnormalized(x, p)
    int_bounds = (-20.0, 20.0)
    norm_const, _ = quadgk(integrand_norm, int_bounds...)
    
    integrand_mean(x) = x * integrand_norm(x)
    mean_integral, _ = quadgk(integrand_mean, int_bounds...)
    mean_val = mean_integral / norm_const

    integrand_x2(x) = (x^2 * integrand_norm(x))
    x2_integral, _ = quadgk(integrand_x2, int_bounds...)
    x2_val = x2_integral / norm_const
    var_val = x2_val - mean_val^2
    
    return [mean_val, var_val]
end

potential(x, p) = -(p[4] * x + p[1] * x^2 / 2 + p[2] * x^3 / 3 - p[3] * x^4 / 4)
p_unnormalized(x, p) = exp(-potential(x, p) / D_eff)

a = -0.0222
b = -0.2
c = 0.0494
F_tilde = 0.6
s = 0.7071
D_eff = s^2 / 2

params_original = [a, b, c, F_tilde]

f_a(x) = x
f_b(x) = x^2
f_c(x) = -x^3
f_F(x) = 1.0
forcing_fields = [f_a, f_b, f_c, f_F]
N_params = 4

function score(x; p=params_original)
    score_val = 2 * (p[4] + p[1]*x + p[2]*x^2 - p[3]*x^3) / s^2
    return score_val
end

function Bs_a(x)
    score_val = score(x)
    result = -(1 + x * score_val)
    return result
end

function Bs_b(x)
    score_val = score(x)
    result = -(2*x + x^2 * score_val)
    return result
end

function Bs_c(x)
    score_val = score(x)
    result = -(-3*x^2 - x^3 * score_val)
    return result
end

function Bs_F(x)
    score_val = score(x)
    result = -(0 + 1 * score_val)
    return result
end

conjugate_observables_score = [Bs_a, Bs_b, Bs_c, Bs_F]

# Function to compute Jacobian J = dμ/dc numerically (central finite differences)
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

# --- Configuration ---
final_Δμ = [0.5, 0.5]  # Desired final changes
initial_moments = compute_moments(params_original)
target_moments = initial_moments + final_Δμ
max_iters = 10  
tolerance = 1e-4
N_moments = 2
Γ = I(N_moments)
λ = 1.0e-4
A = λ * Diagonal(1 ./ abs.(params_original))
A[3,3] = 1.0

# --- Storage for Results ---
actual_changes = []
params_history = [copy(params_original)]
iter = 0

# --- Initialize current state ---
curr_params = copy(params_original)
curr_moments = initial_moments
curr_change = [0.0, 0.0]
##

function compute_response_matrix_optimized(
    x_t, dt, params; 
    n_moments=2,
    max_lag_fraction=0.1
)
    num_params = length(params)
    n_samples = length(x_t)

    # --- Pre-computation Step ---
    μ = mean(x_t)
    observables = [x -> x, x -> (x - μ)^2]
    
    s_val = 0.7071 # Assuming s is fixed from your script
    score_fn(x) = 2 * (params[4] + params[1]*x + params[2]*x^2 - params[3]*x^3) / s_val^2
    
    # Optimization 1: Pre-calculate the score for the entire time series at once.
    score_t = score_fn.(x_t)

    # Optimization 2: Redefine conjugate observables to accept the pre-calculated score.
    conjugate_observables = [
        (x, s) -> -(1 + x * s),
        (x, s) -> -(2*x + x^2 * s),
        (x, s) -> -(-3*x^2 - x^3 * s),
        (x, s) -> -(0 + 1 * s)
    ]

    A_t = [obs.(x_t) for obs in observables]
    # Use the pre-calculated score to generate B_t time series efficiently.
    B_t = [conj_obs.(x_t, score_t) for conj_obs in conjugate_observables]

    # --- Parallel Calculation Step ---
    R = zeros(n_moments, num_params)
    max_lag = floor(Int, n_samples * max_lag_fraction)
    lags = 0:max_lag

    # Optimization 3: Parallelize the main loop over all (k, i) pairs.
    # Each thread computes a different element of the R matrix.
    Threads.@threads for I in CartesianIndices(R)
        k, i = I.I # Deconstruct CartesianIndex into (k, i)

        # Calculate cross-covariance: C(τ) = <A_k(t+τ) B_i(t)>
        cross_cov = crosscov(A_t[k], B_t[i], lags, demean=true)
        
        # Integrate using the trapezoidal rule: ∫ C(τ) dτ ≈ sum(C) * dt
        integral_value = sum(cross_cov) * dt
        
        R[I] = integral_value # Assign to R[k, i] in a thread-safe way
    end
    
    return R
end


J = compute_jacobian(curr_params; ε=1e-5)
R = compute_response_matrix_optimized(obs[1, :], dt, curr_params)
##

# --- Newton's Method Loop ---
while iter < max_iters
    iter += 1
    println("\n--- Iteration $iter ---")
    
    # Compute current error (Δμ needed)
    remaining_Δμ = target_moments - curr_moments
    if all(abs.(remaining_Δμ) .< tolerance)
        println("Target reached within tolerance.")
        break
    end
    println("Remaining Δμ: $(round.(remaining_Δμ, digits=5))")
    
    # Compute Jacobian at current parameters
    J = compute_response_matrix_analytic(curr_params)
    # Compute Newton step (regularized for underdetermined system)
    matrix_to_invert = J' * Γ * J + A
    δc = matrix_to_invert \ (J' * Γ * remaining_Δμ)
    
    # Simple line search: find step size α that reduces error
    α = 1.0
    trial_params = curr_params + α * δc
    trial_moments = compute_moments(trial_params)
    trial_error = norm(target_moments - trial_moments)
    curr_error = norm(target_moments - curr_moments)
    
    while trial_error > curr_error && α > 1e-4
        α *= 0.5
        trial_params = curr_params + α * δc
        trial_moments = compute_moments(trial_params)
        trial_error = norm(target_moments - trial_moments)
    end
    
    # Update if improvement found
    if trial_error < curr_error
        curr_params = trial_params
        curr_moments = trial_moments
        curr_change = curr_moments - initial_moments
        push!(actual_changes, copy(curr_change))
        push!(params_history, copy(curr_params))
        println("Step accepted with α=$α. New change: $(round.(curr_change, digits=5))")
    else
        println("No improvement; reducing damping.")
        λ *= 2  # Increase damping if stuck
    end
end

if iter == max_iters
    println("Warning: Maximum iterations reached without converging to target.")
end

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
color_actual = :red
color_target = :darkgreen

# --- Panel 1: Mean Change ---
p1 = plot(
    iters, 
    getindex.(actual_changes, 1),
    label="Numerical Integration",
    xlabel="Iteration",
    ylabel="Δμ₁ (Change in Mean)",
    legend=:bottomright,
    lw=2.5,
    color=color_actual,
    markershape=:square,
    markersize=3,
    markerstrokewidth=0,
    size=(400, 300)
)
hline!(p1, [final_Δμ[1]], 
       color=color_target, 
       ls=:dot, 
       lw=2, 
       label="Target (μ₁)",
       alpha=0.8)

# Add subtle shading around target
# target_band = 0.02 * abs(final_Δμ[1])
# plot!(p1, [0, maximum(iters)], [final_Δμ[1] - target_band, final_Δμ[1] - target_band], 
#       fillrange=[final_Δμ[1] + target_band, final_Δμ[1] + target_band], 
#       alpha=0.1, color=color_target, label="")

# --- Panel 2: Variance Change ---
p2 = plot(
    iters, 
    getindex.(actual_changes, 2),
    label="Numerical Integration",
    xlabel="Iteration",
    ylabel="Δμ₂ (Change in Variance)",
    legend=:topright,
    lw=2.5,
    color=color_actual,
    markershape=:square,
    markersize=3,
    markerstrokewidth=0,
    size=(400, 300)
)
hline!(p2, [final_Δμ[2]], 
       color=color_target, 
       ls=:dot, 
       lw=2, 
       label="Target (μ₂)",
       alpha=0.8)

# Add subtle shading around target
# target_band = 0.02 * abs(final_Δμ[2])
# plot!(p2, [0, maximum(iters)], [final_Δμ[2] - target_band, final_Δμ[2] - target_band], 
#       fillrange=[final_Δμ[2] + target_band, final_Δμ[2] + target_band], 
#       alpha=0.1, color=color_target, label="")

# --- Create comprehensive title with subtitle ---
main_title = "Accelerated Recursive Moment Control via Newton's Method\n(λ_init = $λ, Target: Δμ = $(final_Δμ))"

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
# savefig(final_plot, "accelerated_moment_control.pdf")
savefig(final_plot, "accelerated_moment_control1.png")

display(final_plot)



