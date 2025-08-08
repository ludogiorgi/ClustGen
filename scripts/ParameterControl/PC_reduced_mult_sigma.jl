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
using GLMakie
using CairoMakie
using LinearAlgebra
using ProgressBars
using Random
using QuadGK
using Base.Threads
using StatsBase
##
# --- Triad Model System Definition ---

# Base parameters
const L11 = -2.0; const L12 = 0.2; const L13 = 0.1; const g2 = 0.6; const g3 = 0.4;
const s2_param = 1.2; const s3 = 0.8; const II = 1.0; const ϵ = 0.1; # Renamed s2 to avoid conflict

# Coefficients of the reduced model
const a = L11 + ϵ * ( (II^2 * s2_param^2) / (2 * g2^2) - (L12^2) / g2 - (L13^2) / g3 )
const b = -2 * (L12 * II) / g2 * ϵ
const c = (II^2) / g2 * ϵ
const B = -(II * s2_param) / g2 * sqrt(ϵ)
const A = -(L12 * B) / II
const s_noise = (L13 * s3) / g3 * sqrt(ϵ)
const F_tilde = (A * B) / 2

# Store all parameters in a single vector for easy access and perturbation
# Order: [a, b, c, F_tilde, A, B, s_noise]
const params_triad = [a, b, c, F_tilde, A, B, s_noise]

# Deterministic drift function F(x)
F(x, t; p=params_triad) = [-p[4] + p[1] * x[1] + p[2] * x[1]^2 - p[3] * x[1]^3]
# Multiplicative noise term sigma_1(x)
sigma1(x, t; p=params_triad) = (p[5] - p[6] * x[1]) / √2
# Additive noise term sigma_2(x)
sigma2(x, t; p=params_triad) = p[7] / √2

# --- Score Functions & Derivatives ---

function create_true_score_and_derivative_triad(p)
    a, b, c, F_tilde, A_p, B_p, s_val = p
    den(u) = s_val^2 + (A_p - B_p*u)^2              # = σ^2(u)
    f(u)   = -F_tilde + a*u + b*u^2 - c*u^3         # drift
    denp(u) = 2*(A_p - B_p*u)*(-B_p)                # d/du σ^2(u)
    num(u)  = 2*f(u) + 2*B_p*(A_p - B_p*u)          # 2f - (σ^2)' = 2f + 2B(A-Bu)
    nump(u) = 2*(a + 2*b*u - 3*c*u^2) + 2*B_p*(-B_p)

    score_func(x) = num(x) / den(x)
    score_derivative_func(u) = (nump(u)*den(u) - num(u)*denp(u)) / den(u)^2
    return score_func, score_derivative_func
end

function create_linear_s_ds(x_t)
    μ = mean(x_t)
    variance = var(x_t)
    score_func(x) = -(x - μ) / variance
    score_derivative(x) = -1 / variance
    return score_func, score_derivative
end

_to_scalar(v) = v isa Number ? v :
    (v isa AbstractArray && length(v) == 1 ? (ndims(v) == 2 ? v[1,1] : v[1]) :
        error("Score function returned non-scalar value of size $(size(v))"))

function construct_divergence_score(s::Function; n_points::Int=500, range::Tuple{Real,Real}=(-5.0, 5.0))
    # Local scalarizing wrapper
    s_scalar(x) = _to_scalar(s(x))
    x_grid = Base.range(range[1], range[2], length=n_points)
    divergence_values = similar(x_grid)
    h = 1e-6
    @threads for i in eachindex(x_grid)
        sp = s_scalar(x_grid[i] + h)
        sm = s_scalar(x_grid[i] - h)
        divergence_values[i] = (sp - sm) / (2 * h)
    end
    function fast_divergence_func(x_val::Real)
        if x_val < range[1] return divergence_values[1] end
        if x_val > range[2] return divergence_values[end] end
        idx_float = (x_val - range[1]) / step(x_grid) + 1
        idx_low = floor(Int, idx_float)
        idx_high = min(idx_low + 1, n_points)
        idx_low = max(1, idx_low)
        t = idx_float - idx_low
        return divergence_values[idx_low] * (1 - t) + divergence_values[idx_high] * t
    end
    return fast_divergence_func
end

# --- Potentials and Jacobians (for analytical comparison) ---

function potential_triad(x, p)
    a_p, b_p, c_p, F_tilde_p, A_p, B_p, s_val_p = p
    sigma_sq(y) = (A_p - B_p*y)^2 + s_val_p^2
    drift(y) = -F_tilde_p + a_p*y + b_p*y^2 - c_p*y^3
    integral_part, _ = quadgk(y -> 2 * drift(y) / sigma_sq(y), 0, x, rtol=1e-6)
    return -integral_part + log(sigma_sq(x))
end

p_unnormalized_triad(x, p) = exp(-potential_triad(x, p))

function compute_observables_triad(p, observables)
    int_bounds = (-6, 6)
    norm_const, _ = quadgk(x -> p_unnormalized_triad(x, p), int_bounds..., rtol=1e-8)
    if norm_const == 0.0 error("Normalization constant is zero. Check potential function or integration bounds.") end
    map(observables) do obs
        obs_integral, _ = quadgk(x -> obs(x) * p_unnormalized_triad(x, p), int_bounds..., rtol=1e-8)
        obs_integral / norm_const
    end
end

function compute_jacobian_triad(p, param_indices, observables; ε=1e-5)
    J = zeros(length(observables), length(param_indices))
    for (i, p_idx) in enumerate(param_indices)
        p_plus, p_minus = copy(p), copy(p)
        p_plus[p_idx] += ε
        p_minus[p_idx] -= ε
        obs_plus = compute_observables_triad(p_plus, observables)
        obs_minus = compute_observables_triad(p_minus, observables)
        J[:, i] = (obs_plus - obs_minus) / (2 * ε)
    end
    return J
end

# --- Conjugate Observables & Response Matrix ---

function create_conjugate_observables_triad(score, score_derivative, p)
    A_p, B_p, s_val_p = p[5], p[6], p[7]
    s_func = score
    s_prime = score_derivative

    # Conjugate observable for perturbing A (diffusion only)
    B_A(x) = 2 * B_p * s_func(x) - (A_p - B_p * x) * (s_func(x)^2 + s_prime(x))

    # Conjugate observable for perturbing B (diffusion only)
    B_B(x) = -2 * B_p - 2 * s_func(x) * (-A_p + 2 * B_p * x) - (s_func(x)^2 + s_prime(x)) * (-A_p * x + B_p * x^2)

    # Conjugate observable for perturbing s_noise (diffusion only)
    B_s(x) = -s_val_p * (s_func(x)^2 + s_prime(x))

    return [B_A, B_B, B_s]
end


function create_response_matrix(time_series, dt, observables, conjugate_observables; max_lag_time=30.0)
    N_moments = length(observables)
    N_params = length(conjugate_observables)
    
    lag_indices = 0:floor(Int, max_lag_time / dt)
    lags = lag_indices .* dt

    R = zeros(N_moments, N_params)
    Rt = zeros(N_moments, N_params, length(lag_indices))

    all_ts_A = Vector{Vector{Float64}}(undef, length(observables))
    all_ts_B = Vector{Vector{Float64}}(undef, length(conjugate_observables))
    
    @threads for k in eachindex(observables)
        all_ts_A[k] = observables[k].(time_series)
    end

    @threads for i in eachindex(conjugate_observables)
        all_ts_B[i] = conjugate_observables[i].(time_series)
    end

    @threads for k in 1:N_moments
        for i in 1:N_params
            ts_A = all_ts_A[k]
            ts_B = all_ts_B[i]
            corr_series = .- crosscov(ts_A, ts_B, lag_indices; demean=true)
            Rt[k, i, :] = corr_series
            integral_val = dt * (sum(corr_series) - 0.5 * (corr_series[1] + corr_series[end]))
            R[k, i] = integral_val
        end
    end
    return R, Rt, lags
end

##

# --- Main Execution Script ---

# 1. Generate baseline time series data
println("Generating baseline time series data for the triad model...")
dt_sim = 0.01
N_steps = 20_000_000
time_series_triad = evolve([0.0], dt_sim, N_steps, F, sigma1, sigma2; timestepper=:rk4)[1,:]
println("Data generation complete.")

# 2. Define observables and control parameters
observables_to_control = [x -> x^3] # Control mean and second moment
observable_labels = ["Mean", "Variance"]
# MODIFIED: Now controlling A, B, and s_noise
param_indices_to_control = [5, 6, 7] # Indices for A, B, s_noise
param_labels = ["δA", "δB", "δs_noise"]

# 3. Calculate initial state
initial_observables = compute_observables_triad(params_triad, observables_to_control)

# 4. Compute all three types of response matrices (now 2x3)
# A) Using the TRUE score function
println("\n--- Computing R_true ---");
true_score, true_score_deriv = create_true_score_and_derivative_triad(params_triad);
conjugate_obs_true = create_conjugate_observables_triad(true_score, true_score_deriv, params_triad);
R_true, R_true_t, _ = create_response_matrix(time_series_triad, dt_sim, observables_to_control, conjugate_obs_true)

# B) Using the LINEAR score function
println("\n--- Computing R_linear ---");
linear_score, linear_score_deriv = create_linear_s_ds(time_series_triad);
conjugate_obs_linear = create_conjugate_observables_triad(linear_score, linear_score_deriv, params_triad);
R_linear, R_linear_t, _ = create_response_matrix(time_series_triad, dt_sim, observables_to_control, conjugate_obs_linear);

# C) Using the CLUSTERED (KGMM) score function
println("\n--- Computing R_clustered ---");
kgmm_results = calculate_score_kgmm(reshape(time_series_triad, 1, :); σ_value=0.05, clustering_prob=0.0005, verbose=false);
raw_score_clustered = kgmm_results.score_function
score_clustered = x -> _to_scalar(raw_score_clustered(x))
score_clustered_derivative = construct_divergence_score(score_clustered)
conjugate_obs_clustered = create_conjugate_observables_triad(score_clustered, score_clustered_derivative, params_triad);
R_clustered, R_clustered_t, _ = create_response_matrix(time_series_triad, dt_sim, observables_to_control, conjugate_obs_clustered)

# D) Using the ANALYTICAL Jacobian
println("\n--- Computing J_analytical ---");
J_analytical = compute_jacobian_triad(params_triad, param_indices_to_control, observables_to_control; ε=1e-3)

##
println("\n--- COMPARISON OF MATRICES (now 2x3) ---")
println("Response Matrix R (from true score):"); display(R_true)
println("\nResponse Matrix R (from linear score):"); display(R_linear)
println("\nResponse Matrix R (from clustered/KGMM score):"); display(R_clustered)
println("\nJacobian J (from analytic PDF):"); display(J_analytical)
println("--------------------------------\n")
##
using Plots
idx1 = 1 # Index for the first observable (mean)
idx2 = 3 # Index for the second observable (variance)

Plots.plot(R_true_t[idx1,idx2,:])
#Plots.plot!(R_linear_t[idx1,idx2,:])
Plots.plot!(R_clustered_t[idx1,idx2,:])
