# -----------------------------------------------------
# --- Environment Setup
# -----------------------------------------------------
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using ClustGen          # Provides the `evolve` function
using MarkovChainHammer
using Statistics
using LinearAlgebra
using Interpolations
using QuadGK
using Plots
using KernelDensity
using StatsBase
using Base.Threads

##
# -----------------------------------------------------
# --- 1. Core Function Definitions
# -----------------------------------------------------

"""
    potential(x, p)
Calculates the potential U(x) for the Langevin system.
"""
potential(x, p) = -(p[4]*x + p[1]*x^2/2 + p[2]*x^3/3 - p[3]*x^4/4)

"""
    p_unnormalized(x, p; D_eff)
Calculates the unnormalized steady-state probability distribution p(x) ~ exp(-U(x)/D).
"""
p_unnormalized(x, p; D_eff) = exp(-potential(x, p) / D_eff)

"""
    compute_moments(p; D_eff)
Calculates the first two moments (mean, variance) by numerically integrating over the analytical probability distribution.
"""
function compute_moments(p; D_eff)
    integrand_norm(x) = p_unnormalized(x, p; D_eff=D_eff)
    int_bounds = (-20.0, 20.0)
    norm_const, _ = quadgk(integrand_norm, int_bounds...)
    
    integrand_mean(x) = x * integrand_norm(x)
    mean_integral, _ = quadgk(integrand_mean, int_bounds...)
    mean_val = mean_integral / norm_const

    integrand_x2(x) = x^2 * integrand_norm(x)
    x2_integral, _ = quadgk(integrand_x2, int_bounds...)
    x2_val = x2_integral / norm_const
    var_val = x2_val - mean_val^2
    
    return [mean_val, var_val]
end

"""
    create_conjugate_observable_diffusion(p, s_val)
Creates the conjugate observable for a diffusion perturbation.
"""
function create_conjugate_observable_diffusion(p, s_val)
    exact_score(x) = 2 * (p[4] + p[1]*x + p[2]*x^2 - p[3]*x^3) / s_val^2
    exact_score_derivative(x) = 2 * (p[1] + 2*p[2]*x - 3*p[3]*x^2) / s_val^2
    B_diff(x) = s_val * (exact_score_derivative(x) + exact_score(x)^2)
    return B_diff
end

"""
    create_response_matrix(time_series, dt, observables, conjugate_observables; max_lag_time=30.0)
Calculates the response matrix R by integrating the cross-correlation function.
"""
function create_response_matrix(time_series, dt, observables, conjugate_observables; max_lag_time=30.0)
    N_moments = length(observables)
    N_params = length(conjugate_observables)
    lag_indices = 0:floor(Int, max_lag_time / dt)
    R = zeros(N_moments, N_params)

    all_ts_A = [obs.(time_series) for obs in observables]
    all_ts_B = [conj_obs.(time_series) for conj_obs in conjugate_observables]

    @threads for k in 1:N_moments
        for i in 1:N_params
            corr_series = crosscov(all_ts_A[k], all_ts_B[i], lag_indices; demean=false)
            integral_val = dt * (sum(corr_series) - 0.5 * (corr_series[1] + corr_series[end]))
            R[k, i] = integral_val
        end
    end
    return R
end

"""
    run_sensitivity_analysis_s(R_s_scalar, params_original, s_initial, initial_variance; n_control=20, max_total_change=0.1, λ=0.0)
Performs a sensitivity analysis for the diffusion parameter 's'.
"""
function run_sensitivity_analysis_s(R_s_scalar, params_original, s_initial, initial_variance; n_control=20, max_total_change=0.1, λ=0.0)
    target_changes = range(0, max_total_change, length=n_control)
    predicted_s_changes = zeros(n_control)
    actual_var_changes = zeros(n_control)

    for (i, Δvar) in enumerate(target_changes)
        # Predict change in s needed
        δs_predicted = Δvar / (R_s_scalar + λ)
        predicted_s_changes[i] = δs_predicted
        
        # Calculate actual variance for this new s
        s_new = s_initial + δs_predicted
        D_eff_new = s_new^2 / 2
        new_moments = compute_moments(params_original; D_eff=D_eff_new)
        actual_var_changes[i] = new_moments[2] - initial_variance
    end
    return target_changes, actual_var_changes
end

"""
    run_newton_optimization_s(s_initial, params_original, initial_variance, target_variance; 
                              max_iters=15, tolerance=1e-5, λ=0.0, recalculate_R=false, sim_args=())
Performs Newton's method to find the value of 's' that matches a target variance.
"""
function run_newton_optimization_s(s_initial, params_original, initial_variance, target_variance; 
                                   max_iters=15, tolerance=1e-5, λ=0.0, recalculate_R=false, sim_args=())
    
    method_name = recalculate_R ? "Full Newton" : "Quasi-Newton"
    println("\n--- Starting Optimization for 's': $method_name ---")
    
    s_history = [s_initial]
    var_history = [initial_variance]
    
    s_current = s_initial
    var_current = initial_variance

    # Initial R calculation
    println("Calculating initial response R(s)...")
    local R_s_current
    let
        F_sim_local(x, t; p=params_original) = [p[4] + p[1] * x[1] + p[2] * x[1]^2 - p[3] * x[1]^3]
        sigma_sim_local(x, t) = s_current / √2
        local_obs = evolve([0.0], sim_args.dt_sim, sim_args.Nsteps, F_sim_local, sigma_sim_local; timestepper=:euler, resolution=sim_args.res)[1, :]
        local_M_obs = mean(local_obs)
        local_observable_var = [x -> (x - local_M_obs)^2]
        local_conj_obs = create_conjugate_observable_diffusion(params_original, s_current)
        R_s_current = create_response_matrix(local_obs, sim_args.dt_obs, local_observable_var, [local_conj_obs])[1, 1]
    end

    for iter in 1:max_iters
        println("--- Iteration $iter ---")
        
        remaining_Δvar = target_variance - var_current
        if abs(remaining_Δvar) < tolerance
            println("Target variance reached within tolerance.")
            break
        end

        if recalculate_R && iter > 1
            println("Recalculating response R(s) for s = $(round(s_current, digits=4))...")
            F_sim_local(x, t; p=params_original) = [p[4] + p[1] * x[1] + p[2] * x[1]^2 - p[3] * x[1]^3]
            sigma_sim_local(x, t) = s_current / √2
            local_obs = evolve([0.0], sim_args.dt_sim, sim_args.Nsteps, F_sim_local, sigma_sim_local; timestepper=:euler, resolution=sim_args.res)[1, :]
            local_M_obs = mean(local_obs)
            local_observable_var = [x -> (x - local_M_obs)^2]
            local_conj_obs = create_conjugate_observable_diffusion(params_original, s_current)
            R_s_current = create_response_matrix(local_obs, sim_args.dt_obs, local_observable_var, [local_conj_obs])[1, 1]
        end
        
        # Newton's step
        δs = remaining_Δvar / (R_s_current + λ)
        s_current += δs
        
        # Update variance
        D_eff_new = s_current^2 / 2
        var_current = compute_moments(params_original; D_eff=D_eff_new)[2]
        
        push!(s_history, s_current)
        push!(var_history, var_current)
        println("New s: $(round(s_current, digits=5)), New variance: $(round(var_current, digits=5))")
        
        if iter == max_iters println("Warning: Maximum iterations reached.") end
    end
    
    return s_history, var_history
end

# -----------------------------------------------------
# --- 2. Main Script Execution
# -----------------------------------------------------

# --- System & Simulation Parameters ---
a = -0.0222; b = -0.2; c = 0.0494; F_tilde = 0.6;
s_initial = 0.7071
params_original = [a, b, c, F_tilde]

F_sim(x, t; p=params_original) = [p[4] + p[1] * x[1] + p[2] * x[1]^2 - p[3] * x[1]^3]
sigma_sim(x, t) = s_initial / √2

# --- Generate Simulation Data ---
dt_sim = 0.01; res = 10; dt_obs = dt_sim * res; Nsteps = 20_000_000
println("Generating unperturbed time series...")
obs_timeseries = evolve([0.0], dt_sim, Nsteps, F_sim, sigma_sim; timestepper=:euler, resolution=res)[1, :]
println("Time series generation complete.")

# --- Get Sample Moments (for reference) and Analytical Moments (for baseline) ---
# FIX: Use analytical moments as the true baseline to avoid sample error issues.
initial_moments_analytical = compute_moments(params_original; D_eff=(s_initial^2/2))
M_analytical = initial_moments_analytical[1]
initial_variance_analytical = initial_moments_analytical[2]

println("Sample variance from time series: $(round(var(obs_timeseries), digits=5))")
println("Analytical variance for initial parameters: $(round(initial_variance_analytical, digits=5))")


##
###################################################################################
### SECTION: Sensitivity Analysis for Diffusion Parameter 's'
###################################################################################
println("\n\n--- STARTING SENSITIVITY ANALYSIS FOR 's' ---")

# --- Compute initial response ---
# The observable for variance should be centered around the true analytical mean
conjugate_obs_s_initial = create_conjugate_observable_diffusion(params_original, s_initial)
observable_variance = [x -> (x - M_analytical)^2]
R_s_initial = create_response_matrix(obs_timeseries, dt_obs, observable_variance, [conjugate_obs_s_initial])[1, 1]
println("Initial Response ∂<var>/∂s = $(round(R_s_initial, digits=4))")

# --- Run and Plot Sensitivity Analysis ---
# FIX: Pass the ANALYTICAL variance as the baseline for a correct comparison at the origin.
target_changes, actual_var_changes = run_sensitivity_analysis_s(R_s_initial, params_original, s_initial, initial_variance_analytical)

p_sens = plot(target_changes, actual_var_changes, label="Actual Change", lw=3, marker=:circle,
              xlabel="Target Change in Variance (Δ<var>)", ylabel="Actual Change in Variance",
              title="Sensitivity Analysis for Diffusion Parameter 's'", legend=:topleft)
plot!(p_sens, x->x, 0, maximum(target_changes), label="Ideal (Prediction)", lw=2, linestyle=:dash, color=:red)
display(p_sens)
# savefig(p_sens, "sensitivity_analysis_diffusion_s.png")
println("Sensitivity analysis plot saved as 'sensitivity_analysis_diffusion_s.png'")

##
###################################################################################
### SECTION: Newton's Method for Diffusion Parameter 's'
###################################################################################
println("\n\n--- STARTING NEWTON'S METHOD FOR 's' ---")
# FIX: The target variance should be relative to the analytical baseline.
target_variance = initial_variance_analytical + 0.05
sim_args = (dt_sim=dt_sim, Nsteps=Nsteps, res=res, dt_obs=dt_obs)

# --- Run Quasi-Newton ---
# FIX: Start the optimization from the analytical variance.
_, var_history_quasi = run_newton_optimization_s(s_initial, params_original, initial_variance_analytical, target_variance; 
                                                 recalculate_R=false, sim_args=sim_args)

# --- Run Full Newton ---
# FIX: Start the optimization from the analytical variance.
_, var_history_full = run_newton_optimization_s(s_initial, params_original, initial_variance_analytical, target_variance; 
                                                recalculate_R=true, sim_args=sim_args)

# --- Plot Convergence ---
p_conv = plot(0:length(var_history_quasi)-1, var_history_quasi, label="Quasi-Newton", lw=2.5, marker=:circle,
              xlabel="Iteration", ylabel="Variance", title="Convergence to Target Variance", legend=:right)
plot!(p_conv, 0:length(var_history_full)-1, var_history_full, label="Full Newton", lw=2.5, marker=:diamond, linestyle=:dash)
hline!(p_conv, [target_variance], label="Target Variance", color=:red, linestyle=:dot, lw=3)
display(p_conv)
savefig(p_conv, "newton_convergence_diffusion_s.png")
println("Newton's method convergence plot saved as 'newton_convergence_diffusion_s.png'")
