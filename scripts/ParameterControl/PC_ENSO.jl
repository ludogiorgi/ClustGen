# -----------------------------------------------------
# --- Environment Setup
# -----------------------------------------------------
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using ClustGen
using MarkovChainHammer
using Statistics
using LinearAlgebra
using Plots
using StatsBase
using Base.Threads

##
# -----------------------------------------------------
# --- 1. Core Function Definitions
# -----------------------------------------------------

"""
Runs a simulation with parameters `p` to find the system's moments.
This is the slow, simulation-based replacement for the original analytical function.
"""
function compute_moments_from_simulation(p, observables; nsteps_sim=1_000_000, dt=0.01, res=10)
    F_local(x, t) = [-p[:dᵤ]*x[1] - p[:wᵤ]*x[2] + x[3], -p[:dᵤ]*x[2] + p[:wᵤ]*x[1], -p[:dₜ]*x[3]]
    sigma_local(x, t) = [p[:σ₁], p[:σ₂], 1.5*(tanh(x[1]) + 1.0)]
    
    obs = evolve([0.0, 0.0, 0.0], dt, nsteps_sim, F_local, sigma_local; timestepper=:euler, resolution=res, seed=rand(1:typemax(Int)))
    
    return [mean(observable(x) for x in eachcol(obs)) for observable in observables]
end

"""
Computes the numerical Jacobian J = d(moments)/d(params) via finite differences.
Calls `compute_moments_from_simulation` repeatedly.
"""
function compute_numerical_jacobian(p_base, params_to_perturb, observables; ε=1e-3, sim_args...)
    N_moments = length(observables)
    N_params = length(params_to_perturb)
    J = zeros(N_moments, N_params)
    
    println("Starting numerical Jacobian calculation (will be slow)...")
    Threads.@threads for i in 1:N_params
        param_key = params_to_perturb[i]
        println("  Thread $(Threads.threadid()): Perturbing :$param_key")
        
        p_plus = copy(p_base); p_plus[param_key] += ε
        p_minus = copy(p_base); p_minus[param_key] -= ε
        
        moments_plus = compute_moments_from_simulation(p_plus, observables; sim_args...)
        moments_minus = compute_moments_from_simulation(p_minus, observables; sim_args...)
        
        J[:, i] = (moments_plus - moments_minus) / (2 * ε)
    end
    println("Numerical Jacobian calculation complete.")
    return J
end

"""
Calculates the score function s(x) = -Cov(x)⁻¹ * (x - μ) using a Gaussian approximation.
"""
function linear_score(x_t)
    μ = mean(x_t, dims=2)
    inv_Σ = inv(cov(x_t'))
    score_func(x) = -inv_Σ * (x .- μ)
    return score_func
end

"""
[CORRECTED] Creates conjugate observable functions based on the formula B = -[∇⋅f + f⋅s].
"""
function create_conjugate_observables(score_function)
    B_dᵤ(x) = -(-2.0 + (-x[1] * score_function(x)[1] - x[2] * score_function(x)[2]))
    B_wᵤ(x) = -(0.0 + (-x[2] * score_function(x)[1] + x[1] * score_function(x)[2]))
    B_dₜ(x) = -(-1.0 + (-x[3] * score_function(x)[3]))
    return [B_dᵤ, B_wᵤ, B_dₜ]
end

"""
Calculates the response matrix R by integrating the cross-correlation function.
"""
function create_response_matrix(time_series, dt, observables, conjugate_observables; max_lag_time=50.0)
    N_moments = length(observables)
    N_params = length(conjugate_observables)
    lag_indices = 0:floor(Int, max_lag_time / dt)
    R = zeros(N_moments, N_params)

    println("Evaluating observables on the time series...")
    all_ts_A = [[obs(x) for x in eachcol(time_series)] for obs in observables]
    all_ts_B = [[conj_obs(x) for x in eachcol(time_series)] for conj_obs in conjugate_observables]

    println("Computing cross-correlations and integrating...")
    Threads.@threads for k in 1:N_moments
        for i in 1:N_params
            corr_series = crosscov(all_ts_A[k], all_ts_B[i], lag_indices; demean=false)
            R[k, i] = dt * (sum(corr_series) - 0.5 * (corr_series[1] + corr_series[end]))
        end
    end
    return R
end

"""
[MODIFIED] Performs a sensitivity analysis, now also returning the history of parameter changes.
"""
function run_sensitivity_analysis(response_matrix, params_original, params_to_control, initial_moments, observables; 
                                  n_control=10, base_Δμ=fill(0.01, 9), λ=0.001, sim_args...)
    println("\n--- STARTING SENSITIVITY ANALYSIS (EXPECT LONG RUN TIMES) ---")
    
    N_moments = size(response_matrix, 1)
    N_params = length(params_to_control)
    predicted_changes = zeros(N_moments, n_control)
    actual_changes = zeros(N_moments, n_control)
    params_change_history = zeros(N_params, n_control) # To store parameter changes
    
    params_initial_to_control_vals = [params_original[k] for k in params_to_control]
    A = λ * Diagonal(1 ./ abs.(params_initial_to_control_vals))
    Γ = I(N_moments)
    matrix_to_invert = response_matrix' * Γ * response_matrix + A

    for k in 1:n_control
        println("  Sensitivity step $k of $n_control...")
        Δμ_k = k * base_Δμ
        δc_opt = matrix_to_invert \ (response_matrix' * Γ * Δμ_k)
        params_change_history[:, k] = δc_opt # Store the parameter change
        
        params_new = copy(params_original)
        for (i, key) in enumerate(params_to_control)
            params_new[key] += δc_opt[i]
        end
        
        moments_new = compute_moments_from_simulation(params_new, observables; sim_args...)
        predicted_changes[:, k] = Δμ_k
        actual_changes[:, k] = moments_new - initial_moments
    end
    
    println("--- SENSITIVITY ANALYSIS COMPLETE ---")
    return predicted_changes, actual_changes, params_change_history
end

"""
Performs Newton-like optimization to drive a model's moments to a target.
"""
function run_newton_optimization(initial_jacobian, params_original, params_to_control, initial_moments, target_moments, observables; 
                                 max_iters=5, tolerance=1e-3, λ=0.1, recalculate_jacobian=false, sim_args...)
    
    method_name = recalculate_jacobian ? "Full Newton" : "Quasi-Newton"
    println("\n--- Starting Optimization: $method_name (EXPECT EXTREME SLOWNESS) ---")
    
    curr_params = copy(params_original)
    curr_moments = copy(initial_moments)
    params_history = [copy(curr_params)]
    moments_history = [copy(curr_moments)]
    
    params_initial_to_control_vals = [params_original[k] for k in params_to_control]
    A = λ * Diagonal(1 ./ abs.(params_initial_to_control_vals))
    Γ = I(length(initial_moments))
    
    jacobian_to_use = copy(initial_jacobian)

    for iter in 1:max_iters
        println("--- Iteration $iter of $max_iters ---")
        
        if recalculate_jacobian
            println("  Recalculating Jacobian...")
            jacobian_to_use = compute_numerical_jacobian(curr_params, params_to_control, observables; sim_args...)
        end
        
        remaining_Δμ = target_moments - curr_moments
        if norm(remaining_Δμ) < tolerance
            println("Target reached within tolerance.")
            break
        end
        
        matrix_to_invert = jacobian_to_use' * Γ * jacobian_to_use + A
        δc = matrix_to_invert \ (jacobian_to_use' * Γ * remaining_Δμ)
        
        α = 1.0
        curr_error = norm(target_moments - curr_moments)
        while α > 1e-6
            println("    Line search with α = $α...")
            trial_params = copy(curr_params)
            for (i, key) in enumerate(params_to_control)
                trial_params[key] += α * δc[i]
            end
            
            trial_moments = compute_moments_from_simulation(trial_params, observables; sim_args...)
            trial_error = norm(target_moments - trial_moments)
            
            if trial_error < curr_error
                curr_params = trial_params
                curr_moments = trial_moments
                push!(params_history, copy(curr_params))
                push!(moments_history, copy(curr_moments))
                println("    Step accepted. New moments: $(round.(curr_moments, digits=4))")
                break
            end
            α *= 0.5
        end

        if α <= 1e-6
            println("Line search failed. Stopping optimization.")
            break
        end
    end
    
    if length(moments_history) == max_iters+1 println("Warning: Maximum iterations reached.") end
    return params_history, moments_history
end

##
# -----------------------------------------------------
# --- 2. Main Script Execution
# -----------------------------------------------------

# --- System & Simulation Parameters ---
params = Dict(:dᵤ => 0.2, :wᵤ => 0.4, :dₜ => 2.0, :σ₁ => 0.3, :σ₂ => 0.3)
params_to_control = [:dᵤ, :wᵤ, :dₜ]

# --- Step 1: Generate a long baseline simulation ---
println("Generating baseline time series for Response Matrix R...")
F_sim(x,t) = [-params[:dᵤ]*x[1] - params[:wᵤ]*x[2] + x[3], -params[:dᵤ]*x[2] + params[:wᵤ]*x[1], -params[:dₜ]*x[3]]
sigma_sim(x,t) = [params[:σ₁], params[:σ₂], 1.5*(tanh(x[1]) + 1.0)]
obs_data = evolve([0.0, 0.0, 0.0], 0.01, 100_000_000, F_sim, sigma_sim; resolution=10, seed=rand(1:typemax(Int)))

# --- Step 2: Define Observables and get Initial Moments ---
μ_obs = mean(obs_data, dims=2); μ₁, μ₂, μ₃ = μ_obs[1],μ_obs[2],μ_obs[3]
# observables = [x->x[1], x->x[2], x->x[3], x->(x[1]-μ₁)^2, x->(x[2]-μ₂)^2, x->(x[3]-μ₃)^2, x->(x[1]-μ₁)*(x[2]-μ₂), x->(x[1]-μ₁)*(x[3]-μ₃), x->(x[2]-μ₂)*(x[3]-μ₃)]
observables = [x->(x[1]-μ₁)^2, x->(x[2]-μ₂)^2, x->(x[3]-μ₃)^2]
initial_moments = [mean(obs(x) for x in eachcol(obs_data)) for obs in observables]

# --- Step 3: Compute R_linear and J_numerical ---
println("\n--- Computing R_linear ---")
R_linear = create_response_matrix(obs_data, 0.1, observables, create_conjugate_observables(linear_score(obs_data)))

sim_args_fast = (nsteps_sim=100_000_000, dt=0.01, res=10) 
J_numerical = compute_numerical_jacobian(params, params_to_control, observables; ε=1e-3, sim_args_fast...)

# --- Step 4: Display Matrices ---
println("\n" * "="^40 * "\n     FINAL MATRIX COMPARISON\n" * "="^40)
println("➡️  Response Matrix R (from Linear Score Approximation):"); display(round.(R_linear, digits=4))
println("\n➡️  Numerical Jacobian J (from Finite Differences):"); display(round.(J_numerical, digits=4))
println("-"^40 * "\n")

##

sim_args_fast = (nsteps_sim=10_000_000, dt=0.01, res=10) 
# --- Sensitivity Analysis Examples ---
println("RUNNING SENSITIVITY ANALYSIS (THIS WILL TAKE A VERY LONG TIME)")

# Run 1: Using the Numerical Jacobian (ground truth)
println("\n--- Analyzing with J_numerical ---")
predicted_J, actual_J, params_hist_J = run_sensitivity_analysis(J_numerical, params, params_to_control, initial_moments, observables;
                                             n_control=10,
                                             base_Δμ=fill(0.02, 3),
                                             λ=0.001,
                                             sim_args_fast...)                                              

# Run 2: Using the Linear Score Response Matrix
println("\n--- Analyzing with R_linear ---")
predicted_R, actual_R, params_hist_R = run_sensitivity_analysis(R_linear, params, params_to_control, initial_moments, observables;
                                             n_control=10,
                                             base_Δμ=fill(0.02, 3),
                                             λ=0.001,
                                             sim_args_fast...)

##
# Test individual sensitivity calculation
response_matrix = J_numerical
params_original = params
params_to_control = [:dᵤ, :wᵤ, :dₜ]
λ = 0.001
n_control = 2
base_Δμ = fill(0.03, 3)

N_moments = size(response_matrix, 1)
N_params = length(params_to_control)
    
params_initial_to_control_vals = [params_original[k] for k in params_to_control]
A = λ * Diagonal(1 ./ abs.(params_initial_to_control_vals))
Γ = I(N_moments)
matrix_to_invert = response_matrix' * Γ * response_matrix + A

k = 2
Δμ_k = k * base_Δμ
δc_opt = matrix_to_invert \ (response_matrix' * Γ * Δμ_k)
        
params_new = copy(params_original)
for (i, key) in enumerate(params_to_control)
    params_new[key] += δc_opt[i]
end
params_new
        
moments_new = compute_moments_from_simulation(params_new, observables; sim_args_fast...)
actual_change = moments_new - initial_moments
display(actual_change)
println(mean(actual_change))

##
# -----------------------------------------------------
# --- 3. Plotting Section
# -----------------------------------------------------
println("\n--- Generating Plots ---")

# Define labels for observables and parameters
# moment_labels = ["<x₁>", "<x₂>", "<x₃>", "Var(x₁)", "Var(x₂)", "Var(x₃)", "Cov(x₁,x₂)", "Cov(x₁,x₃)", "Cov(x₂,x₃)"]
moment_labels = ["<x₁>", "<x₂>", "<x₃>"]
param_labels = ["δdᵤ", "δwᵤ", "δdₜ"]

# --- Plot 1: Predicted vs. Actual Moment Changes ---
if @isdefined(predicted_J) && @isdefined(actual_J)
    println("Plotting sensitivity analysis: Predicted vs Actual...")
    plot_array_sens = []
    for i in 1:3
    
        p = scatter(predicted_J[i, :], actual_J[i, :], label="Jacobian", color=:purple, markersize=5, markerstrokewidth=0.5,
                    xlabel="Predicted Change", ylabel="Actual Change", title=moment_labels[i],
                    legend=:bottomright)
        scatter!(p, predicted_R[i, :], actual_R[i, :], label="Linear Score (R)", color=:green, markersize=5, markerstrokewidth=0.5)
        plot!(p, predicted_J[i, :], predicted_J[i, :], label="Perfect Prediction", linestyle=:dash, color=:red, lw=2)
        push!(plot_array_sens, p)
    end
    sensitivity_plot = plot(plot_array_sens..., layout=(3, 3), size=(1400, 1200),
                            plot_title="Sensitivity Analysis: Predicted vs. Actual Moment Changes", plot_titlefontsize=16, margin=5Plots.mm)
    display(sensitivity_plot)
    savefig(sensitivity_plot, "sensitivity_analysis_comparison_3D.png")
end

##
# --- Plot 2: Parameter Changes vs. Target Moment Changes ---
if @isdefined(params_hist_J) && @isdefined(params_hist_R)
    println("Plotting sensitivity analysis: Parameter vs Moment Changes...")
    plot_array_params = []
    
    # We will plot the change in each parameter vs. the target change in the three means
    moments_to_plot_indices = 1:3 
    
    for param_idx in 1:length(params_to_control)
        for moment_idx in moments_to_plot_indices
            # X-axis is the target change for the specific moment - use actual n_control=8 and base_Δμ=0.005
            target_moment_changes = (1:8) .* 0.005 # Use actual values from sensitivity analysis
            
            p = plot(target_moment_changes, params_hist_J[param_idx, :],
                     label="Jacobian",
                     xlabel="Target Δ($(moment_labels[moment_idx]))",
                     ylabel=param_labels[param_idx],
                     lw=2, color=:purple)
            
            plot!(p, target_moment_changes, params_hist_R[param_idx, :],
                  label="Linear Score (R)",
                  lw=2, color=:green)
            
            # Add legend only to the first plot of each row for clarity
            if moment_idx != 1
                plot!(p, legend=false)
            end

            push!(plot_array_params, p)
        end
    end
    
    param_change_plot = plot(plot_array_params...,
                             layout=(length(params_to_control), length(moments_to_plot_indices)),
                             size=(1500, 1200),
                             plot_title="Parameter Changes vs. Target Moment Changes",
                             plot_titlefontsize=16,
                             margin=5Plots.mm)
    display(param_change_plot)
    savefig(param_change_plot, "parameter_changes_vs_moments_3D.png")
end

##
# --- Plot 3: Newton's Method Convergence ---
println("Plotting Newton's method convergence...")

# Define a target: e.g., increase mean of x1 by 0.1 and variance of x2 by 0.1
target_moments = copy(initial_moments)
target_moments .+= 0.05 # Target for <x1>
sim_args_fast = (nsteps_sim=10_000_000, dt=0.01, res=10) 

# Using Quasi-Newton (Jacobian computed once)
params_hist, moments_hist = run_newton_optimization(J_numerical, params, params_to_control, initial_moments, target_moments, observables;
                                                    max_iters=8,
                                                    recalculate_jacobian=false,
                                                    sim_args_fast...)
                                                    
plot_array_newton = []
moments_matrix = hcat(moments_hist...)
iterations = 0:(size(moments_matrix, 2) - 1)

for i in 1:length(observables)
    p = plot(iterations, moments_matrix[i, :], label="Moment Value", lw=2.5, marker=:circle, markersize=4,
             xlabel="Iteration", ylabel="Value", title="Variance $(i)", legend=:best)
    hline!(p, [initial_moments[i]], label="Initial", linestyle=:dash, color=:gray, lw=2)
    hline!(p, [target_moments[i]], label="Target", linestyle=:dot, color=:red, lw=3)
    push!(plot_array_newton, p)
end

newton_plot = plot(plot_array_newton..., layout=(1, 3), size=(1400, 400),
                   plot_title="Newton's Method Convergence", plot_titlefontsize=16, margin=5Plots.mm)
display(newton_plot)
savefig(newton_plot, "newton_convergence_3D.png")

println("\n--- All plotting complete ---")

# End of script