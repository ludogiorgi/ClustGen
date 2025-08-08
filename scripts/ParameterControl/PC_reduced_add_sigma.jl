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
using QuadGK
##

# Coefficients of the reduced model
a = -0.0222
b = -0.2
c = 0.0494
F_tilde = 0.6
s = 0.7071
D_eff = s^2 / 2  # Effective diffusion coefficient
params_original = [a, b, c, F_tilde, s]

function F(x,t; F_tilde=F_tilde)
    u = x[1]
    return [F_tilde + a * u + b * u^2 - c * u^3]
end

function sigma(x,t)
    return s/√2
end

function create_exact_s_ds(p)
    exact_score(x) = 2 * (p[4] + p[1]*x + p[2]*x^2 - p[3]*x^3) / p[5]^2
    exact_score_derivative(x) = 2 * (p[1] + 2*p[2]*x - 3*p[3]*x^2) / p[5]^2
    return exact_score, exact_score_derivative
end

function create_linear_s_ds(x_t)
    μ = mean(x_t)
    variance = var(x_t)
    score_func(x) = -(x - μ) / variance
    score_derivative(x) = -1 / variance
    return score_func, score_derivative
end

function create_kgmm_s_ds(score, s_val)
    score_derivative = construct_divergence_score(score)
    B_diff(x) = s_val * (score_derivative(x) + score(x)^2)
    return B_diff
end

"""
    potential(x, p)
Calculates the potential U(x) for the Langevin system.
The potential is the negative integral of the deterministic force.
"""
potential(x, p) = -(p[4]*x + p[1]*x^2/2 + p[2]*x^3/3 - p[3]*x^4/4)

"""
    p_unnormalized(x, p)
Calculates the unnormalized steady-state probability distribution p(x) ~ exp(-U(x)/D).
"""
p_unnormalized(x, p) = exp(-potential(x, p)/p[5]^2*2)

"""
    compute_observables(p, observables)
Calculates the expected values of arbitrary observables for a given parameter set `p`
by numerically integrating over the analytical probability distribution.
"""
function compute_observables(p, observables)
    integrand_norm(x) = p_unnormalized(x, p)
    int_bounds = (-3, 4)
    norm_const, _ = quadgk(integrand_norm, int_bounds...)
    
    observable_values = zeros(length(observables))
    
    for (i, obs) in enumerate(observables)
        integrand_obs(x) = obs(x) * integrand_norm(x)
        obs_integral, _ = quadgk(integrand_obs, int_bounds...)
        observable_values[i] = obs_integral / norm_const
    end
    
    return observable_values
end

"""
    compute_jacobian(p, param_indices, observables; ε=1e-5)
Computes the Jacobian matrix J = d(observables)/d(params) for the specified `param_indices`
using central finite differences on the analytical `compute_observables` function.
"""
function compute_jacobian(p, param_indices, observables; ε=1e-5)
    N_observables = length(observables)
    J = zeros(N_observables, length(param_indices))
    
    for (i, p_idx) in enumerate(param_indices)
        p_plus = copy(p)
        p_minus = copy(p)
        p_plus[p_idx] += ε
        p_minus[p_idx] -= ε
        
        obs_plus = compute_observables(p_plus, observables)
        obs_minus = compute_observables(p_minus, observables)       
        
        J[:, i] = (obs_plus - obs_minus) / (2 * ε)
    end
    return J
end

"""
    create_conjugate_observables(score_function)
Creates and returns a vector of all conjugate observable functions (B_a, B_b, B_c, B_F)
using a provided score function.
"""
function create_conjugate_observables(score, score_derivative, s_val)
    # Helper function to extract scalar from score function result
    score_scalar(x) = begin
        s = score(x)
        return isa(s, AbstractArray) ? s[1] : s
    end

    score_derivative_scalar(x) = begin
        s = score_derivative(x)
        return isa(s, AbstractArray) ? s[1] : s
    end

    B_diff(x) = s_val * (score_derivative_scalar(x) + score_scalar(x)^2)
    return [B_diff]
end

"""
    create_response_matrix(time_series, dt, observables, conjugate_observables; max_lag_time=30.0)
Calculates the response matrix R by integrating the cross-correlation function 
between observables and their conjugate counterparts, as required by the GFDT.
"""
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
            corr_series = crosscov(ts_A, ts_B, lag_indices; demean=true)
            Rt[k, i, :] = corr_series
            integral_val = dt * (sum(corr_series) - 0.5 * (corr_series[1] + corr_series[end]))
            R[k, i] = integral_val
        end
    end
    return R, Rt, lags
end


"""
    run_sensitivity_analysis(response_matrix, params_original, param_indices_to_control, initial_observables, observables; D_eff, n_control=100, base_Δμ=[0.01, 0.01], λ=0.1)
Performs a sensitivity analysis to compare linear predictions with actual changes.
"""
function run_sensitivity_analysis(response_matrix, params_original, param_indices_to_control, initial_observables, observables; n_control=100, base_Δμ=nothing, λ=0.1)
    println("\n--- STARTING SENSITIVITY ANALYSIS for response matrix ---")
    
    N_observables = size(response_matrix, 1)
    N_params = length(param_indices_to_control)
    
    # Set default base_Δμ if not provided
    if base_Δμ === nothing
        base_Δμ = fill(0.01, N_observables)
    end
    
    predicted_changes = zeros(N_observables, n_control)
    actual_changes = zeros(N_observables, n_control)
    params_change_history = zeros(N_params, n_control) # Store parameter changes
    control_steps = 0:n_control-1
    
    params_initial_to_control = params_original[param_indices_to_control]
    Γ = I(N_observables)
    A = λ * Diagonal(1 ./ abs.(params_initial_to_control))

    for k in control_steps
        Δμ_k = k * base_Δμ
        matrix_to_invert = response_matrix' * Γ * response_matrix + A
        δc_opt = matrix_to_invert \ (response_matrix' * Γ * Δμ_k)
        
        predicted_change_k = response_matrix * δc_opt
        predicted_changes[:, k+1] = predicted_change_k
        params_change_history[:, k+1] = δc_opt # Store the parameter change
        
        params_new_full = copy(params_original)
        params_new_full[param_indices_to_control] .+= δc_opt
        
        observables_new = compute_observables(params_new_full, observables)
        actual_change_k = observables_new - initial_observables
        actual_changes[:, k+1] = actual_change_k
    end
    
    println("--- SENSITIVITY ANALYSIS COMPLETE ---")
    return predicted_changes, actual_changes, params_change_history
end

"""
    run_newton_optimization(initial_jacobian, params_original, param_indices_to_control, initial_observables, target_observables, observables; 
                            D_eff, max_iters=15, tolerance=1e-4, λ=0.1, 
                            recalculate_jacobian=false, jacobian_method=:response_matrix, 
                            score_type=:exact, use_line_search=true, sim_args=())
Performs a Newton-like optimization to drive a model's observables to a target.
Allows choosing between quasi-Newton (fixed Jacobian) and full Newton (recalculated Jacobian) methods.
"""
function run_newton_optimization(initial_jacobian, params_original, param_indices_to_control, initial_observables, target_observables, observables; 
                                 max_iters=15, tolerance=1e-4, λ=0.1, 
                                 recalculate_jacobian=false, jacobian_method=:response_matrix, 
                                 score_type=:exact, use_line_search=false, sim_args=())
    
    method_name = if recalculate_jacobian
        "Full Newton ($jacobian_method, $score_type)"
    else
        "Quasi-Newton"
    end
    line_search_info = use_line_search ? "with line search" : "without line search"
    println("\n--- Starting Optimization: $method_name ($line_search_info) ---")
    
    # Initialization
    curr_params = copy(params_original)
    curr_observables = copy(initial_observables)
    params_history = [copy(curr_params)]
    observables_history = [copy(curr_observables)]
    
    # Regularization Matrix
    params_initial_to_control = params_original[param_indices_to_control]
    A = λ * Diagonal(1 ./ abs.(params_initial_to_control))
    Γ = I(length(initial_observables))
    
    jacobian_to_use = copy(initial_jacobian)

    iter = 0
    while iter < max_iters
        iter += 1
        println("--- Iteration $iter ---")
        
        # --- Optional: Recalculate Jacobian at each step (Full Newton's Method) ---
        if recalculate_jacobian && iter > 1
            println("Recalculating Jacobian/Response Matrix...")
            if jacobian_method == :analytical_jacobian
                # Recalculate J analytically (cheap)
                jacobian_to_use = compute_jacobian(curr_params, param_indices_to_control, observables)
                println("Updated Analytical Jacobian:"); display(jacobian_to_use)

            elseif jacobian_method == :response_matrix
                # Recalculate R by re-simulating (expensive)
                F_current(x, t; p=curr_params) = [p[4] + p[1]*x[1] + p[2]*x[1]^2 - p[3]*x[1]^3]
                sigma_current(x, t) = sim_args.s / √2
                new_obs = evolve([0.0], sim_args.dt_sim, sim_args.Nsteps, F_current, sigma_current; timestepper=:euler, resolution=sim_args.res)
                
                # Choose the score function for recalculation
                local new_score_func
                if score_type == :exact
                    new_score_func, new_score_func_derivative = create_exact_s_ds(curr_params)
                elseif score_type == :linear
                    new_score_func, new_score_func_derivative = create_linear_s_ds(curr_params)
                elseif score_type == :clustered
                    new_kgmm_results = calculate_score_kgmm(new_obs;
                        # Clustering & Score parameters
                        σ_value=0.05,
                        clustering_prob=0.0005,
                        clustering_conv_param=0.002,
                        clustering_max_iter=150,
                        use_normalization_for_clustering=true,
                        # NN Training parameters
                        epochs=1000,
                        batch_size=32,
                        hidden_layers=[100, 50],
                        activation=swish,
                        last_activation=identity,
                        optimizer=Flux.Adam(0.001),
                        use_gpu=false,
                        # Control parameters
                        verbose=false
                    )
                    new_score_func = new_kgmm_results.score_function
                    new_score_func_derivative = construct_divergence_score(new_score_func; 
                                                                            sigma=0.01, 
                                                                            rtol=1e-6, 
                                                                            n_points=100, 
                                                                            range=(-4.0, 5.0)  # Extended range to cover the data domain
                    )
                else
                    error("Unknown score_type: $score_type")
                end
                
                new_conjugates_to_use = create_conjugate_observables(new_score_func, 
                                                                  new_score_func_derivative, 
                                                                  sim_args.s)
                jacobian_to_use, _, _ = create_response_matrix(new_obs[1, :], sim_args.dt_obs, observables, new_conjugates_to_use)
                println("Updated Response Matrix ($score_type score):"); display(jacobian_to_use)
            end
            
            # Check for NaN or Inf in the jacobian
            if any(isnan, jacobian_to_use) || any(isinf, jacobian_to_use)
                println("ERROR: Jacobian/Response Matrix contains NaN or Inf values. Stopping optimization.")
                println("Current parameters: $(curr_params)")
                println("This may indicate numerical instability or that the parameters have moved to an invalid region.")
                break
            end
        end
        
        remaining_Δμ = target_observables - curr_observables
        if norm(remaining_Δμ) < tolerance
            println("Target reached within tolerance.")
            break
        end
        
        matrix_to_invert = jacobian_to_use' * Γ * jacobian_to_use + A
        
        # Check for NaN or Inf in the matrix to invert
        if any(isnan, matrix_to_invert) || any(isinf, matrix_to_invert)
            println("ERROR: Matrix to invert contains NaN or Inf values. Stopping optimization.")
            println("This may indicate numerical instability.")
            break
        end
        
        δc = matrix_to_invert \ (jacobian_to_use' * Γ * remaining_Δμ)
        
        # Check for NaN or Inf in the step
        if any(isnan, δc) || any(isinf, δc)
            println("ERROR: Parameter step δc contains NaN or Inf values. Stopping optimization.")
            println("This may indicate that the matrix inversion failed due to singularity.")
            break
        end
        
        if use_line_search
            # Perform line search
            α = 1.0
            trial_params = copy(curr_params)
            trial_params[param_indices_to_control] .+= α * δc
            trial_observables = compute_observables(trial_params, observables)
            trial_error = norm(target_observables - trial_observables)
            curr_error = norm(target_observables - curr_observables)
            
            while trial_error >= curr_error && α > 1e-4
                α *= 0.5
                trial_params = copy(curr_params)
                trial_params[param_indices_to_control] .+= α * δc
                trial_observables = compute_observables(trial_params, observables)
                trial_error = norm(target_observables - trial_observables)
            end
            
            if trial_error < curr_error
                curr_params = trial_params
                curr_observables = trial_observables
                push!(params_history, copy(curr_params))
                push!(observables_history, copy(curr_observables))
                println("Step accepted with α=$α. New observables: $(round.(curr_observables, digits=5))")
            else
                println("Line search failed. Stopping optimization.")
                break
            end
        else
            # Use full step without line search
            curr_params[param_indices_to_control] .+= δc
            curr_observables = compute_observables(curr_params, observables)
            push!(params_history, copy(curr_params))
            push!(observables_history, copy(curr_observables))
            println("Full step taken. New observables: $(round.(curr_observables, digits=5))")
        end
    end
    
    if iter == max_iters println("Warning: Maximum iterations reached.") end
    return params_history, observables_history
end

"""
    construct_divergence_score(s; sigma=0.01, rtol=1e-6, n_points=200, range=(-5.0, 5.0))

Constructs a fast function that approximates the divergence of a vector field `s` 
using pre-computed values over a specified range and linear interpolation.

This approach pre-computes divergence values at grid points once, then uses 
fast linear interpolation for all subsequent evaluations.

# Arguments
- `s::Function`: The vector field. A function that takes a scalar or vector
  and returns a scalar or vector of the same dimension.
- `sigma::Real`: The standard deviation of the Gaussian kernel.
- `rtol::Real`: The relative tolerance (reduced for speed).
- `n_points::Int`: Number of quadrature points for integration.
- `range::Tuple{Real,Real}`: Range (min, max) for pre-computing divergence values.

# Returns
- `Function`: A new function `divergence_func(x)` that takes a point `x`
  and returns the estimated divergence using fast interpolation.
"""
function construct_divergence_score(s::Function; sigma::Real=0.01, rtol::Real=1e-6, n_points::Int=200, range::Tuple{Real,Real}=(-5.0, 5.0))
    
    println("🔄 Pre-computing divergence values for fast interpolation...")
    
    # Pre-compute quadrature points and weights for speed
    limit = 5.0 * sigma
    nodes, weights = begin
        # Simple trapezoidal rule with pre-computed points
        z_vals = Base.range(-limit, limit, length=n_points)
        dz = step(z_vals)
        w_vals = fill(dz, n_points)
        w_vals[1] = w_vals[end] = dz/2  # Trapezoidal correction
        (collect(z_vals), w_vals)
    end
    
    # Pre-compute Gaussian normalization
    norm_const = 1.0 / sqrt(2 * pi * sigma^2)
    sigma2_inv = 1.0 / (sigma^2)
    
    # Create grid points for pre-computation
    x_min, x_max = range
    grid_size = max(500, n_points)  # Use at least 500 points for good interpolation
    x_grid = Base.range(x_min, x_max, length=grid_size)
    divergence_values = Vector{Float64}(undef, grid_size)
    
    println("   📊 Computing divergence at $(grid_size) grid points in range [$(x_min), $(x_max)]...")
    
    # Pre-compute divergence values at all grid points (parallelized)
    @threads for k in 1:grid_size
        x_val = x_grid[k]
        
        # Vectorized computation for speed
        integrand_vals = Vector{Float64}(undef, n_points)
        
        # Parallel evaluation of the integrand at all quadrature points
        @threads for i in 1:n_points
            z = nodes[i]
            g_val = norm_const * exp(-z^2 * 0.5 * sigma2_inv)
            
            s_val = s(x_val - z)
            # Handle both scalar and vector outputs from s
            s_scalar = s_val isa AbstractArray ? s_val[1] : s_val
            
            integrand_vals[i] = -s_scalar * (z * sigma2_inv) * g_val * weights[i]
        end
        
        # Sum up the integral
        divergence_values[k] = sum(integrand_vals)
    end
    
    println("   ✅ Pre-computation complete! Divergence function ready for fast evaluation.")
    
    # Convert to regular arrays for better performance
    x_grid_array = collect(x_grid)
    dx = step(x_grid)
    
    # Return fast interpolating function
    function fast_divergence_func(x::Union{Real, Vector{<:Real}})
        # Convert scalar input to vector for uniform handling
        x_vec = x isa Real ? [x] : x
        D = length(x_vec)

        if D == 1
            x_val = x_vec[1]
            
            # Clamp to range bounds
            if x_val < x_min
                return divergence_values[1]
            elseif x_val > x_max
                return divergence_values[end]
            end
            
            # Fast linear interpolation
            # Find the index for interpolation
            idx_float = (x_val - x_min) / dx + 1
            idx_low = floor(Int, idx_float)
            idx_high = idx_low + 1
            
            # Ensure indices are within bounds
            idx_low = max(1, min(idx_low, grid_size-1))
            idx_high = max(2, min(idx_high, grid_size))
            
            # Linear interpolation
            t = idx_float - idx_low
            return divergence_values[idx_low] * (1 - t) + divergence_values[idx_high] * t
            
        elseif D == 2
            # For 2D, fall back to direct computation (not pre-computed)
            # This is a simplified fallback - could be extended to 2D grids
            x1, x2 = x_vec[1], x_vec[2]
            norm_const_2d = 1.0 / (2 * pi * sigma^2)
            
            integral_sum = 0.0
            
            # Use nested loops for 2D integration (can be parallelized)
            @threads for i in 1:n_points
                z1 = nodes[i]
                w1 = weights[i]
                
                inner_sum = 0.0
                for j in 1:n_points
                    z2 = nodes[j]
                    w2 = weights[j]
                    
                    z_norm_sq = z1^2 + z2^2
                    g_val = norm_const_2d * exp(-z_norm_sq * 0.5 * sigma2_inv)
                    
                    s_val = s([x1 - z1, x2 - z2])
                    s_vec = s_val isa AbstractArray ? s_val : [s_val, 0.0]
                    
                    dot_product = s_vec[1] * z1 + (length(s_vec) > 1 ? s_vec[2] * z2 : 0.0)
                    
                    inner_sum += -dot_product * sigma2_inv * g_val * w2
                end
                
                integral_sum += inner_sum * w1
            end
            
            return integral_sum
            
        else
            error("Fast divergence supports only 1D and 2D cases. For higher dimensions, consider using HCubature.")
        end
    end

    return fast_divergence_func
end

##

dim = 1
dt = 0.01
Nsteps = 20_000_000
res = 10
dt_sim = dt * res
obs_nn = evolve([0.0], dt, Nsteps, F, sigma; timestepper=:euler, resolution=res)
M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S

autocov_obs = zeros(dim, 300)
for i in 1:dim
    autocov_obs[i,:] = autocovariance(obs[i,1:1000000]; timesteps=300)
end

kde_obs = kde(obs[200:end])

autocov_obs_mean = mean(autocov_obs, dims=1)

# Use GLMakie for plotting
GLMakie.activate!()

fig = Figure(size=(800, 800))
ax1 = Axis(fig[1, 1], xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")
ax2 = Axis(fig[2, 1], xlabel="X", ylabel="Density", title="Observed PDF")

lines!(ax1, autocov_obs_mean[1,:], label="X")
lines!(ax2, kde_obs.x, kde_obs.density, label="Observed")

display(fig)

##

# Use the KGMM score estimation function from the module
kgmm_results = calculate_score_kgmm(obs_nn;
    # Clustering & Score parameters
    σ_value=0.05,
    clustering_prob=0.0005,
    clustering_conv_param=0.002,
    clustering_max_iter=150,
    use_normalization_for_clustering=true,
    # NN Training parameters
    epochs=1000,
    batch_size=32,
    hidden_layers=[100, 50],
    activation=swish,
    last_activation=identity,
    optimizer=Flux.Adam(0.001),
    use_gpu=false,
    # Control parameters
    verbose=false
);
##
score_clustered = kgmm_results.score_function
score_clustered_derivative = construct_divergence_score(score_clustered; 
    sigma=0.01, 
    rtol=1e-6, 
    n_points=100, 
    range=(-4.0, 5.0)  # Extended range to cover the data domain
)

exact_score, exact_score_derivative = create_exact_s_ds(params_original)

xax = [-3.0:0.05:4.0...]
# Plot comparison of all three methods
fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1], xlabel="X", ylabel="Score Derivative", title="Score Derivative Comparison")
lines!(ax, xax, exact_score_derivative.(xax), label="Exact Derivative", linewidth=3)
lines!(ax, xax, score_clustered_derivative.(xax), label="Clustered Derivative", linewidth=2, linestyle=:dash)
axislegend(ax)
display(fig)
##

observable_names = ["Variance"]

# --- Step 2: Define Observables & Parameters for Control ---
M_obs = mean(obs_nn, dims=2)[1]
S_obs = std(obs_nn, dims=2)[1]
observables_to_control = [x -> (x - M_obs)^2]
initial_observables = compute_observables(params_original, observables_to_control)
param_indices_to_control = [5]  # Indices of parameters to control (a, b, c, F_tilde)
# --- Step 3: Compute All Necessary Response Matrices ---

# A) Using the EXACT score function
println("\n--- Computing R_exact ---")
exact_score, exact_score_derivative = create_exact_s_ds(params_original)
conjugate_observables_exact = create_conjugate_observables(exact_score, exact_score_derivative, s)
R_exact, R_exact_t, R_exact_lags = create_response_matrix(obs_nn[1, :], dt_sim, observables_to_control, conjugate_observables_exact; max_lag_time=20.0)

# B) Using the LINEAR score function
println("\n--- Computing R_linear ---")
linear_score_func, linear_score_derivative = create_linear_s_ds(obs_nn[1, :])
conjugate_observables_linear = create_conjugate_observables(linear_score_func, linear_score_derivative, s)
R_linear, R_linear_t, R_linear_lags = create_response_matrix(obs_nn[1, :], dt_sim, observables_to_control, conjugate_observables_linear; max_lag_time=30.0)

# D) Using the CLUSTERED score function
println("\n--- Computing R_clustered ---")
conjugate_observables_clustered = create_conjugate_observables(score_clustered, score_clustered_derivative, s)
R_clustered, R_clustered_t, R_clustered_lags = create_response_matrix(obs_nn[1, :], dt_sim, observables_to_control, conjugate_observables_clustered; max_lag_time=20.0)

# C) Using the ANALYTICAL Jacobian
println("\n--- Computing J (Analytic Jacobian) ---")
J = compute_jacobian(params_original, param_indices_to_control, observables_to_control; ε=1e-2)

println("\n--- COMPARISON OF MATRICES ---")
println("Response Matrix R (from exact score):"); display(R_exact)
println("\nResponse Matrix R (from linear score):"); display(R_linear)
println("\nResponse Matrix R (from clustered score):"); display(R_clustered)
println("\nJacobian J (from analytic PDF):"); display(J)
println("--------------------------------\n")

##
idx1 = 1
idx2 = 1

fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1], xlabel="Lag Time", ylabel="Response", title="Response Matrix Comparison")
lines!(ax, R_exact_t[idx1,idx2,:], label="Exact Score")
lines!(ax, R_linear_t[idx1,idx2,:], label="Linear Score")
lines!(ax, R_clustered_t[idx1,idx2,:], label="KGMM Score")
axislegend(ax)
display(fig)

##
# Remove Plots.jl backend call
# --- Step 4: Run Sensitivity Analysis for Each Matrix ---
common_args = (params_original, param_indices_to_control, initial_observables, observables_to_control)
common_kwargs = (n_control=15, base_Δμ= .- [0.01], λ=0.001)

predicted_exact, actual_exact, params_hist_exact = run_sensitivity_analysis(R_exact, common_args...; common_kwargs...)
predicted_clustered, actual_clustered, params_hist_clustered = run_sensitivity_analysis(R_clustered, common_args...; common_kwargs...)
predicted_linear, actual_linear, params_hist_linear = run_sensitivity_analysis(R_linear, common_args...; common_kwargs...)
predicted_J, actual_J, params_hist_J = run_sensitivity_analysis(J, common_args...; common_kwargs...)

# --- Step 4: Run Newton's Method Optimization for Each Case ---
target_observables = initial_observables - 0.2 .* initial_observables

common_args = (params_original, param_indices_to_control, initial_observables, target_observables, observables_to_control)
newton_kwargs = (max_iters=6, λ=0.001)
sim_args = (dt_sim=dt, Nsteps=20_000_000, res=10, dt_obs=dt_sim, s=s)

# Run optimization for each of the five methods
_, observables_hist_J_quasi = run_newton_optimization(J, common_args...; newton_kwargs...)
_, observables_hist_R_exact_quasi = run_newton_optimization(R_exact, common_args...; newton_kwargs...)
_, observables_hist_R_linear_quasi = run_newton_optimization(R_linear, common_args...; newton_kwargs...)
_, observables_hist_R_clustered_quasi = run_newton_optimization(R_clustered, common_args...; newton_kwargs...)
_, observables_hist_J_full = run_newton_optimization(J, common_args...; newton_kwargs..., recalculate_jacobian=true, jacobian_method=:analytical_jacobian, use_line_search=false)
_, observables_hist_R_exact_full = run_newton_optimization(R_exact, common_args...; newton_kwargs..., recalculate_jacobian=true, jacobian_method=:response_matrix, score_type=:exact, sim_args=sim_args, use_line_search=false)
_, observables_hist_R_linear_full = run_newton_optimization(R_linear, common_args...; newton_kwargs..., recalculate_jacobian=true, jacobian_method=:response_matrix, score_type=:linear, sim_args=sim_args, use_line_search=false)
_, observables_hist_R_clustered_full = run_newton_optimization(R_clustered, common_args...; newton_kwargs..., recalculate_jacobian=true, jacobian_method=:response_matrix, score_type=:clustered, sim_args=sim_args, use_line_search=false)

##

# --- Step 5: Creating Publication-Ready Combined Figure with GLMakie ---
println("Generating publication-ready combined figure with GLMakie...")

# Common parameters
control_steps = 0:(common_kwargs.n_control-1)
base_Δμ = common_kwargs.base_Δμ
param_labels_all = ["δa", "δb", "δc", "δF̃", "δs"]
# Only use labels for parameters being controlled
param_labels = param_labels_all[param_indices_to_control]
N_observables = length(observables_to_control)
N_params_controlled = length(param_indices_to_control)
observable_labels = ["Variance"]

# Define consistent colors - simplified palette
method_colors = Dict(
    "Exact Score" => :blue,
    "Linear Score" => :green,    
    "Jacobian" => :purple,        
    "KGMM Score" => :orange,      
    "Perfect" => :red,
    "Target" => :red
)

# Helper function for observable history
get_observable_history(history, idx) = [h[idx] for h in history]

# Create a much taller figure with better proportions
fig = Figure(size=(2400, 4000), fontsize=40)

# SECTION 1: Parameter Control Analysis (adjust columns based on parameters controlled)
param_row_start = 1
for moment_idx in 1:N_observables
    for param_idx in 1:N_params_controlled
        row_idx = param_row_start + moment_idx - 1
        col_idx = param_idx
        
        ax = Axis(fig[row_idx, col_idx],
                  xlabel="Target Δ$(observable_labels[moment_idx])",
                  ylabel=param_labels[param_idx],
                  xlabelsize=36,
                  ylabelsize=36,
                  xticklabelsize=32,
                  yticklabelsize=32)
        
        x_vals = control_steps .* base_Δμ[moment_idx]
        
        # Plot with thicker lines and clear colors
        lines!(ax, x_vals, params_hist_exact[param_idx, :], 
               color=method_colors["Exact Score"], linewidth=4)
        lines!(ax, x_vals, params_hist_linear[param_idx, :], 
               color=method_colors["Linear Score"], linewidth=4, linestyle=:dash)
        lines!(ax, x_vals, params_hist_J[param_idx, :], 
               color=method_colors["Jacobian"], linewidth=4, linestyle=:dot)
        lines!(ax, x_vals, params_hist_clustered[param_idx, :], 
               color=method_colors["KGMM Score"], linewidth=4, linestyle=:dashdot)
    end
end

# Add section title
Label(fig[0, 1:N_params_controlled], "Parameter Control Analysis", fontsize=44, font=:bold, tellwidth=false)

# SECTION 2: Sensitivity Analysis - FULL WIDTH (2 plots spanning 2 columns each)
sensitivity_row = N_observables + 2

# Each sensitivity plot spans 2 columns for full width
for obs_idx in 1:N_observables
    if N_observables == 1
        # Center single plot across the available columns
        col_start = 1
        col_end = N_params_controlled
    else
        col_start = (obs_idx - 1) * 2 + 1  # obs_idx=1 -> cols 1:2, obs_idx=2 -> cols 3:4
        col_end = col_start + 1
    end
    
    ax = Axis(fig[sensitivity_row, col_start:col_end],
              xlabel="Target Δ$(observable_labels[obs_idx])",
              ylabel="Achieved Δ$(observable_labels[obs_idx])",
              xlabelsize=38,
              ylabelsize=38,
              xticklabelsize=34,
              yticklabelsize=34)
    
    x_vals = control_steps .* base_Δμ[obs_idx]
    
    lines!(ax, x_vals, actual_exact[obs_idx, :], 
           color=method_colors["Exact Score"], linewidth=5)
    lines!(ax, x_vals, actual_linear[obs_idx, :], 
           color=method_colors["Linear Score"], linewidth=5, linestyle=:dash)
    lines!(ax, x_vals, actual_J[obs_idx, :], 
           color=method_colors["Jacobian"], linewidth=5, linestyle=:dot)
    lines!(ax, x_vals, actual_clustered[obs_idx, :], 
           color=method_colors["KGMM Score"], linewidth=5, linestyle=:dashdot)
    lines!(ax, x_vals, x_vals, 
           color=method_colors["Perfect"], linewidth=4, linestyle=:solid, alpha=0.7)
end

# Add section title
Label(fig[sensitivity_row-1, 1:N_params_controlled], "Sensitivity Analysis", fontsize=44, font=:bold, tellwidth=false)

# LEGEND 1: For Parameter Control and Sensitivity Analysis (first 3 sections)
legend1_row = sensitivity_row + 1
legend_elements_1 = [
    LineElement(color=:blue, linewidth=6),
    LineElement(color=:green, linewidth=6, linestyle=:dash),
    LineElement(color=:purple, linewidth=6, linestyle=:dot),
    LineElement(color=:orange, linewidth=6, linestyle=:dashdot),
    LineElement(color=:red, linewidth=5, alpha=0.8)
]

legend_labels_1 = [
    "Exact Score",
    "Linear Score", 
    "Jacobian",
    "KGMM Score",
    "Perfect/Expected"
]

Legend(fig[legend1_row, 1:N_params_controlled], legend_elements_1, legend_labels_1, 
       orientation=:horizontal, framevisible=true, 
       labelsize=36, tellheight=true, tellwidth=false,
       margin=(20, 20, 20, 20))

# SECTION 3: Newton's Method Convergence - FULL WIDTH (2 plots spanning 2 columns each)
convergence_row = legend1_row + 2

# Each convergence plot spans 2 columns for full width
for obs_idx in 1:N_observables
    if N_observables == 1
        # Center single plot across the available columns
        col_start = 1
        col_end = N_params_controlled
    else
        col_start = (obs_idx - 1) * 2 + 1  # obs_idx=1 -> cols 1:2, obs_idx=2 -> cols 3:4
        col_end = col_start + 1
    end
    
    ax = Axis(fig[convergence_row, col_start:col_end],
              xlabel="Iteration",
              ylabel=observable_labels[obs_idx],
              xlabelsize=38,
              ylabelsize=38,
              xticklabelsize=34,
              yticklabelsize=34)
    
    # Plot ALL methods - Quasi-Newton (solid) and Full Newton (dashed)
    # Quasi-Newton methods (solid lines)
    lines!(ax, 0:length(observables_hist_J_quasi)-1, 
           get_observable_history(observables_hist_J_quasi, obs_idx), 
           color=method_colors["Jacobian"], linewidth=5, linestyle=:solid)
    lines!(ax, 0:length(observables_hist_R_exact_quasi)-1, 
           get_observable_history(observables_hist_R_exact_quasi, obs_idx), 
           color=method_colors["Exact Score"], linewidth=5, linestyle=:solid)
    lines!(ax, 0:length(observables_hist_R_linear_quasi)-1, 
           get_observable_history(observables_hist_R_linear_quasi, obs_idx), 
           color=method_colors["Linear Score"], linewidth=5, linestyle=:solid)
    lines!(ax, 0:length(observables_hist_R_clustered_quasi)-1, 
           get_observable_history(observables_hist_R_clustered_quasi, obs_idx), 
           color=method_colors["KGMM Score"], linewidth=5, linestyle=:solid)
    
    # Full Newton methods (dashed lines)
    lines!(ax, 0:length(observables_hist_J_full)-1, 
           get_observable_history(observables_hist_J_full, obs_idx), 
           color=method_colors["Jacobian"], linewidth=5, linestyle=:dash)
    lines!(ax, 0:length(observables_hist_R_exact_full)-1, 
           get_observable_history(observables_hist_R_exact_full, obs_idx), 
           color=method_colors["Exact Score"], linewidth=5, linestyle=:dash)
    lines!(ax, 0:length(observables_hist_R_linear_full)-1, 
           get_observable_history(observables_hist_R_linear_full, obs_idx), 
           color=method_colors["Linear Score"], linewidth=5, linestyle=:dash)
    lines!(ax, 0:length(observables_hist_R_clustered_full)-1, 
           get_observable_history(observables_hist_R_clustered_full, obs_idx), 
           color=method_colors["KGMM Score"], linewidth=5, linestyle=:dash)

    # Target line
    hlines!(ax, [target_observables[obs_idx]], 
            color=method_colors["Target"], linewidth=4, linestyle=:dash, alpha=0.8)
end

# Add section title
Label(fig[convergence_row-1, 1:N_params_controlled], "Newton's Method Convergence", fontsize=44, font=:bold, tellwidth=false)

# Create a clean, simple legend at the bottom
legend_row = convergence_row + 1  # Reduced gap

# LEGEND 2: For Newton's Method Convergence (last section)
legend_elements_2 = [
    LineElement(color=:blue, linewidth=6, linestyle=:solid),
    LineElement(color=:green, linewidth=6, linestyle=:solid),
    LineElement(color=:purple, linewidth=6, linestyle=:solid),
    LineElement(color=:orange, linewidth=6, linestyle=:solid),
    LineElement(color=:blue, linewidth=6, linestyle=:dash),
    LineElement(color=:green, linewidth=6, linestyle=:dash),
    LineElement(color=:purple, linewidth=6, linestyle=:dash),
    LineElement(color=:orange, linewidth=6, linestyle=:dash),
    LineElement(color=:red, linewidth=5, alpha=0.8)
]

legend_labels_2 = [
    "Exact Score (Quasi)",
    "Linear Score (Quasi)", 
    "Jacobian (Quasi)",
    "KGMM Score (Quasi)",
    "Exact Score (Full)",
    "Linear Score (Full)", 
    "Jacobian (Full)",
    "KGMM Score (Full)",
    "Target"
]

Legend(fig[legend_row, 1:N_params_controlled], legend_elements_2, legend_labels_2, 
       orientation=:horizontal, framevisible=true, 
       labelsize=36, tellheight=true, tellwidth=false,
       margin=(20, 20, 20, 20), nbanks=3)

# Much better layout spacing - more height, less width
rowgap!(fig.layout, 60)
colgap!(fig.layout, 30)

# Set specific row heights to make panels taller
for i in 1:(convergence_row+2)
    if i <= N_observables || i == sensitivity_row || i == convergence_row
        rowsize!(fig.layout, i, Relative(0.18))  # Make plot rows much taller
    end
end

# Display and save
display(fig)

# Switch to CairoMakie for PDF export
CairoMakie.activate!()
fig_pdf = Figure(size=(2400, 4000), fontsize=40)

# Recreate the figure with CairoMakie for PDF output
# SECTION 1: Parameter Control Analysis (adjust columns based on parameters controlled)
param_row_start = 1
for moment_idx in 1:N_observables
    for param_idx in 1:N_params_controlled
        row_idx = param_row_start + moment_idx - 1
        col_idx = param_idx
        
        ax = Axis(fig_pdf[row_idx, col_idx],
                  xlabel="Target Δ$(observable_labels[moment_idx])",
                  ylabel=param_labels[param_idx],
                  xlabelsize=36,
                  ylabelsize=36,
                  xticklabelsize=32,
                  yticklabelsize=32)
        
        x_vals = control_steps .* base_Δμ[moment_idx]
        
        # Plot with thicker lines and clear colors
        lines!(ax, x_vals, params_hist_exact[param_idx, :], 
               color=method_colors["Exact Score"], linewidth=4)
        lines!(ax, x_vals, params_hist_linear[param_idx, :], 
               color=method_colors["Linear Score"], linewidth=4, linestyle=:dash)
        lines!(ax, x_vals, params_hist_J[param_idx, :], 
               color=method_colors["Jacobian"], linewidth=4, linestyle=:dot)
        lines!(ax, x_vals, params_hist_clustered[param_idx, :], 
               color=method_colors["KGMM Score"], linewidth=4, linestyle=:dashdot)
    end
end

# Add section title
Label(fig_pdf[0, 1:N_params_controlled], "Parameter Control Analysis", fontsize=44, font=:bold, tellwidth=false)

# SECTION 2: Sensitivity Analysis - FULL WIDTH (2 plots spanning 2 columns each)
sensitivity_row = N_observables + 2

# Each sensitivity plot spans 2 columns for full width
for obs_idx in 1:N_observables
    if N_observables == 1
        # Center single plot across the available columns
        col_start = 1
        col_end = N_params_controlled
    else
        col_start = (obs_idx - 1) * 2 + 1  # obs_idx=1 -> cols 1:2, obs_idx=2 -> cols 3:4
        col_end = col_start + 1
    end
    
    ax = Axis(fig_pdf[sensitivity_row, col_start:col_end],
              xlabel="Target Δ$(observable_labels[obs_idx])",
              ylabel="Achieved Δ$(observable_labels[obs_idx])",
              xlabelsize=38,
              ylabelsize=38,
              xticklabelsize=34,
              yticklabelsize=34)
    
    x_vals = control_steps .* base_Δμ[obs_idx]
    
    lines!(ax, x_vals, actual_exact[obs_idx, :], 
           color=method_colors["Exact Score"], linewidth=5)
    lines!(ax, x_vals, actual_linear[obs_idx, :], 
           color=method_colors["Linear Score"], linewidth=5, linestyle=:dash)
    lines!(ax, x_vals, actual_J[obs_idx, :], 
           color=method_colors["Jacobian"], linewidth=5, linestyle=:dot)
    lines!(ax, x_vals, actual_clustered[obs_idx, :], 
           color=method_colors["KGMM Score"], linewidth=5, linestyle=:dashdot)
    lines!(ax, x_vals, x_vals, 
           color=method_colors["Perfect"], linewidth=4, linestyle=:solid, alpha=0.7)
end

# Add section title
Label(fig_pdf[sensitivity_row-1, 1:N_params_controlled], "Sensitivity Analysis", fontsize=44, font=:bold, tellwidth=false)

# LEGEND 1: For Parameter Control and Sensitivity Analysis (first 3 sections)
legend1_row = sensitivity_row + 1
legend_elements_1 = [
    LineElement(color=:blue, linewidth=6),
    LineElement(color=:green, linewidth=6, linestyle=:dash),
    LineElement(color=:purple, linewidth=6, linestyle=:dot),
    LineElement(color=:orange, linewidth=6, linestyle=:dashdot),
    LineElement(color=:red, linewidth=5, alpha=0.8)
]

legend_labels_1 = [
    "Exact Score",
    "Linear Score", 
    "Jacobian",
    "KGMM Score",
    "Perfect/Expected"
]

Legend(fig_pdf[legend1_row, 1:N_params_controlled], legend_elements_1, legend_labels_1, 
       orientation=:horizontal, framevisible=true, 
       labelsize=36, tellheight=true, tellwidth=false,
       margin=(20, 20, 20, 20))

# SECTION 3: Newton's Method Convergence - FULL WIDTH (2 plots spanning 2 columns each)
convergence_row = legend1_row + 2

# Each convergence plot spans 2 columns for full width
for obs_idx in 1:N_observables
    if N_observables == 1
        # Center single plot across the available columns
        col_start = 1
        col_end = N_params_controlled
    else
        col_start = (obs_idx - 1) * 2 + 1  # obs_idx=1 -> cols 1:2, obs_idx=2 -> cols 3:4
        col_end = col_start + 1
    end
    
    ax = Axis(fig_pdf[convergence_row, col_start:col_end],
              xlabel="Iteration",
              ylabel=observable_labels[obs_idx],
              xlabelsize=38,
              ylabelsize=38,
              xticklabelsize=34,
              yticklabelsize=34)
    
    # Plot ALL methods - Quasi-Newton (solid) and Full Newton (dashed)
    # Quasi-Newton methods (solid lines)
    lines!(ax, 0:length(observables_hist_J_quasi)-1, 
           get_observable_history(observables_hist_J_quasi, obs_idx), 
           color=method_colors["Jacobian"], linewidth=5, linestyle=:solid)
    lines!(ax, 0:length(observables_hist_R_exact_quasi)-1, 
           get_observable_history(observables_hist_R_exact_quasi, obs_idx), 
           color=method_colors["Exact Score"], linewidth=5, linestyle=:solid)
    lines!(ax, 0:length(observables_hist_R_linear_quasi)-1, 
           get_observable_history(observables_hist_R_linear_quasi, obs_idx), 
           color=method_colors["Linear Score"], linewidth=5, linestyle=:solid)
    lines!(ax, 0:length(observables_hist_R_clustered_quasi)-1, 
           get_observable_history(observables_hist_R_clustered_quasi, obs_idx), 
           color=method_colors["KGMM Score"], linewidth=5, linestyle=:solid)
    
    # Full Newton methods (dashed lines)
    lines!(ax, 0:length(observables_hist_J_full)-1, 
           get_observable_history(observables_hist_J_full, obs_idx), 
           color=method_colors["Jacobian"], linewidth=5, linestyle=:dash)
    lines!(ax, 0:length(observables_hist_R_exact_full)-1, 
           get_observable_history(observables_hist_R_exact_full, obs_idx), 
           color=method_colors["Exact Score"], linewidth=5, linestyle=:dash)
    lines!(ax, 0:length(observables_hist_R_linear_full)-1, 
           get_observable_history(observables_hist_R_linear_full, obs_idx), 
           color=method_colors["Linear Score"], linewidth=5, linestyle=:dash)
    lines!(ax, 0:length(observables_hist_R_clustered_full)-1, 
           get_observable_history(observables_hist_R_clustered_full, obs_idx), 
           color=method_colors["KGMM Score"], linewidth=5, linestyle=:dash)

    # Target line
    hlines!(ax, [target_observables[obs_idx]], 
            color=method_colors["Target"], linewidth=4, linestyle=:dash, alpha=0.8)
end

# Add section title
Label(fig_pdf[convergence_row-1, 1:N_params_controlled], "Newton's Method Convergence", fontsize=44, font=:bold, tellwidth=false)

# Create a clean, simple legend at the bottom
legend_row = convergence_row + 1

# LEGEND 2: For Newton's Method Convergence (last section)
legend_elements_2 = [
    LineElement(color=:blue, linewidth=6, linestyle=:solid),
    LineElement(color=:green, linewidth=6, linestyle=:solid),
    LineElement(color=:purple, linewidth=6, linestyle=:solid),
    LineElement(color=:orange, linewidth=6, linestyle=:solid),
    LineElement(color=:blue, linewidth=6, linestyle=:dash),
    LineElement(color=:green, linewidth=6, linestyle=:dash),
    LineElement(color=:purple, linewidth=6, linestyle=:dash),
    LineElement(color=:orange, linewidth=6, linestyle=:dash),
    LineElement(color=:red, linewidth=5, alpha=0.8)
]

legend_labels_2 = [
    "Exact Score (Quasi)",
    "Linear Score (Quasi)", 
    "Jacobian (Quasi)",
    "KGMM Score (Quasi)",
    "Exact Score (Full)",
    "Linear Score (Full)", 
    "Jacobian (Full)",
    "KGMM Score (Full)",
    "Target"
]

Legend(fig_pdf[legend_row, 1:N_params_controlled], legend_elements_2, legend_labels_2, 
       orientation=:horizontal, framevisible=true, 
       labelsize=36, tellheight=true, tellwidth=false,
       margin=(20, 20, 20, 20), nbanks=3)

# Much better layout spacing - more height, less width
rowgap!(fig_pdf.layout, 60)
colgap!(fig_pdf.layout, 30)

# Set specific row heights to make panels taller
for i in 1:(convergence_row+2)
    if i <= N_observables || i == sensitivity_row || i == convergence_row
        rowsize!(fig_pdf.layout, i, Relative(0.18))  # Make plot rows much taller
    end
end

# Save PDF with CairoMakie
save("reduced_add_sigma.pdf", fig_pdf)

# Switch back to GLMakie for interactive display
GLMakie.activate!()

println("✅ Publication-ready figure created with GLMakie:")
println("   📊 Combined Figure: Parameter Control Analysis (2400×4000 - PDF format)")
println("   📈 Section 1: Parameter Analysis ($(N_observables)×4 grid)")
println("   🎯 Section 2: Sensitivity Analysis (2 plots, each spanning 2 columns)")  
println("   🎯 Section 3: Convergence Analysis (2 plots, each spanning 2 columns)")
println("   🏷️  Simplified legend with main methods only")
println("   📄 Saved as PDF: combined_parameter_control_analysis_makie.pdf")

##
# --- Save All Data for Figure Reproduction ---
println("\n📀 Saving all data for figure reproduction...")

# Create data directory if it doesn't exist
data_dir = "scripts/ParameterControl/PC_data"
if !isdir(data_dir)
    mkpath(data_dir)
end

# Save all the essential data to HDF5 file
data_file = joinpath(data_dir, "reduced_add_sigma.h5")
h5open(data_file, "w") do file
    # --- System Parameters ---
    g_system = create_group(file, "system_parameters")
    g_system["a"] = a
    g_system["b"] = b  
    g_system["c"] = c
    g_system["F_tilde"] = F_tilde
    g_system["s"] = s
    g_system["D_eff"] = D_eff
    g_system["params_original"] = params_original
    g_system["param_indices_to_control"] = param_indices_to_control
    
    # --- Simulation Parameters ---
    g_sim = create_group(file, "simulation_parameters")
    g_sim["dt_sim"] = dt_sim
    g_sim["dt_obs"] = dt_obs
    g_sim["Nsteps"] = Nsteps
    g_sim["res"] = res
    g_sim["n_control"] = common_kwargs.n_control
    g_sim["base_Δμ"] = collect(common_kwargs.base_Δμ)
    g_sim["λ_sensitivity"] = common_kwargs.λ
    g_sim["λ_newton"] = newton_kwargs.λ
    g_sim["max_iters"] = newton_kwargs.max_iters
    
    # --- Observable Data ---
    g_obs = create_group(file, "observables")
    g_obs["initial_observables"] = initial_observables
    g_obs["target_observables"] = target_observables
    g_obs["M_obs"] = M_obs
    g_obs["S_obs"] = S_obs
    g_obs["N_observables"] = N_observables
    g_obs["observable_labels"] = observable_labels
    g_obs["param_labels"] = param_labels
    
    # --- Response Matrices ---
    g_matrices = create_group(file, "matrices")
    g_matrices["R_exact"] = R_exact
    g_matrices["R_linear"] = R_linear
    g_matrices["R_clustered"] = R_clustered
    g_matrices["J"] = J
    
    # --- Sensitivity Analysis Results ---
    g_sensitivity = create_group(file, "sensitivity_analysis")
    g_sensitivity["control_steps"] = collect(control_steps)
    
    # Parameter histories
    g_sensitivity["params_hist_exact"] = params_hist_exact
    g_sensitivity["params_hist_linear"] = params_hist_linear
    g_sensitivity["params_hist_J"] = params_hist_J
    g_sensitivity["params_hist_clustered"] = params_hist_clustered
    
    # Actual changes
    g_sensitivity["actual_exact"] = actual_exact
    g_sensitivity["actual_linear"] = actual_linear
    g_sensitivity["actual_J"] = actual_J
    g_sensitivity["actual_clustered"] = actual_clustered
    
    # --- Newton's Method Results ---
    g_newton = create_group(file, "newton_optimization")
    
    # Convert observable histories to matrices for storage
    function observable_history_to_matrix(history)
        n_iters = length(history)
        n_obs = length(history[1])
        result = zeros(n_obs, n_iters)
        for (i, obs) in enumerate(history)
            result[:, i] = obs
        end
        return result
    end
    
    g_newton["observables_hist_J_quasi"] = observable_history_to_matrix(observables_hist_J_quasi)
    g_newton["observables_hist_R_exact_quasi"] = observable_history_to_matrix(observables_hist_R_exact_quasi)
    g_newton["observables_hist_R_linear_quasi"] = observable_history_to_matrix(observables_hist_R_linear_quasi)
    g_newton["observables_hist_R_clustered_quasi"] = observable_history_to_matrix(observables_hist_R_clustered_quasi)
    g_newton["observables_hist_J_full"] = observable_history_to_matrix(observables_hist_J_full)
    g_newton["observables_hist_R_exact_full"] = observable_history_to_matrix(observables_hist_R_exact_full)
    g_newton["observables_hist_R_linear_full"] = observable_history_to_matrix(observables_hist_R_linear_full)
    g_newton["observables_hist_R_clustered_full"] = observable_history_to_matrix(observables_hist_R_clustered_full)
    
    # --- Figure Parameters ---
    g_figure = create_group(file, "figure_parameters")
    g_figure["fig_size_width"] = 2400
    g_figure["fig_size_height"] = 4000
    g_figure["fontsize"] = 40
    g_figure["section_title_fontsize"] = 44
    g_figure["xlabel_fontsize"] = 36
    g_figure["ylabel_fontsize"] = 36
    g_figure["tick_fontsize"] = 32
    g_figure["legend_fontsize"] = 36
    g_figure["linewidth_main"] = 5
    g_figure["linewidth_secondary"] = 4
end

println("✅ Data saved to: $data_file")
println("   📊 System parameters, matrices, and simulation results")
println("   📈 Sensitivity analysis data (4 methods × $(length(control_steps)) steps)")
println("   🎯 Newton optimization histories (8 methods)")
println("   🎨 Figure formatting parameters")

##
# --- Data Loading Function ---
"""
    load_parameter_control_data(data_file)

Load all parameter control analysis data from HDF5 file for figure reproduction.
Returns a named tuple with all the data organized by category.
"""
function load_parameter_control_data(data_file::String)
    println("\n📖 Loading parameter control analysis data from: $data_file")
    
    if !isfile(data_file)
        error("Data file not found: $data_file")
    end
    
    h5open(data_file, "r") do file
        # --- Load System Parameters ---
        g_system = file["system_parameters"]
        system_params = (
            a = read(g_system["a"]),
            b = read(g_system["b"]),
            c = read(g_system["c"]),
            F_tilde = read(g_system["F_tilde"]),
            s = read(g_system["s"]),
            D_eff = read(g_system["D_eff"]),
            params_original = read(g_system["params_original"]),
            param_indices_to_control = read(g_system["param_indices_to_control"])
        )
        
        # --- Load Simulation Parameters ---
        g_sim = file["simulation_parameters"]
        sim_params = (
            dt_sim = read(g_sim["dt_sim"]),
            dt_obs = read(g_sim["dt_obs"]),
            Nsteps = read(g_sim["Nsteps"]),
            res = read(g_sim["res"]),
            n_control = read(g_sim["n_control"]),
            base_Δμ = read(g_sim["base_Δμ"]),
            λ_sensitivity = read(g_sim["λ_sensitivity"]),
            λ_newton = read(g_sim["λ_newton"]),
            max_iters = read(g_sim["max_iters"])
        )
        
        # --- Load Observable Data ---
        g_obs = file["observables"]
        observable_data = (
            initial_observables = read(g_obs["initial_observables"]),
            target_observables = read(g_obs["target_observables"]),
            M_obs = read(g_obs["M_obs"]),
            S_obs = read(g_obs["S_obs"]),
            N_observables = read(g_obs["N_observables"]),
            observable_labels = read(g_obs["observable_labels"]),
            param_labels = read(g_obs["param_labels"])
        )
        
        # --- Load Response Matrices ---
        g_matrices = file["matrices"]
        matrices = (
            R_exact = read(g_matrices["R_exact"]),
            R_linear = read(g_matrices["R_linear"]),
            R_clustered = read(g_matrices["R_clustered"]),
            J = read(g_matrices["J"])
        )
        
        # --- Load Sensitivity Analysis Results ---
        g_sensitivity = file["sensitivity_analysis"]
        sensitivity_data = (
            control_steps = read(g_sensitivity["control_steps"]),
            params_hist_exact = read(g_sensitivity["params_hist_exact"]),
            params_hist_linear = read(g_sensitivity["params_hist_linear"]),
            params_hist_J = read(g_sensitivity["params_hist_J"]),
            params_hist_clustered = read(g_sensitivity["params_hist_clustered"]),
            actual_exact = read(g_sensitivity["actual_exact"]),
            actual_linear = read(g_sensitivity["actual_linear"]),
            actual_J = read(g_sensitivity["actual_J"]),
            actual_clustered = read(g_sensitivity["actual_clustered"])
        )
        
        # --- Load Newton's Method Results ---
        g_newton = file["newton_optimization"]
        
        # Convert matrices back to vector of vectors for compatibility
        function matrix_to_observable_history(matrix)
            n_obs, n_iters = size(matrix)
            return [matrix[:, i] for i in 1:n_iters]
        end
        
        newton_data = (
            observables_hist_J_quasi = matrix_to_observable_history(read(g_newton["observables_hist_J_quasi"])),
            observables_hist_R_exact_quasi = matrix_to_observable_history(read(g_newton["observables_hist_R_exact_quasi"])),
            observables_hist_R_linear_quasi = matrix_to_observable_history(read(g_newton["observables_hist_R_linear_quasi"])),
            observables_hist_R_clustered_quasi = matrix_to_observable_history(read(g_newton["observables_hist_R_clustered_quasi"])),
            observables_hist_J_full = matrix_to_observable_history(read(g_newton["observables_hist_J_full"])),
            observables_hist_R_exact_full = matrix_to_observable_history(read(g_newton["observables_hist_R_exact_full"])),
            observables_hist_R_linear_full = matrix_to_observable_history(read(g_newton["observables_hist_R_linear_full"])),
            observables_hist_R_clustered_full = matrix_to_observable_history(read(g_newton["observables_hist_R_clustered_full"]))
        )
        
        # --- Load Figure Parameters ---
        g_figure = file["figure_parameters"]
        figure_params = (
            fig_size = (read(g_figure["fig_size_width"]), read(g_figure["fig_size_height"])),
            fontsize = read(g_figure["fontsize"]),
            section_title_fontsize = read(g_figure["section_title_fontsize"]),
            xlabel_fontsize = read(g_figure["xlabel_fontsize"]),
            ylabel_fontsize = read(g_figure["ylabel_fontsize"]),
            tick_fontsize = read(g_figure["tick_fontsize"]),
            legend_fontsize = read(g_figure["legend_fontsize"]),
            linewidth_main = read(g_figure["linewidth_main"]),
            linewidth_secondary = read(g_figure["linewidth_secondary"])
        )
        
        println("✅ Successfully loaded all data:")
        println("   📊 System & simulation parameters")
        println("   📈 $(length(sensitivity_data.control_steps)) sensitivity analysis steps")
        println("   🎯 Newton optimization histories for 8 methods")
        println("   🎨 Figure formatting parameters")
        
        return (
            system = system_params,
            simulation = sim_params,
            observables = observable_data,
            matrices = matrices,
            sensitivity = sensitivity_data,
            newton = newton_data,
            figure = figure_params
        )
    end
end

# Example usage:
# To load the data later:
# data = load_parameter_control_data("scripts/ParameterControl/PC_data/reduced_add_sigma.h5")
# Then access data like: data.system.a, data.sensitivity.actual_exact, data.newton.observables_hist_J_quasi, etc.

println("\n💾 Data saving and loading functions ready!")
println("   📁 Data will be saved to: scripts/ParameterControl/PC_data/")
println("   🔄 Use load_parameter_control_data() to reload all data for figure reproduction")
