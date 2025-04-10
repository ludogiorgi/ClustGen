"""
    covariance(x1, x2; timesteps=length(x), progress=false)

Compute the empirical time-lagged covariance between two time series.

# Arguments
- `x1`: First time series
- `x2`: Second time series
- `timesteps`: Maximum number of time lags to compute (default: length of series)
- `progress`: Whether to show progress bar (default: false)

# Returns
- Vector of covariance values at each lag
"""
function covariance(x1, x2; timesteps=length(x1), progress=false)
    # Calculate means for centering
    μ1 = mean(x1)
    μ2 = mean(x2)
    
    # Initialize covariance array
    autocor = zeros(timesteps)
    
    # Set up iterator with optional progress bar
    progress ? iter = ProgressBar(1:timesteps) : iter = 1:timesteps
    
    # Compute time-lagged covariance for each lag
    for i in iter
        autocor[i] = mean(x1[i:end] .* x2[1:end-i+1]) - μ1 * μ2
    end
    
    return autocor
end

"""
    covariance(g⃗1, g⃗2, Q::Eigen, timelist; progress=false)

Compute covariance using eigenvector decomposition of generator matrix.

# Arguments
- `g⃗1`: First observable vector
- `g⃗2`: Second observable vector
- `Q`: Eigensystem of generator matrix (eigenvalues and eigenvectors)
- `timelist`: List of time points at which to evaluate covariance
- `progress`: Whether to show progress bar (default: false)

# Returns
- Covariance evaluated at each time point
"""
function covariance(g⃗1, g⃗2, Q::Eigen, timelist; progress=false)
    # Uncomment to check for ergodicity
    # @assert all(real.(Q.values[1:end-1]) .< 0) "Did not pass an ergodic generator matrix"
    
    # Initialize covariance array
    autocov = zeros(length(timelist))
    
    # Unpack eigenvalues and eigenvectors
    Λ, V = Q
    
    # Compute stationary distribution from eigenvector of zero eigenvalue
    p = real.(V[:, end] ./ sum(V[:, end]))
    
    # Project observables onto eigenbasis
    v1 = V \ (p .* g⃗1)
    w2 = g⃗2' * V
    
    # Calculate means for centering
    μ1 = sum(p .* g⃗1)
    μ2 = sum(p .* g⃗2)
    
    # Set up iterator with optional progress bar
    progress ? iter = ProgressBar(eachindex(timelist)) : iter = eachindex(timelist)
    
    # Compute covariance at each time point
    for i in iter
        autocov[i] = real(w2 * (exp.(Λ .* timelist[i]) .* v1) - μ1 * μ2)
    end
    
    return autocov
end

"""
    covariance(g⃗1, g⃗2, Q, timelist; progress=false)

Compute covariance using generator matrix directly (computes eigendecomposition).

# Arguments
- `g⃗1`: First observable vector
- `g⃗2`: Second observable vector
- `Q`: Generator matrix
- `timelist`: List of time points at which to evaluate covariance
- `progress`: Whether to show progress bar (default: false)

# Returns
- Covariance evaluated at each time point
"""
covariance(g⃗1, g⃗2, Q, timelist; progress=false) = covariance(g⃗1, g⃗2, eigen(Q), timelist; progress=progress)

"""
    cleaning(averages, centers, labels)

Clean up cluster assignments by renumbering to ensure contiguous indices.

# Arguments
- `averages`: Matrix of average values for each cluster
- `centers`: Matrix of cluster centers
- `labels`: Vector of cluster assignments

# Returns
- Tuple containing:
  - `averages_new`: Filtered average values
  - `centers_new`: Filtered cluster centers
  - Number of unique clusters
  - `labels_new`: Remapped cluster assignments
"""
function cleaning(averages, centers, labels)
    # Find unique cluster indices sorted in ascending order
    unique_clusters = sort(unique(labels))
    
    # Create mapping from old cluster indices to new contiguous indices
    mapping = Dict(old_cluster => new_cluster for (new_cluster, old_cluster) in enumerate(unique_clusters))
    
    # Apply mapping to labels
    labels_new = [mapping[cluster] for cluster in labels]
    
    # Select columns corresponding to valid clusters
    averages_new = averages[:, unique_clusters]
    centers_new = centers[:, unique_clusters]
    
    return averages_new, centers_new, length(unique_clusters), labels_new
end

"""
    model_to_score(nn, dim_in, normalization_params=nothing)

Convert a trained neural network to a score function.

# Arguments
- `nn`: Neural network model
- `dim_in`: Input dimension
- `normalization_params`: Optional tuple (max, min) for denormalizing outputs

# Returns
- Score function that takes a state vector and returns gradient of log density
"""
function model_to_score(nn, dim_in, normalization_params=nothing)
    # Create score function based on whether normalization was used
    if normalization_params === nothing
        # No normalization case
        function score(x)
            return nn([x...])
        end
    else
        # Denormalize outputs using provided parameters
        M, m = normalization_params
        
        function score(x)
            # Normalize neural network output to recover original scale
            return nn([x...]) .* (M - m) .+ m
        end
    end
    
    return score
end

"""
    compute_density_from_score(x, score_fn; reference_point=nothing, normalize=true)

Convert a score function to a probability density function in any dimension.

# Arguments
- `x`: Points where density should be evaluated:
     - For 1D: A vector of points
     - For nD: A matrix where each column is an n-dimensional point, or
              A vector of n-dimensional vectors
- `score_fn`: Score function s(x) = ∇ ln ρ(x)
- `reference_point`: Reference point for integration (default: origin)
- `normalize`: Whether to normalize the result (default: true)

# Returns
- Vector of density values ρ(x) for each point
"""
function compute_density_from_score(x, score_fn; reference_point=nothing, normalize=true)
    # Convert input to standardized format
    if x isa AbstractVector && !(eltype(x) <: AbstractVector)
        # 1D case with vector of scalars
        points = reshape(x, 1, length(x))
        dim = 1
    elseif x isa AbstractVector && eltype(x) <: AbstractVector
        # Vector of vectors case
        dim = length(x[1])
        n_points = length(x)
        points = zeros(dim, n_points)
        for i in 1:n_points
            points[:,i] = x[i]
        end
    elseif x isa AbstractMatrix
        # Already a matrix with points as columns
        points = x
        dim, n_points = size(points)
    else
        error("Input must be a vector or matrix")
    end
    
    # Set default reference point (origin)
    if isnothing(reference_point)
        reference_point = zeros(dim)
    end
    
    # Special case for 1D - use optimized approach
    if dim == 1
        return compute_density_1d(points, score_fn, reference_point, normalize)
    end
    
    # General case for higher dimensions
    log_densities = zeros(n_points)
    
    # Progress tracking
    prog = Progress(n_points, desc="Computing $(dim)D densities: ")
    
    # Compute log densities through path integration
    Threads.@threads for i in 1:n_points
        point = points[:,i]
        
        # Path from reference_point to point
        path_vector = point - reference_point
        
        # Line integral along the path
        function integrand(t)
            # Point along the path at parameter t
            x_t = reference_point + t * path_vector
            # Score at this point
            s_xt = score_fn(x_t)
            # Dot product with path tangent
            return dot(s_xt, path_vector)
        end
        
        # Compute integral from t=0 to t=1
        integral, _ = quadgk(integrand, 0, 1, rtol=1e-6)
        
        # Log density is negative of the integral
        log_densities[i] = -integral
        
        next!(prog)
    end
    
    # Convert to actual densities
    densities = exp.(log_densities)
    
    # Normalize if requested and possible
    if normalize && !all(iszero, densities)
        if dim == 1
            # For 1D, normalize with trapezoidal rule
            dx = points[1,2] - points[1,1]
            total = sum(densities) * dx
            densities ./= total
        else
            # For higher dimensions, just normalize by sum
            # (proper normalization would require multi-dimensional integration)
            densities ./= sum(densities)
        end
    end
    
    return densities
end

# Helper function for 1D specific case (same as your current implementation)
function compute_density_1d(points, score_fn, reference_point, normalize)
    n_points = size(points, 2)
    reference = reference_point[1]
    log_densities = zeros(n_points)
    
    # Progress tracking
    prog = Progress(n_points, desc="Computing 1D densities: ")
    
    # Compute log-densities
    Threads.@threads for i in 1:n_points
        xi = points[1,i]
        
        # Integration with correct sign handling
        if xi < reference
            integral, _ = quadgk(v -> -score_fn([v])[1], xi, reference, rtol=1e-6)
        else
            integral, _ = quadgk(v -> score_fn([v])[1], reference, xi, rtol=1e-6)
        end
        
        log_densities[i] = integral
        
        next!(prog)
    end
    
    # Convert log-densities to densities
    densities = exp.(log_densities)
    
    # Normalize if requested
    if normalize && !all(iszero, densities)
        dx = points[1,2] - points[1,1]  # Assuming uniform grid
        total = sum(densities) * dx
        densities ./= total
    end
    
    return densities
end

"""
    maxent_distribution(moments; bounds=(-5.0, 5.0), grid_size=500, max_iterations=3000)

Compute a maximum entropy probability distribution that matches the given moments.
Uses parallel optimization with multiple starting points and strategies.

# Arguments
- `moments`: Vector of moments to match (starting with 0th moment = 1)
- `bounds`: Domain boundaries for computation (lower, upper)
- `grid_size`: Number of points in the output grid
- `max_iterations`: Maximum number of iterations for optimization
"""
function maxent_distribution(moments; bounds=(-5.0, 8.0), grid_size=500, max_iterations=3000)
    # Create grid for visualization
    xs = range(bounds[1], bounds[2], length=grid_size)
    N = length(moments)
    
    println("Computing maximum entropy distribution for $N moments...")
    
    # Extract key statistics for better initialization
    μ = moments[2]  # Mean (1st moment)
    σ² = moments[3] - μ^2  # Variance 
    σ = sqrt(max(σ², 1e-6))
    
    # Higher standardized moments if available
    γ₁ = N >= 4 ? (moments[4] - 3*μ*σ² - μ^3) / σ^3 : 0.0  # Skewness
    γ₂ = N >= 5 ? (moments[5] - 4*μ*moments[4] + 6*μ^2*σ² + 3*μ^4) / σ^4 - 3 : 0.0  # Excess kurtosis
    
    println("Distribution statistics: mean=$μ, std=$σ, skewness=$γ₁, excess_kurtosis=$γ₂")
    
    # Cache for avoiding redundant computations
    moment_cache = Dict{Vector{Float64}, Vector{Float64}}()
    z_cache = Dict{Vector{Float64}, Float64}()
    
    # PDF function given Lagrange multipliers λ
    function pdf(x, λ)
        exponent = sum(λ[k] * x^(k-1) for k in 1:N)
        return exp(-exponent)
    end
    
    # Partition function (normalization) with caching
    function Z(λ)
        λ_key = round.(λ, digits=10)  # Round for dictionary key
        if haskey(z_cache, λ_key)
            return z_cache[λ_key]
        end
        
        # Adaptive integration based on parameter values
        rtol = if any(abs.(λ) .> 10.0)
            1e-8  # Higher precision for extreme parameters
        else
            1e-6  # Standard precision for normal ranges
        end
        
        integrand(x) = pdf(x, λ)
        result, _ = quadgk(integrand, bounds[1], bounds[2], rtol=rtol)
        
        # Cache the result
        z_cache[λ_key] = result
        return result
    end
    
    # Compute moments from λ values with caching
    function compute_moments(λ)
        λ_key = round.(λ, digits=10)  # Round for dictionary key
        if haskey(moment_cache, λ_key)
            return moment_cache[λ_key]
        end
        
        z = Z(λ)
        computed = zeros(N)
        
        # Compute all moments in one pass when possible
        if N ≤ 6  # For small N, we can be more efficient
            extended_moments = zeros(N)
            for k in 1:N
                integrand(x) = x^(k-1) * pdf(x, λ)
                moment, _ = quadgk(integrand, bounds[1], bounds[2], rtol=1e-8)
                extended_moments[k] = moment
            end
            computed = extended_moments ./ z
        else
            # For larger N, parallelize moment calculations
            for k in 1:N
                integrand(x) = x^(k-1) * pdf(x, λ)
                moment, _ = quadgk(integrand, bounds[1], bounds[2], rtol=1e-8)
                computed[k] = moment / z
            end
        end
        
        # Cache the result
        moment_cache[λ_key] = computed
        return computed
    end
    
    # Optimized objective function
    function objective(λ)
        try
            computed = compute_moments(λ)
            
            # Progressive weighting with more emphasis on matching all moments
            weights = [5.0, 3.0, 2.0, 1.5, 1.0, 0.7]  # Balanced weights
            weights = weights[1:min(length(weights), N)]
            
            if N > length(weights)
                append!(weights, fill(0.5, N - length(weights)))
            end
            
            # Calculate errors efficiently
            errors = zeros(N)
            for i in 1:N
                target = moments[i]
                achieved = computed[i]
                
                if abs(target) < 1e-10
                    errors[i] = weights[i] * (achieved - target)^2
                else
                    errors[i] = weights[i] * ((achieved - target)/abs(target))^2
                end
            end
            
            return sum(errors)
        catch e
            return 1e10
        end
    end
    
    # Regularized Newton-Raphson for better conditioning
    function newton_moment_match(λ_start, iters=15)
        λ = copy(λ_start)
        
        # Start with larger regularization and reduce it
        reg_factor = 1e-3
        
        for i in 1:iters
            try
                # Compute current moments and error
                current = compute_moments(λ)
                diff = current - moments
                err = sum(diff.^2)
                
                # Early termination if already very accurate
                if err < 1e-12
                    println("Newton converged in $i iterations")
                    break
                end
                
                # Compute Jacobian and regularize it
                J = moment_jacobian(λ)
                J_reg = J + reg_factor * I  # Add regularization
                
                # Use SVD-based solver for better numerical stability
                U, S, V = svd(J_reg)
                tol = maximum(size(J_reg)) * eps(maximum(S))
                S_inv = map(s -> s > tol ? 1/s : 0.0, S)
                update = V * Diagonal(S_inv) * U' * diff
                
                # Line search
                step = 1.0
                new_λ = λ - step * update
                
                # Try to compute new error
                try
                    new_err = sum((compute_moments(new_λ) - moments).^2)
                    
                    # Backtrack if needed
                    while new_err > err && step > 1e-6
                        step *= 0.5
                        new_λ = λ - step * update
                        new_err = sum((compute_moments(new_λ) - moments).^2)
                    end
                    
                    # Update λ if improved
                    if new_err < err
                        λ = new_λ
                        # Reduce regularization on success
                        reg_factor = max(1e-10, reg_factor * 0.8)
                    else
                        # Increase regularization on failure
                        reg_factor = min(1e-1, reg_factor * 2.0)
                        λ = λ .+ 0.005 .* randn(length(λ))
                    end
                catch
                    # If error computation fails, just perturb and continue
                    reg_factor = min(1e-1, reg_factor * 2.0)
                    λ = λ .+ 0.01 .* randn(length(λ))
                end
                
                # Print progress sparingly
                if i % 5 == 0
                    println("Newton iteration $i: error = $err")
                end
            catch e
                # Be more resilient to numerical issues
                reg_factor = min(1e-1, reg_factor * 3.0)
                λ = λ .+ 0.02 .* randn(length(λ))
                println("Newton iteration $i recovered from error")
            end
        end
        
        return λ
    end
    
    # Gradient of moments with respect to λ (for Newton-based methods)
    function moment_jacobian(λ)
        # Use existing moment calculations when possible
        computed = compute_moments(λ)
        z = Z(λ)
        
        # More efficient computation of the extended moments
        extended_moments = zeros(2*N - 1)
        extended_moments[1:N] = computed
        
        # Compute the higher moments needed for the Jacobian
        for i in (N+1):(2*N-1)
            integrand(x) = x^(i-1) * pdf(x, λ)
            moment, _ = quadgk(integrand, bounds[1], bounds[2], rtol=1e-8)
            extended_moments[i] = moment / z
        end
        
        # Compute the Jacobian
        jac = zeros(N, N)
        for i in 1:N
            for j in 1:N
                # Cov(X^(i-1), X^(j-1)) = E[X^(i+j-2)] - E[X^(i-1)]E[X^(j-1)]
                jac[i,j] = -(extended_moments[i+j-1] - extended_moments[i]*extended_moments[j])
            end
        end
        
        return jac
    end
    
    # Generate starting points more intelligently
    function generate_starting_points()
        points = []
        
        # Universal starting points
        push!(points, zeros(N))  # Uniform
        push!(points, [0.0, 0.0, 0.5, zeros(N-3)...])  # Standard normal
        
        # Normal distribution matched to mean and variance
        normal_start = zeros(N)
        if N >= 3
            normal_start[1] = -log(sqrt(2π*σ²))
            normal_start[2] = μ/(σ²)
            normal_start[3] = 1/(2*σ²)
            push!(points, normal_start)
        end
        
        # Gamma distribution for positive skewness
        if N >= 3 && γ₁ > 0
            gamma_start = zeros(N)
            k = max(1.0, 4/γ₁^2)
            θ = σ/sqrt(k)
            gamma_start[1] = -k*log(θ) - lgamma(k)
            gamma_start[2] = (k-1)/θ
            gamma_start[3] = 1/θ
            push!(points, gamma_start)
        end
        
        # Starting points for skewness
        if N >= 4
            skew_dir = sign(γ₁)
            if abs(γ₁) > 0.1
                push!(points, [0.0, 0.0, 0.5, -0.1*skew_dir, zeros(N-4)...])
                push!(points, [0.0, 0.0, 0.5, -0.2*skew_dir, zeros(N-4)...])
            end
        end
        
        # Starting points for kurtosis
        if N >= 5
            kurt_dir = sign(γ₂)
            if abs(γ₂) > 0.5
                push!(points, [0.0, 0.0, 0.5, 0.0, 0.01*kurt_dir, zeros(N-5)...])
                push!(points, [0.0, 0.0, 0.3, -0.1*skew_dir, 0.02*kurt_dir, zeros(N-5)...])
            end
        end
        
        return points
    end
    
    # Function to parallelize the optimization of starting points
    function parallel_optimize_starting_points(λ_starts)
        n_points = length(λ_starts)
        results = Vector{Any}(undef, n_points)
        
        # Use threads for parallel optimization
        Threads.@threads for i in 1:n_points
            try
                # First try with NelderMead
                result = optimize(
                    objective, 
                    λ_starts[i], 
                    NelderMead(),
                    Optim.Options(iterations=max_iterations÷2, g_tol=1e-8)
                )
                
                # Store the result
                results[i] = (
                    error = result.minimum,
                    lambda = result.minimizer,
                    source = "NelderMead"
                )
                
                # Try LBFGS refinement for promising results
                if result.minimum < 0.01
                    refined = optimize(
                        objective, 
                        result.minimizer, 
                        LBFGS(),
                        Optim.Options(iterations=500, g_tol=1e-10)
                    )
                    
                    if refined.minimum < result.minimum
                        results[i] = (
                            error = refined.minimum,
                            lambda = refined.minimizer,
                            source = "LBFGS"
                        )
                    end
                end
            catch e
                results[i] = nothing
            end
        end
        
        # Filter out failed optimizations
        valid_results = filter(x -> x !== nothing, results)
        
        # Sort by error
        sort!(valid_results, by = x -> x.error)
        
        return valid_results
    end
    
    # Generate starting points
    λ_starts = generate_starting_points()
    println("Generated $(length(λ_starts)) starting points")
    
    # Parallelize optimization of starting points
    println("\nStage 1: Parallel optimization of starting points")
    optimization_results = parallel_optimize_starting_points(λ_starts)
    
    # Get best result
    if isempty(optimization_results)
        error("All optimization attempts failed")
    end
    
    best_result = first(optimization_results)
    best_λ = best_result.lambda
    best_error = best_result.error
    
    println("  Best solution ($(best_result.source)): error = $best_error")
    
    # Try Newton-Raphson refinement for the best points
    println("\nStage 2: Newton-Raphson refinement")
    
    # Use up to 3 best points for refinement
    for i in 1:min(3, length(optimization_results))
        result = optimization_results[i]
        try
            println("  Refining solution #$i ($(result.source), error = $(result.error))...")
            newton_λ = newton_moment_match(result.lambda)
            newton_error = objective(newton_λ)
            
            if newton_error < best_error
                best_error = newton_error
                best_λ = newton_λ
                println("  Improved with Newton-Raphson: error = $best_error")
            end
        catch e
            println("  Newton-Raphson refinement failed for point $i")
        end
    end
    
    # Final LBFGS optimization
    println("\nStage 3: Final LBFGS refinement")
    try
        result = optimize(
            objective, 
            best_λ, 
            LBFGS(),
            Optim.Options(iterations=1000, g_tol=1e-12)
        )
        
        if result.minimum < best_error
            best_error = result.minimum
            best_λ = result.minimizer
            println("  Improved with LBFGS: error = $best_error")
        end
    catch e
        println("  Final LBFGS refinement failed")
    end
    
    # Final λ values
    println("\nFinal Lagrange multipliers λ = $best_λ")
    
    # Compute normalizing constant
    Z_final = Z(best_λ)
    
    # Create normalized PDF function
    final_pdf(x) = pdf(x, best_λ) / Z_final
    
    # Compute PDF values on the grid - parallelize for large grids
    ys = if grid_size > 1000
        # Parallel computation for large grids
        shared_ys = SharedArray{Float64}(grid_size)
        @sync @distributed for i in 1:grid_size
            shared_ys[i] = final_pdf(xs[i])
        end
        Array(shared_ys)
    else
        # Direct computation for smaller grids
        [final_pdf(x) for x in xs]
    end
    
    # Verify final moments efficiently
    final_moments = compute_moments(best_λ)
    
    println("\nTarget moments: $moments")
    println("Achieved moments: $final_moments")
    
    # Calculate relative errors
    rel_errors = 100.0 * abs.(final_moments .- moments) ./ (abs.(moments) .+ 1e-10)
    println("Relative errors (%): $rel_errors")
    
    return final_pdf, xs, ys, best_λ
end

# # Add parallel workers if not already added
# if nprocs() == 1
#     addprocs(min(4, Sys.CPU_THREADS-1))  # Add workers but leave one core free
#     println("Added $(nprocs()-1) worker processes")
    
#     # Load necessary packages on all workers
#     @everywhere begin
#         using Optim
#         using QuadGK
#         using LinearAlgebra
#         using SpecialFunctions
#     end
# end

# using Optim
# using QuadGK
# using LinearAlgebra
# using SpecialFunctions
# using ForwardDiff

# function solve_maxent_moments(moments; bounds=(-10.0, 10.0), tol=1e-8, max_iter=5000)
#     # Ensure we have 4 moments
#     if length(moments) != 4
#         error("Exactly 4 moments (m₁, m₂, m₃, m₄) must be provided")
#     end
    
#     μ = moments[1]
#     σ² = moments[2] - μ^2
#     σ = sqrt(max(σ², 1e-6))
    
#     # Determine appropriate bounds based on moments
#     # This is critical for proper integration in heavy-tailed cases
#     auto_bounds = (μ - 5*σ, μ + 5*σ)
#     # Extend bounds further if skewness or kurtosis suggest heavy tails
#     γ₁ = (moments[3] - 3*μ*σ² - μ^3) / σ^3  # Skewness
#     γ₂ = (moments[4] - 4*μ*moments[3] + 6*μ^2*σ² + 3*μ^4) / σ^4 - 3  # Excess kurtosis
    
#     if abs(γ₁) > 1.0 || γ₂ > 3.0
#         # Extend bounds for heavy tails or high skewness
#         auto_bounds = (μ - (7 + abs(γ₁))*σ, μ + (7 + abs(γ₁))*σ)
#     end
    
#     # Use user-provided bounds or auto-determined bounds, whichever is wider
#     actual_bounds = (min(bounds[1], auto_bounds[1]), max(bounds[2], auto_bounds[2]))
    
#     println("Using integration bounds: $actual_bounds")
#     println("Moment statistics: mean=$μ, std=$σ, skewness=$γ₁, excess_kurtosis=$γ₂")
    
#     # Define the unnormalized PDF function with improved numerical stability
#     function unnormalized_pdf(x, λ)
#         # Avoid extreme exponent values that cause overflow/underflow
#         exponent = λ[1]*x + λ[2]*x^2 + λ[3]*x^3 + λ[4]*x^4
#         # Cap the exponent to avoid numerical issues
#         capped_exponent = max(min(exponent, 500.0), -500.0)
#         return exp(capped_exponent)
#     end
    
#     # Calculate partition function with improved precision
#     function Z(λ)
#         # Adaptively set precision based on parameter magnitudes
#         rtol = if any(abs.(λ) .> 5.0)
#             1e-12  # Higher precision for extreme parameters
#         else
#             1e-10  # Standard precision
#         end
        
#         try
#             integrand(x) = unnormalized_pdf(x, λ)
#             result, error = quadgk(integrand, actual_bounds[1], actual_bounds[2], rtol=rtol)
            
#             # Check for suspicious results
#             if result <= 0 || !isfinite(result)
#                 @warn "Integration returned invalid result: $result"
#                 return 1e-10  # Fallback value
#             end
            
#             return result
#         catch e
#             @warn "Integration error in Z calculation: $e"
#             return 1e-10  # Fallback value
#         end
#     end
    
#     # Calculate moments with robustness improvements
#     function compute_moments(λ)
#         z = Z(λ)
#         calculated_moments = zeros(4)
        
#         for k in 1:4
#             try
#                 # Use higher precision for higher moments
#                 rtol = 1e-10 * (1.0 / (k+1))
                
#                 integrand(x) = x^k * unnormalized_pdf(x, λ)
#                 moment, _ = quadgk(integrand, actual_bounds[1], actual_bounds[2], rtol=rtol)
#                 calculated_moments[k] = moment / z
#             catch e
#                 @warn "Moment calculation error for k=$k: $e"
#                 # Fallback to crude approximation if integration fails
#                 xs = range(actual_bounds[1], actual_bounds[2], length=10000)
#                 pdf_vals = [unnormalized_pdf(x, λ) for x in xs]
#                 pdf_vals ./= sum(pdf_vals) * (xs[2] - xs[1])
#                 calculated_moments[k] = sum(xs.^k .* pdf_vals) * (xs[2] - xs[1])
#             end
#         end
        
#         return calculated_moments
#     end
    
#     # Progressive objective function that starts with lower moments
#     # and gradually increases weight on higher moments
#     function progressive_objective(λ, phase)
#         try
#             calculated = compute_moments(λ)
            
#             # Different weighting schemes for different optimization phases
#             if phase == 1
#                 # Phase 1: Focus primarily on mean and variance
#                 weights = [5.0, 1.0, 0.01, 0.001]
#             elseif phase == 2
#                 # Phase 2: Add more weight to skewness
#                 weights = [2.0, 2.0, 1.0, 0.01]
#             else
#                 # Phase 3: Balance all moments
#                 weights = [1.0, 1.0, 1.0, 1.0]
#             end
            
#             # Calculate weighted errors
#             errors = zeros(4)
#             for i in 1:4
#                 if abs(moments[i]) < 1e-10
#                     # Use absolute error for near-zero moments
#                     errors[i] = weights[i] * (calculated[i] - moments[i])^2
#                 else
#                     # Use relative error otherwise
#                     errors[i] = weights[i] * ((calculated[i] - moments[i])/moments[i])^2
#                 end
#             end
            
#             return sum(errors)
#         catch e
#             @warn "Objective function error: $e"
#             return 1e10
#         end
#     end
    
#     # Generate more intelligent starting points based on statistical relationships
#     function generate_specialized_starting_points()
#         points = []
        
#         # 1. Standard normal-like distribution
#         push!(points, [0.0, -μ/(σ^2), -1/(2*σ^2), 0.0, 0.0])
        
#         # 2. Add variants with different scalings of the quadratic term
#         push!(points, [0.0, -μ/(σ^2), -1/(4*σ^2), 0.0, 0.0])
#         push!(points, [0.0, -μ/(σ^2), -1/(1*σ^2), 0.0, 0.0])
        
#         # 3. Add skewness through cubic term
#         # Gamma-like for positive skewness
#         if γ₁ > 0
#             push!(points, [0.0, -μ/(σ^2), -1/(2*σ^2), -abs(γ₁)/(6*σ), 0.0])
#             push!(points, [0.0, -μ/(σ^2), -1/(2*σ^2), -abs(γ₁)/(3*σ), 0.0])
#         else
#             push!(points, [0.0, -μ/(σ^2), -1/(2*σ^2), abs(γ₁)/(6*σ), 0.0])
#             push!(points, [0.0, -μ/(σ^2), -1/(2*σ^2), abs(γ₁)/(3*σ), 0.0])
#         end
        
#         # 4. Add kurtosis through quartic term
#         # Use both positive and negative values for 4th-order term
#         push!(points, [0.0, -μ/(σ^2), -1/(2*σ^2), 0.0, γ₂/(24*σ^2)])
#         push!(points, [0.0, -μ/(σ^2), -1/(2*σ^2), 0.0, -γ₂/(24*σ^2)])
        
#         # 5. Combined skewness and kurtosis
#         push!(points, [0.0, -μ/(σ^2), -1/(2*σ^2), -γ₁/(6*σ), γ₂/(24*σ^2)])
        
#         # 6. Try some more extreme values for challenging distributions
#         push!(points, [0.0, -2*μ/(σ^2), -1/(4*σ^2), -γ₁/σ, γ₂/(10*σ^2)])
#         push!(points, [0.0, -μ/(2*σ^2), -1/(8*σ^2), -γ₁/(12*σ), γ₂/(48*σ^2)])
        
#         # 7. Try flatter distribution
#         push!(points, [0.0, 0.0, -0.05, 0.0, -0.005])
        
#         return points
#     end
    
#     # Multi-stage optimization strategy
#     best_λ = zeros(5)
#     best_error = Inf
    
#     starting_points = generate_specialized_starting_points()
#     println("Generated $(length(starting_points)) starting points")
    
#     # Phase 1: Focus on matching lower moments first
#     println("Phase 1: Optimizing lower moments...")
#     for (i, λ_start) in enumerate(starting_points)
#         try
#             result = optimize(
#                 λ -> progressive_objective(λ, 1), 
#                 λ_start[2:end], 
#                 NelderMead(),
#                 Optim.Options(iterations=max_iter÷3, g_tol=1e-8)
#             )
            
#             current_error = result.minimum
#             current_λ = result.minimizer
            
#             if current_error < min(best_error, 10.0)  # Be more permissive in phase 1
#                 best_error = current_error
#                 best_λ[2:end] .= current_λ
#                 println("  Improved solution from start point $i: error = $best_error")
#             end
#         catch e
#             println("  Start point $i failed in phase 1: $e")
#         end
#     end
    
#     # Phase 2: Add emphasis on skewness
#     println("Phase 2: Adding emphasis on skewness...")
#     phase2_start = best_λ[2:end]
#     try
#         result = optimize(
#             λ -> progressive_objective(λ, 2), 
#             phase2_start, 
#             NelderMead(),
#             Optim.Options(iterations=max_iter÷3, g_tol=1e-9)
#         )
        
#         if result.minimum < best_error
#             best_error = result.minimum
#             best_λ[2:end] .= result.minimizer
#             println("  Improved in phase 2: error = $best_error")
#         end
#     catch e
#         println("  Phase 2 optimization failed: $e")
#     end
    
#     # Phase 3: Final balancing of all moments
#     println("Phase 3: Final balancing of all moments...")
#     phase3_start = best_λ[2:end]
#     try
#         result = optimize(
#             λ -> progressive_objective(λ, 3), 
#             phase3_start, 
#             LBFGS(),  # Use gradient-based method for final refinement
#             Optim.Options(iterations=max_iter÷3, g_tol=1e-10)
#         )
        
#         if result.minimum < best_error
#             best_error = result.minimum
#             best_λ[2:end] .= result.minimizer
#             println("  Improved in phase 3: error = $best_error")
#         end
#     catch e
#         println("  Phase 3 optimization failed: $e")
#         # Try with NelderMead as backup
#         try
#             result = optimize(
#                 λ -> progressive_objective(λ, 3), 
#                 phase3_start, 
#                 NelderMead(),
#                 Optim.Options(iterations=max_iter÷3, g_tol=1e-10)
#             )
            
#             if result.minimum < best_error
#                 best_error = result.minimum
#                 best_λ[2:end] .= result.minimizer
#                 println("  Improved in phase 3 backup: error = $best_error")
#             end
#         catch e2
#             println("  Phase 3 backup optimization failed: $e2")
#         end
#     end
    
#     # Calculate λ₀ to ensure normalization
#     z = Z(best_λ[2:end])
#     best_λ[1] = -log(z)
    
#     # Create the final normalized PDF function with safeguards
#     function pdf_function(x)
#         if x < actual_bounds[1] || x > actual_bounds[2]
#             return 0.0  # Zero outside integration bounds
#         end
        
#         exponent = best_λ[1] + best_λ[2]*x + best_λ[3]*x^2 + best_λ[4]*x^3 + best_λ[5]*x^4
        
#         # Cap exponent for numerical stability
#         capped_exponent = max(min(exponent, 500.0), -500.0)
#         return exp(capped_exponent)
#     end
    
#     # Calculate moments achieved by the final distribution
#     achieved_moments = compute_moments(best_λ[2:end])
    
#     println("\nOptimization complete")
#     println("λ values: $best_λ")
#     println("Target moments: $moments")
#     println("Achieved moments: $achieved_moments")
    
#     # Calculate relative errors with protection against division by zero
#     rel_errors = zeros(4)
#     for i in 1:4
#         if abs(moments[i]) < 1e-10
#             rel_errors[i] = abs(achieved_moments[i] - moments[i])
#         else
#             rel_errors[i] = 100.0 * abs(achieved_moments[i] - moments[i]) / abs(moments[i])
#         end
#     end
    
#     println("Relative errors (%): $rel_errors")
    
#     # Create a visualization of the resulting PDF
#     xs = range(actual_bounds[1], actual_bounds[2], length=500)
#     ys = [pdf_function(x) for x in xs]
    
#     return best_λ, achieved_moments, pdf_function
# end


# λ_gen, achieved_moments_gen, pdf_gen = solve_maxent_moments(moments_gen[2:5], bounds=(-10.0, 10.0))

# y_gen = [pdf_gen(x) for x in xax]