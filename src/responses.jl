# """
#     generate_numerical_response_f(model, model_pert, dim, dt, n_tau, n_therm, sigma, Obs, dim_Obs; 
#                                  n_ens=1000, resolution=1, timestepper=:rk4, use_threads=true)

# Generate numerical response by comparing trajectories from perturbed and unperturbed models.
# Optimized with parallel computation and memory-efficient accumulation.

# # Arguments
# - `model`: Original dynamical model function
# - `model_pert`: Perturbed dynamical model function
# - `dim`: Dimension of the state space
# - `dt`: Time step size
# - `n_tau`: Number of time steps for response measurement
# - `n_therm`: Number of steps for thermalization
# - `sigma`: Noise function
# - `Obs`: Observable function to compute on trajectories
# - `dim_Obs`: Dimension of the observable
# - `n_ens`: Number of ensemble members (default: 1000)
# - `resolution`: Save frequency for trajectories (default: 1)
# - `timestepper`: Integration method to use (default: :rk4)
# - `use_threads`: Whether to use thread-based parallelism (default: true)

# # Returns
# - Matrix of response differences with shape (dim_Obs, n_tau)
# """
# function generate_numerical_response_f(model, model_pert, dim, dt, n_tau, n_therm, sigma, Obs, dim_Obs; 
#                                      n_ens=1000, resolution=1, timestepper=:rk4, use_threads=true)

#     # Initialize arrays for accumulating results (avoid large 3D arrays)
#     δObs_sum = zeros(dim_Obs, n_tau)
    
#     # Create a thread-safe progress meter
#     prog = Progress(n_ens, desc="Computing model responses: ", barglyphs=BarGlyphs("[=> ]"))
#     prog_lock = ReentrantLock()
    
#     # Determine computation method: threads or sequential
#     if use_threads && Threads.nthreads() > 1
#         # Parallel processing using threads
#         Threads.@threads for i in 1:n_ens
#             # Thread-local storage for results
#             δObs_local = zeros(dim_Obs, n_tau)
            
#             # Generate unique random seed for this thread
#             thread_seed = abs(rand(Int)) + i * 10000 + Base.Threads.threadid() * 1000000
            
#             # Thermalization: evolve from random initial condition
#             X0 = evolve(randn(dim), dt, n_therm, model, sigma; 
#                        seed=thread_seed, resolution=n_therm, timestepper=timestepper)[:,end] 
            
#             # New seed for response calculation
#             response_seed = thread_seed + 1
            
#             # Evolve both original and perturbed models from same initial condition
#             X = evolve(X0, dt, n_tau*resolution, model, sigma; 
#                       seed=response_seed, resolution=resolution, timestepper=timestepper) 
#             X_pert = evolve(X0, dt, n_tau*resolution, model_pert, sigma; 
#                            seed=response_seed, resolution=resolution, timestepper=timestepper) 
            
#             # Calculate observable differences
#             δObs_local .= Obs(X_pert) .- Obs(X)
            
#             # Atomically accumulate results
#             lock(prog_lock) do
#                 δObs_sum .+= δObs_local
#                 next!(prog)
#             end
#         end
#     else
#         # Sequential processing
#         for i in 1:n_ens
#             # Generate random seed for reproducibility
#             seed = abs(rand(Int)) + i * 10000
            
#             # Thermalization: evolve from random initial condition
#             X0 = evolve(randn(dim), dt, n_therm, model, sigma; 
#                        seed=seed, resolution=n_therm, timestepper=timestepper)[:,end] 
            
#             # New seed for response calculation
#             response_seed = seed + 1
            
#             # Evolve both original and perturbed models from same initial condition
#             X = evolve(X0, dt, n_tau*resolution, model, sigma; 
#                       seed=response_seed, resolution=resolution, timestepper=timestepper) 
#             X_pert = evolve(X0, dt, n_tau*resolution, model_pert, sigma; 
#                            seed=response_seed, resolution=resolution, timestepper=timestepper) 
            
#             # Calculate observable differences and accumulate
#             δObs_sum .+= Obs(X_pert) .- Obs(X)
#             next!(prog)
#         end
#     end
    
#     # Compute mean by dividing by ensemble size
#     δObs = δObs_sum ./ n_ens

#     return δObs
# end

# """
#     generate_numerical_response(model, u, dim, dt, n_tau, n_therm, sigma, Obs, dim_Obs; 
#                                n_ens=1000, resolution=1, timestepper=:rk4, use_threads=true)

# Generate numerical response by applying perturbation to initial condition with single noise source.
# Optimized with parallel computation and memory-efficient accumulation.

# # Arguments
# - `model`: Dynamical model function
# - `u`: Perturbation function
# - `dim`: Dimension of the state space
# - `dt`: Time step size
# - `n_tau`: Number of time steps for response measurement
# - `n_therm`: Number of steps for thermalization
# - `sigma`: Noise function
# - `Obs`: Observable function to compute on trajectories
# - `dim_Obs`: Dimension of the observable
# - `n_ens`: Number of ensemble members (default: 1000)
# - `resolution`: Save frequency for trajectories (default: 1)
# - `timestepper`: Integration method to use (default: :rk4)
# - `use_threads`: Whether to use thread-based parallelism (default: true)

# # Returns
# - Matrix of response differences with shape (dim_Obs, n_tau+1)
# """
# function generate_numerical_response(model, u, dim, dt, n_tau, n_therm, sigma, Obs, dim_Obs; 
#                                    n_ens=1000, resolution=1, timestepper=:rk4, use_threads=true)
#     # Initialize arrays for accumulating results (avoid large 3D arrays)
#     δObs_sum = zeros(dim_Obs, n_tau+1)
    
#     # Create a thread-safe progress meter
#     prog = Progress(n_ens, desc="Computing responses: ", barglyphs=BarGlyphs("[=> ]"))
#     prog_lock = ReentrantLock()
    
#     # Determine computation method: threads or sequential
#     if use_threads && Threads.nthreads() > 1
#         # Parallel processing using threads
#         Threads.@threads for i in 1:n_ens
#             # Thread-local storage for results
#             δObs_local = zeros(dim_Obs, n_tau+1)
            
#             # Generate unique random seed for this thread
#             thread_seed = abs(rand(Int)) + i * 10000 + Base.Threads.threadid() * 1000000
            
#             # Thermalization: evolve from random initial condition
#             X0 = evolve(0.001 .* randn(dim), dt, n_therm, model, sigma; 
#                        seed=thread_seed, resolution=n_therm, timestepper=timestepper)[:,end] 
            
#             # New seed for response calculation
#             response_seed = thread_seed + 1
            
#             # Apply perturbation to initial condition
#             ϵ = u(X0)
#             X0_pert = X0 .+ ϵ
            
#             # Evolve both original and perturbed initial states
#             X = evolve(X0, dt, n_tau*resolution, model, sigma; 
#                       seed=response_seed, resolution=resolution, timestepper=timestepper) 
#             X_pert = evolve(X0_pert, dt, n_tau*resolution, model, sigma; 
#                            seed=response_seed, resolution=resolution, timestepper=timestepper) 
            
#             # Calculate observable differences
#             δObs_local .= Obs(X_pert) .- Obs(X)
            
#             # Atomically accumulate results
#             lock(prog_lock) do
#                 δObs_sum .+= δObs_local
#                 next!(prog)
#             end
#         end
#     else
#         # Sequential processing
#         for i in 1:n_ens
#             # Generate random seed for thermalization
#             seed = abs(rand(Int)) + i * 10000
            
#             # Thermalization: evolve from random initial condition
#             X0 = evolve(0.001 .* randn(dim), dt, n_therm, model, sigma; 
#                        seed=seed, resolution=n_therm, timestepper=timestepper)[:,end] 
            
#             # New seed for response calculation
#             response_seed = seed + 1
            
#             # Apply perturbation to initial condition
#             ϵ = u(X0)
#             X0_pert = X0 .+ ϵ
            
#             # Evolve both original and perturbed initial states
#             X = evolve(X0, dt, n_tau*resolution, model, sigma; 
#                       seed=response_seed, resolution=resolution, timestepper=timestepper) 
#             X_pert = evolve(X0_pert, dt, n_tau*resolution, model, sigma; 
#                            seed=response_seed, resolution=resolution, timestepper=timestepper) 
            
#             # Calculate observable differences and accumulate
#             δObs_sum .+= Obs(X_pert) .- Obs(X)
#             next!(prog)
#         end
#     end
    
#     # Compute mean by dividing by ensemble size
#     δObs = δObs_sum ./ n_ens
#     return δObs
# end

# """
#     generate_numerical_response(model, u, dim, dt, n_tau, n_therm, sigma1, sigma2, Obs, dim_Obs; 
#                                n_ens=1000, resolution=1, timestepper=:rk4, use_threads=true)

# Generate numerical response by applying perturbation to initial condition with two noise sources.
# Optimized with parallel computation and memory-efficient accumulation.

# # Arguments
# - `model`: Dynamical model function
# - `u`: Perturbation function
# - `dim`: Dimension of the state space
# - `dt`: Time step size
# - `n_tau`: Number of time steps for response measurement
# - `n_therm`: Number of steps for thermalization
# - `sigma1`: First noise function
# - `sigma2`: Second noise function
# - `Obs`: Observable function to compute on trajectories
# - `dim_Obs`: Dimension of the observable
# - `n_ens`: Number of ensemble members (default: 1000)
# - `resolution`: Save frequency for trajectories (default: 1)
# - `timestepper`: Integration method to use (default: :rk4)
# - `use_threads`: Whether to use thread-based parallelism (default: true)

# # Returns
# - Matrix of response differences with shape (dim_Obs, n_tau+1)
# """
# function generate_numerical_response(model, u, dim, dt, n_tau, n_therm, sigma1, sigma2, Obs, dim_Obs; 
#                                    n_ens=1000, resolution=1, timestepper=:rk4, use_threads=true)
#     # Initialize arrays for accumulating results (avoid large 3D arrays)
#     δObs_sum = zeros(dim_Obs, n_tau+1)
    
#     # Create a thread-safe progress meter
#     prog = Progress(n_ens, desc="Computing dual-noise responses: ", barglyphs=BarGlyphs("[=> ]"))
#     prog_lock = ReentrantLock()
    
#     # Determine computation method: threads or sequential
#     if use_threads && Threads.nthreads() > 1
#         # Parallel processing using threads
#         Threads.@threads for i in 1:n_ens
#             # Thread-local storage for results
#             δObs_local = zeros(dim_Obs, n_tau+1)
            
#             # Generate unique random seed for this thread
#             thread_seed = abs(rand(Int)) + i * 10000 + Base.Threads.threadid() * 1000000
            
#             # Thermalization: evolve from random initial condition with two noise sources
#             X0 = evolve(randn(dim), dt, n_therm, model, sigma1, sigma2; 
#                        seed=thread_seed, resolution=n_therm, timestepper=timestepper)[:,end] 
            
#             # New seed for response calculation
#             response_seed = thread_seed + 1
            
#             # Apply perturbation to initial condition
#             ϵ = u(X0)
#             X0_pert = X0 .+ ϵ
            
#             # Evolve both original and perturbed initial states with two noise sources
#             X = evolve(X0, dt, n_tau*resolution, model, sigma1, sigma2; 
#                       seed=response_seed, resolution=resolution, timestepper=timestepper) 
#             X_pert = evolve(X0_pert, dt, n_tau*resolution, model, sigma1, sigma2; 
#                            seed=response_seed, resolution=resolution, timestepper=timestepper) 
            
#             # Calculate observable differences
#             δObs_local .= Obs(X_pert) .- Obs(X)
            
#             # Atomically accumulate results
#             lock(prog_lock) do
#                 δObs_sum .+= δObs_local
#                 next!(prog)
#             end
#         end
#     else
#         # Sequential processing
#         for i in 1:n_ens
#             # Generate random seed for thermalization
#             seed = abs(rand(Int)) + i * 10000
            
#             # Thermalization: evolve from random initial condition with two noise sources
#             X0 = evolve(randn(dim), dt, n_therm, model, sigma1, sigma2; 
#                        seed=seed, resolution=n_therm, timestepper=timestepper)[:,end] 
            
#             # New seed for response calculation
#             response_seed = seed + 1
            
#             # Apply perturbation to initial condition
#             ϵ = u(X0)
#             X0_pert = X0 .+ ϵ
            
#             # Evolve both original and perturbed initial states with two noise sources
#             X = evolve(X0, dt, n_tau*resolution, model, sigma1, sigma2; 
#                       seed=response_seed, resolution=resolution, timestepper=timestepper) 
#             X_pert = evolve(X0_pert, dt, n_tau*resolution, model, sigma1, sigma2; 
#                            seed=response_seed, resolution=resolution, timestepper=timestepper) 
            
#             # Calculate observable differences and accumulate
#             δObs_sum .+= Obs(X_pert) .- Obs(X)
#             next!(prog)
#         end
#     end
    
#     # Compute mean by dividing by ensemble size
#     δObs = δObs_sum ./ n_ens
#     return δObs
# end

"""
    generate_numerical_response_f(model, model_pert, dim, dt, n_tau, n_therm, sigma, Obs, dim_Obs; n_ens=1000, resolution=1)

Generate numerical response by comparing trajectories from perturbed and unperturbed models.

# Arguments
- `model`: Original dynamical model function
- `model_pert`: Perturbed dynamical model function
- `dim`: Dimension of the state space
- `dt`: Time step size
- `n_tau`: Number of time steps for response measurement
- `n_therm`: Number of steps for thermalization
- `sigma`: Noise function
- `Obs`: Observable function to compute on trajectories
- `dim_Obs`: Dimension of the observable
- `n_ens`: Number of ensemble members (default: 1000)
- `resolution`: Save frequency for trajectories (default: 1)

# Returns
- Matrix of response differences with shape (dim_Obs, n_tau)
"""
function generate_numerical_response_f(model, model_pert, dim, dt, n_tau, n_therm, sigma, Obs, dim_Obs; n_ens=1000, resolution=1)

    # Initialize array to store ensemble results
    δObs_ens = zeros(dim_Obs, n_tau, n_ens)
    
    # Loop over ensemble members
    for i in ProgressBar(1:n_ens)
        # Generate random seed for reproducibility
        seed = abs(rand(Int))
        
        # Thermalization: evolve from random initial condition
        X0 = evolve(randn(dim), dt, n_therm, model, sigma; seed=seed, resolution=n_therm)[:,end] 
        
        # New seed for response calculation
        seed = abs(rand(Int))
        
        # Evolve both original and perturbed models from same initial condition
        X = evolve(X0, dt, n_tau*resolution, model, sigma; seed=seed, resolution=resolution) 
        X_pert = evolve(X0, dt, n_tau*resolution, model_pert, sigma; seed=seed, resolution=resolution) 
        
        # Calculate observable differences
        δObs_ens[:,:,i] = Obs(X_pert) .- Obs(X)
    end
    
    # Average over ensemble members
    δObs = mean(δObs_ens, dims=3)[:,:,1]

    return δObs
end

"""
    generate_numerical_response(model, u, dim, dt, n_tau, n_therm, sigma, Obs, dim_Obs; n_ens=1000, resolution=1)

Generate numerical response by applying perturbation to initial condition with single noise source.

# Arguments
- `model`: Dynamical model function
- `u`: Perturbation function
- `dim`: Dimension of the state space
- `dt`: Time step size
- `n_tau`: Number of time steps for response measurement
- `n_therm`: Number of steps for thermalization
- `sigma`: Noise function
- `Obs`: Observable function to compute on trajectories
- `dim_Obs`: Dimension of the observable
- `n_ens`: Number of ensemble members (default: 1000)
- `resolution`: Save frequency for trajectories (default: 1)

# Returns
- Matrix of response differences with shape (dim_Obs, n_tau+1)
"""
function generate_numerical_response(model, u, dim, dt, n_tau, n_therm, sigma, Obs, dim_Obs; n_ens=1000, resolution=1)
    # Initialize array to store ensemble results
    δObs_ens = zeros(dim_Obs, n_tau+1, n_ens)
    
    # Loop over ensemble members
    for i in ProgressBar(1:n_ens)
        # Generate random seed for thermalization
        seed = abs(rand(Int))
        
        # Thermalization: evolve from random initial condition
        X0 = evolve(0.001 .* randn(dim), dt, n_therm, model, sigma; seed=seed, resolution=n_therm)[:,end] 
        
        # New seed for response calculation
        seed = abs(rand(Int))
        
        # Apply perturbation to initial condition
        ϵ = u(X0)
        X0_pert = X0 .+ ϵ
        
        # Evolve both original and perturbed initial states
        X = evolve(X0, dt, n_tau*resolution, model, sigma; seed=seed, resolution=resolution) 
        X_pert = evolve(X0_pert, dt, n_tau*resolution, model, sigma; seed=seed, resolution=resolution) 
        
        # Calculate observable differences
        δObs_ens[:,:,i] = Obs(X_pert) .- Obs(X)
    end
    
    # Average over ensemble members
    δObs = mean(δObs_ens, dims=3)[:,:,1]
    return δObs
end

"""
    generate_numerical_response(model, u, dim, dt, n_tau, n_therm, sigma1, sigma2, Obs, dim_Obs; n_ens=1000, resolution=1)

Generate numerical response by applying perturbation to initial condition with two noise sources.

# Arguments
- `model`: Dynamical model function
- `u`: Perturbation function
- `dim`: Dimension of the state space
- `dt`: Time step size
- `n_tau`: Number of time steps for response measurement
- `n_therm`: Number of steps for thermalization
- `sigma1`: First noise function
- `sigma2`: Second noise function
- `Obs`: Observable function to compute on trajectories
- `dim_Obs`: Dimension of the observable
- `n_ens`: Number of ensemble members (default: 1000)
- `resolution`: Save frequency for trajectories (default: 1)

# Returns
- Matrix of response differences with shape (dim_Obs, n_tau+1)
"""
function generate_numerical_response(model, u, dim, dt, n_tau, n_therm, sigma1, sigma2, Obs, dim_Obs; n_ens=1000, resolution=1)
    # Initialize array to store ensemble results
    δObs_ens = zeros(dim_Obs, n_tau+1, n_ens)
    
    # Loop over ensemble members
    for i in ProgressBar(1:n_ens)
        # Generate random seed for thermalization
        seed = abs(rand(Int))
        
        # Thermalization: evolve from random initial condition with two noise sources
        X0 = evolve(randn(dim), dt, n_therm, model, sigma1, sigma2; seed=seed, resolution=n_therm)[:,end] 
        
        # New seed for response calculation
        seed = abs(rand(Int))
        
        # Apply perturbation to initial condition
        ϵ = u(X0)
        X0_pert = X0 .+ ϵ
        
        # Evolve both original and perturbed initial states with two noise sources
        X = evolve(X0, dt, n_tau*resolution, model, sigma1, sigma2; seed=seed, resolution=resolution) 
        X_pert = evolve(X0_pert, dt, n_tau*resolution, model, sigma1, sigma2; seed=seed, resolution=resolution) 
        
        # Calculate observable differences
        δObs_ens[:,:,i] = Obs(X_pert) .- Obs(X)
    end
    
    # Average over ensemble members
    δObs = mean(δObs_ens, dims=3)[:,:,1]
    return δObs
end

"""
    generate_score_response(trj, u, div_u, f, score, dt, n_tau, Obs, dim_Obs)

Generate response functions using the score-based approach from GFDT with parallel processing.

# Arguments
- `trj`: Trajectory data with shape (dim, steps)
- `u`: Perturbation function
- `div_u`: Divergence of the perturbation function
- `f`: Time-dependent forcing function
- `score`: Score function (gradient of log probability)
- `dt`: Time step size
- `n_tau`: Number of time steps for response
- `Obs`: Observable function
- `dim_Obs`: Dimension of the observable

# Returns
- Tuple containing:
  - R: Response function with shape (dim_Obs, n_tau+1)
  - δObs: Predicted response with shape (dim_Obs, n_tau+1)
"""
function generate_score_response(trj, u, div_u, f, score, dt, n_tau, Obs, dim_Obs)
    # Get dimensions of trajectory data
    dim_trj, steps_trj = size(trj)
    
    # Define conjugate variable B(x) according to GFDT
    B(x) = - div_u(x) .- (u(x)' * score(x))[1]
    
    # Pre-compute B and Obs values for the entire trajectory to avoid redundant calculations
    # This reduces computation without increasing memory usage significantly
    B_values = zeros(steps_trj)
    Obs_values = Array{Float64}(undef, dim_Obs, steps_trj)
    
    # Pre-compute in parallel
    Threads.@threads for j in 1:steps_trj
        B_values[j] = B(trj[:,j])
        Obs_values[:,j] = Obs(trj[:,j])
    end
    
    # Initialize arrays for response functions and responses
    R = zeros(dim_Obs, n_tau+1)
    δObs = zeros(dim_Obs, n_tau+1)
    
    # Create a thread-safe progress meter
    prog = Progress(n_tau+1, desc="Computing response: ", barglyphs=BarGlyphs("[=> ]"))
    prog_lock = ReentrantLock()
    
    # Phase 1: Calculate R values in parallel
    # Each lag i can be computed independently
    Threads.@threads for i in 1:n_tau+1
        # Thread-local accumulator for R
        R_local = zeros(dim_Obs)
        
        # Calculate correlation for this lag
        available_points = steps_trj - i + 1
        for j in 1:available_points
            R_local .+= B_values[j] .* Obs_values[:,j+i-1]
        end
        
        # Normalize by number of points and store in the global R array
        if available_points > 0
            R[:,i] = R_local ./ available_points
        end
        
        # Update progress
        lock(prog_lock) do
            next!(prog)
        end
    end
    
    # Phase 2: Compute δObs using the pre-computed R
    # This part has dependencies and must be done sequentially
    for i in 2:n_tau+1
        for j in 1:i-1
            δObs[:, i] .+= R[:, i - j + 1] * f(j * dt) * dt
        end
    end

    return R, δObs
end

"""
    generate_numerical_response_HO(model, u, dim, dt, n_tau, n_therm, sigma, M; n_ens=1000, resolution=1)

Generate numerical response functions for higher-order moments by comparing trajectories from perturbed and unperturbed initial conditions,
using multi-threading for parallel computation.

# Arguments
- `model`: Dynamical model function
- `u`: Perturbation function
- `dim`: Dimension of the state space
- `dt`: Time step size
- `n_tau`: Number of time steps for response measurement
- `n_therm`: Number of steps for thermalization
- `sigma`: Noise function
- `M`: Reference point for computing centered moments
- `n_ens`: Number of ensemble members (default: 1000)
- `resolution`: Save frequency for trajectories (default: 1)

# Returns
- Tuple of four matrices (δObs_1, δObs_2, δObs_3, δObs_4), each with shape (dim, n_tau+1):
  - δObs_1: First moment response (direct difference)
  - δObs_2: Second moment response (difference in squared deviations)
  - δObs_3: Third moment response (difference in cubed deviations)
  - δObs_4: Fourth moment response (difference in fourth power deviations)

# Note
This function computes higher-order statistics by centering around reference point M,
allowing the analysis of non-Gaussian responses to perturbations in dynamical systems.
"""
function generate_numerical_response_HO(model, u, dim, dt, n_tau, n_therm, sigma, M; n_ens=1000, resolution=1, timestepper=timestepper)
    # Initialize arrays for the final results (avoid using large 3D arrays)
    δObs_sum_1 = zeros(dim, n_tau+1)
    δObs_sum_2 = zeros(dim, n_tau+1)
    δObs_sum_3 = zeros(dim, n_tau+1)
    δObs_sum_4 = zeros(dim, n_tau+1)
    
    # Create a thread-safe progress meter
    prog = Progress(n_ens, desc="Computing ensemble responses: ", barglyphs=BarGlyphs("[=> ]"))
    prog_lock = ReentrantLock()
    
    # Atomic counter for thread-safe progress updates
    counter = Threads.Atomic{Int}(0)
    
    # Parallel loop over ensemble members
    Threads.@threads for i in 1:n_ens
        # Thread-local storage for differences
        δObs_local_1 = zeros(dim, n_tau+1)
        δObs_local_2 = zeros(dim, n_tau+1)
        δObs_local_3 = zeros(dim, n_tau+1)
        δObs_local_4 = zeros(dim, n_tau+1)
        
        # Generate unique random seed for this thread
        thread_seed = abs(rand(Int)) + i * 10000 + Base.Threads.threadid() * 1000000
        
        # Thermalization: evolve from random initial condition
        X0 = evolve(0.001 .* randn(dim), dt, n_therm, model, sigma; 
                    seed=thread_seed, resolution=n_therm, timestepper=timestepper)[:,end] 
        
        # New seed for response calculation
        response_seed = thread_seed + 1
        
        # Apply perturbation to initial condition
        ϵ = u(X0)
        X0_pert = X0 .+ ϵ
        
        # Evolve both original and perturbed initial states
        X = evolve(X0, dt, n_tau*resolution, model, sigma; 
                   seed=response_seed, resolution=resolution, timestepper=timestepper) 
        X_pert = evolve(X0_pert, dt, n_tau*resolution, model, sigma; 
                        seed=response_seed, resolution=resolution, timestepper=timestepper) 
        
        # Calculate observable differences and store in thread-local arrays
        δObs_local_1 .= X_pert .- X
        δObs_local_2 .= (X_pert .- M) .^ 2 .- (X .- M) .^ 2
        δObs_local_3 .= (X_pert .- M) .^ 3 .- (X .- M) .^ 3
        δObs_local_4 .= (X_pert .- M) .^ 4 .- (X .- M) .^ 4
        
        # Atomically accumulate results into the shared arrays
        lock(prog_lock) do
            δObs_sum_1 .+= δObs_local_1
            δObs_sum_2 .+= δObs_local_2
            δObs_sum_3 .+= δObs_local_3
            δObs_sum_4 .+= δObs_local_4
            next!(prog)
        end
    end
    
    # Compute mean by dividing by ensemble size
    δObs_1 = δObs_sum_1 ./ n_ens
    δObs_2 = δObs_sum_2 ./ n_ens
    δObs_3 = δObs_sum_3 ./ n_ens
    δObs_4 = δObs_sum_4 ./ n_ens

    return δObs_1, δObs_2, δObs_3, δObs_4
end

"""
    generate_numerical_response_HO(model, u, dim, dt, n_tau, n_therm, sigma1, sigma2, M; n_ens=1000, resolution=1)

Generate numerical response functions for higher-order moments by comparing trajectories from perturbed and unperturbed initial conditions,
using two separate noise sources and multi-threading for parallel computation.

# Arguments
- `model`: Dynamical model function
- `u`: Perturbation function
- `dim`: Dimension of the state space
- `dt`: Time step size
- `n_tau`: Number of time steps for response measurement
- `n_therm`: Number of steps for thermalization
- `sigma1`: First noise function
- `sigma2`: Second noise function
- `M`: Reference point for computing centered moments
- `n_ens`: Number of ensemble members (default: 1000)
- `resolution`: Save frequency for trajectories (default: 1)

# Returns
- Tuple of four matrices (δObs_1, δObs_2, δObs_3, δObs_4), each with shape (dim, n_tau+1):
  - δObs_1: First moment response (direct difference)
  - δObs_2: Second moment response (difference in squared deviations)
  - δObs_3: Third moment response (difference in cubed deviations)
  - δObs_4: Fourth moment response (difference in fourth power deviations)
"""
function generate_numerical_response_HO(model, u, dim, dt, n_tau, n_therm, sigma1, sigma2, M; n_ens=1000, resolution=1, timestepper=timestepper)
    # Initialize arrays for the final results (avoid using large 3D arrays)
    δObs_sum_1 = zeros(dim, n_tau+1)
    δObs_sum_2 = zeros(dim, n_tau+1)
    δObs_sum_3 = zeros(dim, n_tau+1)
    δObs_sum_4 = zeros(dim, n_tau+1)
    
    # Create a thread-safe progress meter
    prog = Progress(n_ens, desc="Computing ensemble responses: ", barglyphs=BarGlyphs("[=> ]"))
    prog_lock = ReentrantLock()
    
    # Parallel loop over ensemble members
    Threads.@threads for i in 1:n_ens
        # Thread-local storage for differences
        δObs_local_1 = zeros(dim, n_tau+1)
        δObs_local_2 = zeros(dim, n_tau+1)
        δObs_local_3 = zeros(dim, n_tau+1)
        δObs_local_4 = zeros(dim, n_tau+1)
        
        # Generate unique random seed for this thread
        # Use Base.Threads.threadid() with full qualification
        thread_seed = abs(rand(Int)) + i * 10000 + Base.Threads.threadid() * 1000000
        
        # Thermalization: evolve from random initial condition
        X0 = evolve(0.001 .* randn(dim), dt, n_therm, model, sigma1, sigma2; 
                    seed=thread_seed, resolution=n_therm, timestepper=timestepper)[:,end] 
        
        # New seed for response calculation
        response_seed = thread_seed + 1
        
        # Apply perturbation to initial condition
        ϵ = u(X0)
        X0_pert = X0 .+ ϵ
        
        # Evolve both original and perturbed initial states
        X = evolve(X0, dt, n_tau*resolution, model, sigma1, sigma2; 
                   seed=response_seed, resolution=resolution, timestepper=timestepper) 
        X_pert = evolve(X0_pert, dt, n_tau*resolution, model, sigma1, sigma2; 
                        seed=response_seed, resolution=resolution, timestepper=timestepper) 
        
        # Calculate observable differences and store in thread-local arrays
        δObs_local_1 .= X_pert .- X
        δObs_local_2 .= (X_pert .- M) .^ 2 .- (X .- M) .^ 2
        δObs_local_3 .= (X_pert .- M) .^ 3 .- (X .- M) .^ 3
        δObs_local_4 .= (X_pert .- M) .^ 4 .- (X .- M) .^ 4
        
        # Atomically accumulate results into the shared arrays
        lock(prog_lock) do
            δObs_sum_1 .+= δObs_local_1
            δObs_sum_2 .+= δObs_local_2
            δObs_sum_3 .+= δObs_local_3
            δObs_sum_4 .+= δObs_local_4
            next!(prog)
        end
    end
    
    # Compute mean by dividing by ensemble size
    δObs_1 = δObs_sum_1 ./ n_ens
    δObs_2 = δObs_sum_2 ./ n_ens
    δObs_3 = δObs_sum_3 ./ n_ens
    δObs_4 = δObs_sum_4 ./ n_ens

    return δObs_1, δObs_2, δObs_3, δObs_4
end


"""
    generate_numerical_response_f_HO(model, model_pert, dim, dt, n_tau, n_therm, sigma, M; 
                                    n_ens=1000, resolution=1, timestepper=:rk4, use_threads=true)

Generate numerical response functions for higher-order moments by comparing trajectories 
between the original and perturbed models, with highly efficient parallel computation.

# Arguments
- `model`: Original dynamical model function
- `model_pert`: Perturbed dynamical model function
- `dim`: Dimension of the state space
- `dt`: Time step size
- `n_tau`: Number of time steps for response measurement
- `n_therm`: Number of steps for thermalization
- `sigma`: Noise function
- `M`: Reference point for computing centered moments
- `n_ens`: Number of ensemble members (default: 1000)
- `resolution`: Save frequency for trajectories (default: 1)
- `timestepper`: Integration method to use (default: :rk4)
- `use_threads`: Whether to use thread-based parallelism (default: true)

# Returns
- Tuple of four matrices (δObs_1, δObs_2, δObs_3, δObs_4), each with shape (dim, n_tau+1):
  - δObs_1: First moment response (direct difference)
  - δObs_2: Second moment response (difference in squared deviations)
  - δObs_3: Third moment response (difference in cubed deviations)
  - δObs_4: Fourth moment response (difference in fourth power deviations)
"""
function generate_numerical_response_f_HO(model, model_pert, dim, dt, n_tau, n_therm, sigma, M; 
                                         n_ens=1000, resolution=1, timestepper=:rk4, use_threads=true)
    # Initialize arrays for accumulating results
    δObs_sum_1 = zeros(dim, n_tau+1)
    δObs_sum_2 = zeros(dim, n_tau+1)
    δObs_sum_3 = zeros(dim, n_tau+1)
    δObs_sum_4 = zeros(dim, n_tau+1)
    
    # Create a thread-safe progress meter
    prog = Progress(n_ens, desc="Computing model responses: ", barglyphs=BarGlyphs("[=> ]"))
    prog_lock = ReentrantLock()
    
    # Determine computation method: threads or sequential
    if use_threads && Threads.nthreads() > 1
        # Parallel processing using threads
        Threads.@threads for i in 1:n_ens
            # Thread-local storage for differences
            δObs_local_1 = zeros(dim, n_tau+1)
            δObs_local_2 = zeros(dim, n_tau+1)
            δObs_local_3 = zeros(dim, n_tau+1)
            δObs_local_4 = zeros(dim, n_tau+1)
            
            # Generate unique random seed for this thread
            thread_seed = abs(rand(Int)) + i * 10000 + Base.Threads.threadid() * 1000000
            
            # Compute trajectory for this ensemble member
            process_ensemble_member!(
                δObs_local_1, δObs_local_2, δObs_local_3, δObs_local_4,
                dim, dt, n_tau, n_therm, model, model_pert, sigma, M,
                thread_seed, resolution, timestepper
            )
            
            # Atomically accumulate results
            lock(prog_lock) do
                δObs_sum_1 .+= δObs_local_1
                δObs_sum_2 .+= δObs_local_2
                δObs_sum_3 .+= δObs_local_3
                δObs_sum_4 .+= δObs_local_4
                next!(prog)
            end
        end
    else
        # Sequential processing
        for i in 1:n_ens
            # Local storage for differences
            δObs_local_1 = zeros(dim, n_tau+1)
            δObs_local_2 = zeros(dim, n_tau+1)
            δObs_local_3 = zeros(dim, n_tau+1)
            δObs_local_4 = zeros(dim, n_tau+1)
            
            # Generate random seed
            seed = abs(rand(Int)) + i * 10000
            
            # Compute trajectory
            process_ensemble_member!(
                δObs_local_1, δObs_local_2, δObs_local_3, δObs_local_4,
                dim, dt, n_tau, n_therm, model, model_pert, sigma, M,
                seed, resolution, timestepper
            )
            
            # Accumulate results
            δObs_sum_1 .+= δObs_local_1
            δObs_sum_2 .+= δObs_local_2
            δObs_sum_3 .+= δObs_local_3
            δObs_sum_4 .+= δObs_local_4
            next!(prog)
        end
    end
    
    # Compute mean by dividing by ensemble size
    δObs_1 = δObs_sum_1 ./ n_ens
    δObs_2 = δObs_sum_2 ./ n_ens
    δObs_3 = δObs_sum_3 ./ n_ens
    δObs_4 = δObs_sum_4 ./ n_ens

    return δObs_1, δObs_2, δObs_3, δObs_4
end

"""
Helper function to process a single ensemble member for the response calculation.
Extracts the core simulation logic to avoid code duplication.
"""
function process_ensemble_member!(
    δObs_local_1, δObs_local_2, δObs_local_3, δObs_local_4,
    dim, dt, n_tau, n_therm, model, model_pert, sigma, M,
    seed, resolution, timestepper
)
    # Thermalization: evolve from random initial condition
    X0 = evolve(randn(dim), dt, n_therm, model, sigma; 
               seed=seed, resolution=n_therm, timestepper=timestepper)[:,end]
    
    # New seed for response calculation
    response_seed = seed + 1
    
    # Evolve both original and perturbed models from same initial condition
    # Use batch evolution if available to avoid duplicate calculations
    if n_tau*resolution <= 1000  # For small trajectories, evolve separately
        X = evolve(X0, dt, n_tau*resolution, model, sigma; 
                  seed=response_seed, resolution=resolution, timestepper=timestepper)
        X_pert = evolve(X0, dt, n_tau*resolution, model_pert, sigma; 
                        seed=response_seed, resolution=resolution, timestepper=timestepper)
    else
        # For longer trajectories, use memory-efficient approach
        # This processes the trajectories in batches to avoid excessive memory usage
        batch_size = min(1000, n_tau*resolution)
        num_batches = ceil(Int, n_tau*resolution / batch_size)
        
        X = zeros(dim, n_tau+1)
        X_pert = zeros(dim, n_tau+1)
        X[:, 1] .= X0
        X_pert[:, 1] .= X0
        
        current_state = copy(X0)
        current_state_pert = copy(X0)
        
        save_idx = 1
        for batch in 1:num_batches
            steps_in_batch = min(batch_size, n_tau*resolution - (batch-1)*batch_size)
            
            # Evolve original model for this batch
            batch_result = evolve(current_state, dt, steps_in_batch, model, sigma; 
                                 seed=response_seed + batch, resolution=resolution, 
                                 timestepper=timestepper)
            
            # Evolve perturbed model for this batch
            batch_result_pert = evolve(current_state_pert, dt, steps_in_batch, model_pert, sigma; 
                                      seed=response_seed + batch, resolution=resolution, 
                                      timestepper=timestepper)
            
            # Save results at resolution points
            num_save_points = ceil(Int, steps_in_batch / resolution)
            for j in 1:num_save_points
                if save_idx + j <= n_tau+1
                    X[:, save_idx + j] .= batch_result[:, j+1]
                    X_pert[:, save_idx + j] .= batch_result_pert[:, j+1]
                end
            end
            
            # Update save index and current states for next batch
            save_idx += num_save_points
            current_state .= batch_result[:, end]
            current_state_pert .= batch_result_pert[:, end]
            
            # Early termination if we've filled the arrays
            if save_idx > n_tau+1
                break
            end
        end
    end
    
    # Calculate observable differences using pre-allocated arrays for efficiency
    for t in 1:n_tau+1
        for d in 1:dim
            # Direct differences (first moment)
            δObs_local_1[d, t] = X_pert[d, t] - X[d, t]
            
            # Differences in squared deviations (second moment)
            δObs_local_2[d, t] = (X_pert[d, t] - M[d])^2 - (X[d, t] - M[d])^2
            
            # Differences in cubed deviations (third moment)
            δObs_local_3[d, t] = (X_pert[d, t] - M[d])^3 - (X[d, t] - M[d])^3
            
            # Differences in fourth power deviations (fourth moment)
            δObs_local_4[d, t] = (X_pert[d, t] - M[d])^4 - (X[d, t] - M[d])^4
        end
    end
    
    return nothing
end

"""
    generate_numerical_response_f_HO(model, model_pert, dim, dt, n_tau, n_therm, sigma, M; 
                                    n_ens=1000, resolution=1, timestepper=:rk4, use_threads=true)

Generate numerical response functions for higher-order moments by comparing trajectories 
between the original and perturbed models, with highly efficient parallel computation.

# Arguments
- `model`: Original dynamical model function
- `model_pert`: Perturbed dynamical model function
- `dim`: Dimension of the state space
- `dt`: Time step size
- `n_tau`: Number of time steps for response measurement
- `n_therm`: Number of steps for thermalization
- `sigma`: Noise function
- `M`: Reference point for computing centered moments
- `n_ens`: Number of ensemble members (default: 1000)
- `resolution`: Save frequency for trajectories (default: 1)
- `timestepper`: Integration method to use (default: :rk4)
- `use_threads`: Whether to use thread-based parallelism (default: true)

# Returns
- Tuple of four matrices (δObs_1, δObs_2, δObs_3, δObs_4), each with shape (dim, n_tau+1):
  - δObs_1: First moment response (direct difference)
  - δObs_2: Second moment response (difference in squared deviations)
  - δObs_3: Third moment response (difference in cubed deviations)
  - δObs_4: Fourth moment response (difference in fourth power deviations)
"""
function generate_numerical_response_f_HO(model, model_pert, dim, dt, n_tau, n_therm, sigma1, sigma2, M; 
                                         n_ens=1000, resolution=1, timestepper=:rk4, use_threads=true)
    # Initialize arrays for accumulating results
    δObs_sum_1 = zeros(dim, n_tau+1)
    δObs_sum_2 = zeros(dim, n_tau+1)
    δObs_sum_3 = zeros(dim, n_tau+1)
    δObs_sum_4 = zeros(dim, n_tau+1)
    
    # Create a thread-safe progress meter
    prog = Progress(n_ens, desc="Computing model responses: ", barglyphs=BarGlyphs("[=> ]"))
    prog_lock = ReentrantLock()
    
    # Determine computation method: threads or sequential
    if use_threads && Threads.nthreads() > 1
        # Parallel processing using threads
        Threads.@threads for i in 1:n_ens
            # Thread-local storage for differences
            δObs_local_1 = zeros(dim, n_tau+1)
            δObs_local_2 = zeros(dim, n_tau+1)
            δObs_local_3 = zeros(dim, n_tau+1)
            δObs_local_4 = zeros(dim, n_tau+1)
            
            # Generate unique random seed for this thread
            thread_seed = abs(rand(Int)) + i * 10000 + Base.Threads.threadid() * 1000000
            
            # Compute trajectory for this ensemble member
            process_ensemble_member!(
                δObs_local_1, δObs_local_2, δObs_local_3, δObs_local_4,
                dim, dt, n_tau, n_therm, model, model_pert, sigma1, sigma2, M,
                thread_seed, resolution, timestepper
            )
            
            # Atomically accumulate results
            lock(prog_lock) do
                δObs_sum_1 .+= δObs_local_1
                δObs_sum_2 .+= δObs_local_2
                δObs_sum_3 .+= δObs_local_3
                δObs_sum_4 .+= δObs_local_4
                next!(prog)
            end
        end
    else
        # Sequential processing
        for i in 1:n_ens
            # Local storage for differences
            δObs_local_1 = zeros(dim, n_tau+1)
            δObs_local_2 = zeros(dim, n_tau+1)
            δObs_local_3 = zeros(dim, n_tau+1)
            δObs_local_4 = zeros(dim, n_tau+1)
            
            # Generate random seed
            seed = abs(rand(Int)) + i * 10000
            
            # Compute trajectory
            process_ensemble_member!(
                δObs_local_1, δObs_local_2, δObs_local_3, δObs_local_4,
                dim, dt, n_tau, n_therm, model, model_pert, sigma1, sigma2, M,
                seed, resolution, timestepper
            )
            
            # Accumulate results
            δObs_sum_1 .+= δObs_local_1
            δObs_sum_2 .+= δObs_local_2
            δObs_sum_3 .+= δObs_local_3
            δObs_sum_4 .+= δObs_local_4
            next!(prog)
        end
    end
    
    # Compute mean by dividing by ensemble size
    δObs_1 = δObs_sum_1 ./ n_ens
    δObs_2 = δObs_sum_2 ./ n_ens
    δObs_3 = δObs_sum_3 ./ n_ens
    δObs_4 = δObs_sum_4 ./ n_ens

    return δObs_1, δObs_2, δObs_3, δObs_4
end

"""
Helper function to process a single ensemble member for the response calculation.
Extracts the core simulation logic to avoid code duplication.
"""
function process_ensemble_member!(
    δObs_local_1, δObs_local_2, δObs_local_3, δObs_local_4,
    dim, dt, n_tau, n_therm, model, model_pert, sigma1, sigma2, M,
    seed, resolution, timestepper
)
    # Thermalization: evolve from random initial condition
    X0 = evolve(randn(dim), dt, n_therm, model, sigma1, sigma2; 
               seed=seed, resolution=n_therm, timestepper=timestepper)[:,end]
    
    # New seed for response calculation
    response_seed = seed + 1
    
    # Evolve both original and perturbed models from same initial condition
    # Use batch evolution if available to avoid duplicate calculations
    if n_tau*resolution <= 1000  # For small trajectories, evolve separately
        X = evolve(X0, dt, n_tau*resolution, model, sigma1, sigma2; 
                  seed=response_seed, resolution=resolution, timestepper=timestepper)
        X_pert = evolve(X0, dt, n_tau*resolution, model_pert, sigma1, sigma2; 
                        seed=response_seed, resolution=resolution, timestepper=timestepper)
    else
        # For longer trajectories, use memory-efficient approach
        # This processes the trajectories in batches to avoid excessive memory usage
        batch_size = min(1000, n_tau*resolution)
        num_batches = ceil(Int, n_tau*resolution / batch_size)
        
        X = zeros(dim, n_tau+1)
        X_pert = zeros(dim, n_tau+1)
        X[:, 1] .= X0
        X_pert[:, 1] .= X0
        
        current_state = copy(X0)
        current_state_pert = copy(X0)
        
        save_idx = 1
        for batch in 1:num_batches
            steps_in_batch = min(batch_size, n_tau*resolution - (batch-1)*batch_size)
            
            # Evolve original model for this batch
            batch_result = evolve(current_state, dt, steps_in_batch, model, sigma1, sigma2; 
                                 seed=response_seed + batch, resolution=resolution, 
                                 timestepper=timestepper)
            
            # Evolve perturbed model for this batch
            batch_result_pert = evolve(current_state_pert, dt, steps_in_batch, model_pert, sigma1, sigma2; 
                                      seed=response_seed + batch, resolution=resolution, 
                                      timestepper=timestepper)
            
            # Save results at resolution points
            num_save_points = ceil(Int, steps_in_batch / resolution)
            for j in 1:num_save_points
                if save_idx + j <= n_tau+1
                    X[:, save_idx + j] .= batch_result[:, j+1]
                    X_pert[:, save_idx + j] .= batch_result_pert[:, j+1]
                end
            end
            
            # Update save index and current states for next batch
            save_idx += num_save_points
            current_state .= batch_result[:, end]
            current_state_pert .= batch_result_pert[:, end]
            
            # Early termination if we've filled the arrays
            if save_idx > n_tau+1
                break
            end
        end
    end
    
    # Calculate observable differences using pre-allocated arrays for efficiency
    for t in 1:n_tau+1
        for d in 1:dim
            # Direct differences (first moment)
            δObs_local_1[d, t] = X_pert[d, t] - X[d, t]
            
            # Differences in squared deviations (second moment)
            δObs_local_2[d, t] = (X_pert[d, t] - M[d])^2 - (X[d, t] - M[d])^2
            
            # Differences in cubed deviations (third moment)
            δObs_local_3[d, t] = (X_pert[d, t] - M[d])^3 - (X[d, t] - M[d])^3
            
            # Differences in fourth power deviations (fourth moment)
            δObs_local_4[d, t] = (X_pert[d, t] - M[d])^4 - (X[d, t] - M[d])^4
        end
    end
    
    return nothing
end