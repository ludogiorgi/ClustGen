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
        X0 = evolve(randn(dim), dt, n_therm, model, sigma; seed=seed, resolution=n_therm)[:,end] 
        
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

Generate response functions using the score-based approach from GFDT.

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
    
    # Initialize arrays for response functions and responses
    R = zeros(dim_Obs, n_tau+1)
    δObs = zeros(dim_Obs, n_tau+1)
    
    # Compute response functions for each lag
    for i in ProgressBar(1:n_tau+1)
        # Compute correlation between B and observable at lag i-1
        for j in 1:steps_trj-i+1
            R[:,i] .+= B(trj[:,j]) .* Obs(trj[:,j+i-1])
        end
        
        # Average over available time points
        R[:, i] ./= (steps_trj - i + 1)
        
        # Calculate response by convolving response function with forcing
        if i > 1
            for j in 1:i-1
                δObs[:, i] .+= R[:, i - j + 1] * f(j * dt) * dt
            end
        end
    end

    return R, δObs
end