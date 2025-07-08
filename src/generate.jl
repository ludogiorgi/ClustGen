"""
    rk4_step!(u, dt, f, t)

Performs a single 4th-order Runge-Kutta integration step.

# Arguments
- `u`: State vector, modified in-place
- `dt`: Time step size
- `f`: Function defining the dynamics, should accept (u, t) arguments
- `t`: Current time
"""
function rk4_step!(u, dt, f, t)
    k1 = f(u, t)
    k2 = f(u .+ 0.5 .* dt .* k1, t + 0.5 * dt)
    k3 = f(u .+ 0.5 .* dt .* k2, t + 0.5 * dt)
    k4 = f(u .+ dt .* k3, t + dt)
    @inbounds u .= u .+ (dt / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
end

"""
    euler_step!(u, dt, f, t)

Performs a single Euler integration step.

# Arguments
- `u`: State vector, modified in-place
- `dt`: Time step size
- `f`: Function defining the dynamics, should accept (u, t) arguments
- `t`: Current time
"""
function euler_step!(u, dt, f, t)
    k1 = f(u, t)
    @inbounds u .= u .+ dt .* k1
end


"""
    add_noise!(u, dt, σ::Union{Real,Vector}, dim)

Adds scaled Gaussian noise to state vector using scalar or vector diffusion coefficient.

# Arguments
- `u`: State vector, modified in-place
- `dt`: Time step size
- `σ`: Noise amplitude (scalar or vector)
- `dim`: Dimension of state space
"""
function add_noise!(u, dt, σ::Union{Real,Vector}, dim)
    u .+= sqrt(2dt) .* σ .* randn(dim)
end

"""
    add_noise!(u, dt, σ::Matrix, dim)

Adds correlated Gaussian noise to state vector using matrix diffusion coefficient.

# Arguments
- `u`: State vector, modified in-place
- `dt`: Time step size
- `σ`: Matrix diffusion coefficient
- `dim`: Dimension of state space
"""
function add_noise!(u, dt, σ::Matrix, dim)
    u .+= sqrt(2dt) .* (σ * randn(dim))
end


"""
    evolve(u0, dt, Nsteps, f, sigma; seed=123, resolution=1, timestepper=:rk4, boundary=false)

Evolves a stochastic dynamical system forward in time.

# Arguments
- `u0`: Initial state vector
- `dt`: Time step size
- `Nsteps`: Total number of steps to evolve
- `f`: Deterministic drift function f(u, t)
- `sigma`: Diffusion function sigma(u, t)
- `seed`: Random seed for reproducibility
- `resolution`: Save results every `resolution` steps
- `timestepper`: Integration method (`:rk4` or `:euler`)
- `boundary`: If specified as [min, max], resets to u0 when state exceeds these bounds

# Returns
- Matrix of results with shape (dim, Nsave+1) where Nsave = ceil(Nsteps/resolution)
"""
function evolve(u0, dt, Nsteps, f, sigma; seed=123, resolution=1, timestepper=:rk4, boundary=false)
    # Initialize state and storage
    dim = length(u0)
    Nsave = ceil(Int, Nsteps / resolution)

    u = copy(u0)
    results = Matrix{Float64}(undef, dim, Nsave+1)  # Store results as (dim, Nsave)
    results[:, 1] .= u0

    # Set random seed for reproducibility
    Random.seed!(seed)

    # Select integration method
    if timestepper == :rk4
        timestepper = rk4_step!
    elseif timestepper == :euler
        timestepper = euler_step!
    else
        error("Invalid timestepper specified. Use :rk4 or :euler.")
    end

    # Initialize time tracking
    t = 0.0
    save_index = 1
    
    if boundary == false
        # Standard evolution without boundary conditions
        for step in ProgressBar(1:Nsteps)
            # Get diffusion coefficient at current state
            sig = sigma(u, t)
            
            # Perform deterministic step
            timestepper(u, dt, f, t)
            
            # Add stochastic component
            add_noise!(u, dt, sig, dim)

            # Check for divergence
            if any(isnan.(u)) || any(abs.(u) .> 1e5)
                @warn "Divergenza a step $step: u = $u"
                break
            end
            
            # Update time
            t += dt
            
            # Save results at specified resolution
            if step % resolution == 0
                save_index += 1
                results[:, save_index] .= u
            end
        end
    else
        # Evolution with boundary conditions
        count = 0
        for step in ProgressBar(1:Nsteps)
            # Get diffusion coefficient at current state
            sig = sigma(u, t)
            
            # Perform deterministic step
            timestepper(u, dt, f, t)
            
            # Add stochastic component
            add_noise!(u, dt, sig, dim)
            
            # Update time
            t += dt
            
            # Reset if boundary is crossed
            if any(u .< boundary[1]) || any(u .> boundary[2])
                u .= u0
                count += 1
            end
            
            # Save results at specified resolution
            if step % resolution == 0
                save_index += 1
                results[:, save_index] .= u
            end
        end
        println("Percentage of boundary crossings: ", count/Nsteps)
    end
    return results
end

"""
    evolve(u0, dt, Nsteps, f, sigma1, sigma2; seed=123, resolution=1, timestepper=:rk4)

Evolves a stochastic dynamical system with two independent noise sources.

# Arguments
- `u0`: Initial state vector
- `dt`: Time step size
- `Nsteps`: Total number of steps to evolve
- `f`: Deterministic drift function f(u, t)
- `sigma1`: First diffusion function sigma1(u, t)
- `sigma2`: Second diffusion function sigma2(u, t)
- `seed`: Random seed for reproducibility
- `resolution`: Save results every `resolution` steps
- `timestepper`: Integration method (`:rk4` or `:euler`)

# Returns
- Matrix of results with shape (dim, Nsave+1) where Nsave = ceil(Nsteps/resolution)
"""
function evolve(u0, dt, Nsteps, f, sigma1, sigma2; seed=123, resolution=1, timestepper=:rk4)
    # Initialize state and storage
    dim = length(u0)
    Nsave = ceil(Int, Nsteps / resolution)

    u = copy(u0)
    results = Matrix{Float64}(undef, dim, Nsave+1)  # Store results as (dim, Nsave)
    results[:, 1] .= u0

    # Set random seed for reproducibility
    Random.seed!(seed)

    # Select integration method
    if timestepper == :rk4
        timestepper = rk4_step!
    elseif timestepper == :euler
        timestepper = euler_step!
    else
        error("Invalid timestepper specified. Use :rk4 or :euler.")
    end

    # Initialize time tracking
    t = 0.0
    save_index = 1
    
    for step in 1:Nsteps
        # Get diffusion coefficients at current state
        sig1 = sigma1(u, t)
        sig2 = sigma2(u, t)
        
        # Perform deterministic step
        timestepper(u, dt, f, t)
        
        # Add stochastic components from both noise sources
        add_noise!(u, dt, sig1, dim)
        add_noise!(u, dt, sig2, dim)
        
        # Update time
        t += dt
        
        # Save results at specified resolution
        if (step-1) % resolution == 0
            save_index += 1
            results[:, save_index] .= u
        end
    end
    return results
end

"""
    evolve(u0, dt, Nsteps, f; resolution=1)

Evolves a deterministic dynamical system (no noise).

# Arguments
- `u0`: Initial state vector
- `dt`: Time step size
- `Nsteps`: Total number of steps to evolve
- `f`: Deterministic drift function f(u, t)
- `resolution`: Save results every `resolution` steps

# Returns
- Matrix of results with shape (dim, Nsave+1) where Nsave = ceil(Nsteps/resolution)
"""
evolve(u0, dt, Nsteps, f; resolution=1) = evolve(u0, dt, Nsteps, f, 0.0; resolution=resolution)



"""
    evolve(u0, dt, Nsteps, f, sigma; seed=123, resolution=1, timestepper=:rk4, boundary=false, n_ens=1)

Evolves a stochastic dynamical system forward in time, optionally generating an ensemble of trajectories.

# Arguments
- `u0`: Initial state vector
- `dt`: Time step size
- `Nsteps`: Total number of steps to evolve
- `f`: Deterministic drift function f(u, t)
- `sigma`: Diffusion function sigma(u, t)
- `seed`: Base random seed for reproducibility
- `resolution`: Save results every `resolution` steps
- `timestepper`: Integration method (`:rk4` or `:euler`)
- `boundary`: If specified as [min, max], resets to u0 when state exceeds these bounds
- `n_ens`: Number of ensemble trajectories to generate (parallelized if > 1)

# Returns
- Array of results with shape (dim, Nsave+1, n_ens) where Nsave = ceil(Nsteps/resolution)
"""
function evolve_ens(u0, dt, Nsteps, f, sigma; seed=123, resolution=1, timestepper=:rk4, boundary=false, n_ens=1)
    # Initialize state and storage dimensions
    dim = length(u0)
    Nsave = ceil(Int, Nsteps / resolution)
    
    # Select integration method
    if timestepper == :rk4
        timestepper_fn = rk4_step!
    elseif timestepper == :euler
        timestepper_fn = euler_step!
    else
        error("Invalid timestepper specified. Use :rk4 or :euler.")
    end
    
    # Create shared array for results to avoid copying between workers
    results = n_ens > 1 ? 
        SharedArray{Float64}(dim, Nsave+1, n_ens) : 
        Array{Float64}(undef, dim, Nsave+1, n_ens)
    
    # Define the single trajectory evolution function
    function evolve_single_trajectory(ens_idx)
        # Create a unique seed for this ensemble member
        trajectory_seed = seed + ens_idx * 1000
        Random.seed!(trajectory_seed)
        
        # Set initial conditions
        u = copy(u0)
        results[:, 1, ens_idx] .= u0
        
        # Initialize time tracking
        t = 0.0
        save_index = 1
        
        if boundary == false
            # Standard evolution without boundary conditions
            for step in 1:Nsteps
                # Get diffusion coefficient at current state
                sig = sigma(u, t)
                
                # Perform deterministic step
                timestepper_fn(u, dt, f, t)
                
                # Add stochastic component
                add_noise!(u, dt, sig, dim)
                
                # Update time
                t += dt
                
                # Save results at specified resolution
                if step % resolution == 0
                    save_index += 1
                    results[:, save_index, ens_idx] .= u
                end
            end
        else
            # Evolution with boundary conditions
            count = 0
            for step in 1:Nsteps
                # Get diffusion coefficient at current state
                sig = sigma(u, t)
                
                # Perform deterministic step
                timestepper_fn(u, dt, f, t)
                
                # Add stochastic component
                add_noise!(u, dt, sig, dim)
                
                # Update time
                t += dt
                
                # Reset if boundary is crossed
                if any(u .< boundary[1]) || any(u .> boundary[2])
                    u .= u0
                    count += 1
                end
                
                # Save results at specified resolution
                if step % resolution == 0
                    save_index += 1
                    results[:, save_index, ens_idx] .= u
                end
            end
            
            if ens_idx == 1  # Only print once to avoid clutter
                println("Percentage of boundary crossings: ", count/Nsteps)
            end
        end
        
        return nothing  # Results are stored in the shared array
    end
    
    # Execute trajectories (in parallel if n_ens > 1)
    if n_ens == 1
        evolve_single_trajectory(1)
    else
        # Use distributed computing for multiple trajectories
        @sync @distributed for i in 1:n_ens
            evolve_single_trajectory(i)
        end
    end
    
    return Array(results)  # Convert SharedArray to regular Array before returning
end

"""
    evolve(u0, dt, Nsteps, f, sigma1, sigma2; seed=123, resolution=1, timestepper=:rk4, n_ens=1)

Evolves a stochastic dynamical system with two independent noise sources, optionally generating an ensemble of trajectories.

# Arguments
- `u0`: Initial state vector
- `dt`: Time step size
- `Nsteps`: Total number of steps to evolve
- `f`: Deterministic drift function f(u, t)
- `sigma1`: First diffusion function sigma1(u, t)
- `sigma2`: Second diffusion function sigma2(u, t)
- `seed`: Base random seed for reproducibility
- `resolution`: Save results every `resolution` steps
- `timestepper`: Integration method (`:rk4` or `:euler`)
- `n_ens`: Number of ensemble trajectories to generate (parallelized if > 1)

# Returns
- Array of results with shape (dim, Nsave+1, n_ens) where Nsave = ceil(Nsteps/resolution)
"""
function evolve_ens(u0, dt, Nsteps, f, sigma1, sigma2; seed=123, resolution=1, timestepper=:rk4, n_ens=1)
    # Initialize state and storage dimensions
    dim = length(u0)
    Nsave = ceil(Int, Nsteps / resolution)
    
    # Select integration method
    if timestepper == :rk4
        timestepper_fn = rk4_step!
    elseif timestepper == :euler
        timestepper_fn = euler_step!
    else
        error("Invalid timestepper specified. Use :rk4 or :euler.")
    end
    
    # Create shared array for results to avoid copying between workers
    results = n_ens > 1 ? 
        SharedArray{Float64}(dim, Nsave+1, n_ens) : 
        Array{Float64}(undef, dim, Nsave+1, n_ens)
    
    # Define the single trajectory evolution function
    function evolve_single_trajectory(ens_idx)
        # Create a unique seed for this ensemble member
        trajectory_seed = seed + ens_idx * 1000
        Random.seed!(trajectory_seed)
        
        # Set initial conditions
        u = copy(u0)
        results[:, 1, ens_idx] .= u0
        
        # Initialize time tracking
        t = 0.0
        save_index = 1
        
        for step in 1:Nsteps
            # Get diffusion coefficients at current state
            sig1 = sigma1(u, t)
            sig2 = sigma2(u, t)
            
            # Perform deterministic step
            timestepper_fn(u, dt, f, t)
            
            # Add stochastic components from both noise sources
            add_noise!(u, dt, sig1, dim)
            add_noise!(u, dt, sig2, dim)
            
            # Update time
            t += dt
            
            # Save results at specified resolution
            if (step-1) % resolution == 0
                save_index += 1
                results[:, save_index, ens_idx] .= u
            end
        end
        
        return nothing  # Results are stored in the shared array
    end
    
    # Execute trajectories (in parallel if n_ens > 1)
    if n_ens == 1
        evolve_single_trajectory(1)
    else
        # Use distributed computing for multiple trajectories
        @sync @distributed for i in 1:n_ens
            evolve_single_trajectory(i)
        end
    end
    
    return Array(results)  # Convert SharedArray to regular Array before returning
end

"""
    evolve(u0, dt, Nsteps, f; resolution=1, n_ens=1)

Evolves a deterministic dynamical system (no noise), optionally generating an ensemble of identical trajectories.

# Arguments
- `u0`: Initial state vector
- `dt`: Time step size
- `Nsteps`: Total number of steps to evolve
- `f`: Deterministic drift function f(u, t)
- `resolution`: Save results every `resolution` steps
- `n_ens`: Number of ensemble trajectories to generate (all identical for deterministic systems)

# Returns
- Array of results with shape (dim, Nsave+1, n_ens) where Nsave = ceil(Nsteps/resolution)
"""
function evolve_ens(u0, dt, Nsteps, f; resolution=1, n_ens=1)
    # For deterministic systems, compute a single trajectory and replicate
    trajectory = evolve_ens(u0, dt, Nsteps, f, 0.0; resolution=resolution, n_ens=n_ens)
    
    # If only one ensemble member requested, return as is
    if n_ens == 1
        return trajectory
    end
    
    # Otherwise, replicate the deterministic trajectory for all ensemble members
    dim, timesteps, _ = size(trajectory)
    result = Array{Float64}(undef, dim, timesteps, n_ens)
    
    for i in 1:n_ens
        result[:, :, i] .= trajectory[:, :, 1]
    end
    
    return result
end


"""
Adds chaotic noise to state vector using matrix diffusion coefficient.

# Arguments
- `u`: State vector, modified in-place
- `dt`: Time step size
- `σ`: Matrix diffusion coefficient
- `y2_t`: chaotic noise vector (e.g., from a underlying chaotic process)
"""

function add_y2!(u, dt, σ::Matrix, y2_t)
    u .+= sqrt(2dt) .* (σ * y2_t)
end

""" 
    add_y2!(u, dt, σ::Real, y2_t::Vector)
version of add_noise with scalar sigma
"""

function add_y2!(u, dt, σ::Real, y2_t)
    u .+= sqrt(2 * dt) * σ .* y2_t
end
"""
    evolve(u0, dt, Nsteps, f, sigma; seed=123, resolution=1, timestepper=:rk4, boundary=false)

Evolves a stochastic dynamical system forward in time with chaotic noise.

# Arguments
- `u0`: Initial state vector
- `dt`: Time step size
- `Nsteps`: Total number of steps to evolve
- `f`: Deterministic drift function f(u, t)
- `sigma`: Diffusion function sigma(u, t)
- `Y2_series`: Matrix of chaotic noise vectors (each column corresponds to a time step)
- `seed`: Random seed for reproducibility
- `resolution`: Save results every `resolution` steps
- `timestepper`: Integration method (`:rk4` or `:euler`)
- `boundary`: If specified as [min, max], resets to u0 when state exceeds these bounds

# Returns
- Matrix of results with shape (dim, Nsave+1) where Nsave = ceil(Nsteps/resolution)
"""
function evolve_chaos(u0, dt, Nsteps, f, sigma, Y2_series; seed=123, resolution=1, timestepper=:rk4, boundary=false)
    # Initialize state and storage
    dim = length(u0)
    Nsave = ceil(Int, Nsteps / resolution)

    u = copy(u0)
    results = Matrix{Float64}(undef, dim, Nsave+1)  # Store results as (dim, Nsave)
    results[:, 1] .= u0

    # Set random seed for reproducibility
    Random.seed!(seed)

    # Select integration method
    if timestepper == :rk4
        timestepper = rk4_step!
    elseif timestepper == :euler
        timestepper = euler_step!
    else
        error("Invalid timestepper specified. Use :rk4 or :euler.")
    end

    # Initialize time tracking
    t = 0.0
    save_index = 1
    
    if boundary == false
        # Standard evolution without boundary conditions
        for step in ProgressBar(1:Nsteps)
            # Get diffusion coefficient at current state
            sig = sigma(u, t)
            
            # Perform deterministic step
            timestepper(u, dt, f, t)
            
            # Add stochastic component
            y2_t = Y2_series[step] 
            add_y2!(u, dt, sig, y2_t)

            # Check for divergence
            if any(isnan.(u)) || any(abs.(u) .> 1e5)
                @warn "Divergenza a step $step: u = $u"
                break
            end
            
            # Update time
            t += dt
            
            # Save results at specified resolution
            if step % resolution == 0
                save_index += 1
                results[:, save_index] .= u
            end
        end
    else
        # Evolution with boundary conditions
        count = 0
        for step in ProgressBar(1:Nsteps)
            # Get diffusion coefficient at current state
            sig = sigma(u, t)
            
            # Perform deterministic step
            timestepper(u, dt, f, t)
            
            # Add stochastic component
            add_noise!(u, dt, sig, dim)
            
            # Update time
            t += dt
            
            # Reset if boundary is crossed
            if any(u .< boundary[1]) || any(u .> boundary[2])
                u .= u0
                count += 1
            end
            
            # Save results at specified resolution
            if step % resolution == 0
                save_index += 1
                results[:, save_index] .= u
            end
        end
        println("Percentage of boundary crossings: ", count/Nsteps)
    end
    return results
end
