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
function add_noise!(u::AbstractVector, dt, σ::Union{Real, AbstractVector})
    dim = length(u)
    u .+= sqrt(2*dt) .* σ .* randn(dim)
    return u
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
function add_noise!(u::AbstractVector, dt, σ::AbstractMatrix, ::Val{false})
    q = size(σ, 2)
    u .+= sqrt(2*dt) .* (σ * randn(q))
    return u
end

"""
    add_noise!(u, dt, σ::Matrix, dim)

Adds correlated Gaussian noise to state vector using scalar product between the state vector and the diffusion matrix.

# Arguments
- `u`: State vector, modified in-place
- `dt`: Time step size
- `σ`: Matrix diffusion coefficient
- `dim`: Dimension of state space
"""
function add_noise!(u::AbstractVector, dt, σ::AbstractMatrix, ::Val{true})
    q = size(σ, 2)
    u .+= sqrt(2*dt) .* dot(σ[3, :], randn(q))
    return u
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
function evolve(u0, dt, Nsteps, f, sigma;
                seed=123,
                resolution=1,
                timestepper=:rk4,
                boundary=false,
                scalar_prod=false)

    # setup
    dim    = length(u0)
    Nsave  = ceil(Int, Nsteps / resolution)
    u      = copy(u0)
    results = Matrix{Float64}(undef, dim, Nsave+1)
    results[:, 1] .= u0

    Random.seed!(seed)

    stepper! = timestepper == :rk4  ? rk4_step!  :
               timestepper == :euler ? euler_step! :
               error("Invalid timestepper. Use :rk4 or :euler.")

    t = 0.0
    save_index = 1

    if boundary == false
        for step in ProgressBar(1:Nsteps)
            σt = sigma(u, t)

            # drift
            stepper!(u, dt, f, t)

            # noise: dispatch in base al tipo di σt
            if σt isa AbstractMatrix
                add_noise!(u, dt, σt, Val(scalar_prod))
            elseif (σt isa AbstractVector) || (σt isa Real)
                add_noise!(u, dt, σt)
            else
                throw(ArgumentError("sigma(u,t) must return Matrix, Vector or Real; got $(typeof(σt))"))
            end

            # guard-rails
            if any(isnan.(u)) || any(abs.(u) .> 1e5)
                @warn "Divergenza a step $step: u = $u"
                break
            end

            t += dt
            if step % resolution == 0
                save_index += 1
                results[:, save_index] .= u
            end
        end

    else
        crossings = 0
        for step in ProgressBar(1:Nsteps)
            σt = sigma(u, t)

            stepper!(u, dt, f, t)

            if σt isa AbstractMatrix
                add_noise!(u, dt, σt, Val(scalar_prod))
            elseif (σt isa AbstractVector) || (σt isa Real)
                add_noise!(u, dt, σt)
            else
                throw(ArgumentError("sigma(u,t) must return Matrix, Vector or Real; got $(typeof(σt))"))
            end

            t += dt

            # boundary=[min,max] applicato alla prima componente
            if any(u[1] .< boundary[1]) || any(u[1] .> boundary[2])
                u .= u0
                crossings += 1
            end

            if step % resolution == 0
                save_index += 1
                results[:, save_index] .= u
            end
        end
        println("Percentage of boundary crossings: ", crossings/Nsteps)
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
        error("Invalid timestepper specified. Use :rk4 or :euler.")xasmkk
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
        add_noise!(u, dt, sig1)
        add_noise!(u, dt, sig2)

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
                add_noise!(u, dt, sig)
                
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
                add_noise!(u, dt, sig)
                
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
            add_noise!(u, dt, sig1)
            add_noise!(u, dt, sig2)
            
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
    u .+= sqrt(2*dt) .* (σ * y2_t)
end

""" 
    add_y2!(u, dt, σ::Real, y2_t::Vector)
version of add_noise with scalar sigma
"""

function add_y2!(u, dt, σ::Real, y2_t)
    u .+= sqrt(2*dt) * (σ .* y2_t)
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
function evolve_chaos(u0::AbstractVector, dt, Nsteps, f, sigma, Y2_series; seed=123, resolution=1, timestepper=:rk4, boundary=false)
    # Initialize state and storage
    dim = length(u0)
    Nsave = ceil(Int, Nsteps / resolution)

    u = copy(u0)
    results = Matrix{Float64}(undef, dim, Nsave+1)  # Store results as (dim, Nsave)
    results[:, 1] .= u0

    println("DEBUG: u0 = ", u0)
    println("DEBUG: results[:, 1] = ", results[:, 1])

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
            if step == 1
                println("DEBUG after first step:")
                println("results[:, 1] = ", results[:, 1])
                println("results[:, 2] = ", results[:, 2])
            end

        end
        println("Percentage of boundary crossings: ", count/Nsteps)
    end
    return results
end


"""
    rk4_step_scalar!(u, dt, f, t)

Performs a single Euler integration step.

# Arguments
- `u`: Initial State, modified in-place
- `dt`: Time step size
- `f`: Function defining the dynamics, should accept (u, t) arguments
- `t`: Current time
"""

function rk4_step_scalar!(x, dt::Float64, f::Function, t::Float64)
    k1 = f(x, t)
    k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(x + dt * k3, t + dt)
    return x + dt / 6.0 * (k1 + 2k2 + 2k3 + k4)
end

"""
    euler_step_scalar!(u, dt, f, t)

Performs a single Euler integration step.

# Arguments
- `u`: Initial State, modified in-place
- `dt`: Time step size
- `f`: Function defining the dynamics, should accept (u, t) arguments
- `t`: Current time
"""


function euler_step_scalar!(x, dt::Float64, f::Function, t::Float64)
    return x + dt * f(x, t)
end

"""
Scalar version of evolve_chaos for 1D systems

# Arguments
- `u0`: Initial state scalar
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
function evolve_chaos(x0::Float32, dt, Nsteps, f, sigma, Y2_series;
                      seed=123, resolution=1, timestepper=:rk4, boundary=false)
    Nsave = ceil(Int, Nsteps / resolution)

    x = x0
    results = Vector{Float64}(undef, Nsave + 1)
    results[1] = x

    Random.seed!(seed)

    # seleziona metodo di integrazione
    if timestepper == :rk4
        timestepper = rk4_step_scalar!
    elseif timestepper == :euler
        timestepper = euler_step_scalar!
    else
        error("Invalid timestepper specified. Use :rk4 or :euler.")
    end

    t = 0.0
    save_index = 1

    if boundary == false
        for step in ProgressBar(1:Nsteps)
            sig = sigma(x, t)
            x = timestepper(x, dt, f, t)
            y2_t = Y2_series[step]
            x += sig * sqrt(dt) * y2_t

            if isnan(x) || abs(x) > 1e5
                @warn "Divergenza a step $step: x = $x"
                break
            end

            t += dt
            if step % resolution == 0
                save_index += 1
                results[save_index] = x
            end
        end
    else
        count = 0
        for step in ProgressBar(1:Nsteps)
            sig = sigma(x, t)
            x = timestepper(x, dt, f, t)
            x += sig * sqrt(dt) * randn()
            t += dt

            if x < boundary[1] || x > boundary[2]
                x = x0
                count += 1
            end

            if step % resolution == 0
                save_index += 1
                results[save_index] = x
            end
        end
        println("Percentage of boundary crossings: ", count / Nsteps)
    end

    return results
end
