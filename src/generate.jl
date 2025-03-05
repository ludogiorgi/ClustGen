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
        for step in 1:Nsteps
            # Get diffusion coefficient at current state
            sig = sigma(u, t)
            
            # Perform deterministic step
            timestepper(u, dt, f, t)
            
            # Add stochastic component
            add_noise!(u, dt, sig, dim)
            
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
        for step in 1:Nsteps
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