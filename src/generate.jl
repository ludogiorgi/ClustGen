function rk4_step!(u, dt, f, t)
    k1 = f(u, t)
    k2 = f(u .+ 0.5 .* dt .* k1, t + 0.5 * dt)
    k3 = f(u .+ 0.5 .* dt .* k2, t + 0.5 * dt)
    k4 = f(u .+ dt .* k3, t + dt)
    @inbounds u .= u .+ (dt / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
end

function euler_step!(u, dt, f, t)
    k1 = f(u, t)
    @inbounds u .= u .+ dt .* k1
end

function add_noise!(u, dt, σ::Union{Real,Vector}, dim)
    u .+= sqrt(2dt) .* σ .* randn(dim)
end

function add_noise!(u, dt, σ::Matrix, dim)
    u .+= sqrt(2dt) .* (σ * randn(dim))
end

function evolve(u0, dt, Nsteps, f, sigma; seed=123, resolution=1, timestepper=:rk4, boundary=false)
    dim = length(u0)
    Nsave = ceil(Int, Nsteps / resolution)

    u = copy(u0)
    results = Matrix{Float64}(undef, dim, Nsave+1)  # Store results as (dim, Nsave)
    results[:, 1] .= u0

    Random.seed!(seed)

    if timestepper == :rk4
        timestepper = rk4_step!
    elseif timestepper == :euler
        timestepper = euler_step!
    else
        error("Invalid timestepper specified. Use :rk4 or :euler.")
    end

    t = 0.0
    save_index = 1
    if boundary == false
        for step in 1:Nsteps
            sig = sigma(u, t)
            timestepper(u, dt, f, t)
            add_noise!(u, dt, sig, dim)
            t += dt
            if step % resolution == 0
                save_index += 1
                results[:, save_index] .= u
            end
        end
    else
        count = 0
        for step in 1:Nsteps
            sig = sigma(u, t)
            timestepper(u, dt, f, t)
            add_noise!(u, dt, sig, dim)
            t += dt
            if any(u .< boundary[1]) || any(u .> boundary[2])
                u .= u0
                count += 1
            end
            if step % resolution == 0
                save_index += 1
                results[:, save_index] .= u
            end
        end
        println("Percentiage of boundary crossings: ", count/Nsteps)
    end
    return results
end

function evolve(u0, dt, Nsteps, f, sigma1, sigma2; seed=123, resolution=1, timestepper=:rk4)
    dim = length(u0)
    Nsave = ceil(Int, Nsteps / resolution)

    u = copy(u0)
    results = Matrix{Float64}(undef, dim, Nsave+1)  # Store results as (dim, Nsave)
    results[:, 1] .= u0

    Random.seed!(seed)

    if timestepper == :rk4
        timestepper = rk4_step!
    elseif timestepper == :euler
        timestepper = euler_step!
    else
        error("Invalid timestepper specified. Use :rk4 or :euler.")
    end

    t = 0.0
    save_index = 1
    for step in 1:Nsteps
        sig1 = sigma1(u, t)
        sig2 = sigma2(u, t)
        timestepper(u, dt, f, t)
        add_noise!(u, dt, sig1, dim)
        add_noise!(u, dt, sig2, dim)
        t += dt
        if (step-1) % resolution == 0
            save_index += 1
            results[:, save_index] .= u
        end
    end
    return results
end

evolve(u0, dt, Nsteps, f; resolution=1) = evolve(u0, dt, Nsteps, f, 0.0; resolution=resolution)