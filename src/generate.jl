
function U_1D(x; A1=1.0, B1=0.0)
    return (x + A1)^2 * (x - A1)^2 + B1 * x
end 

function ∇U_1D(x; A1=1.0, B1=0.0)
    ∇U = 2 * (x[1] + A1) * (x[1] - A1)^2 + 2 * (x[1] - A1) * (x[1] + A1)^2 + B1
    return [∇U]

end

function U_2D(x; A1=1.0, A2=1.2, B1=0.6, B2=0.3)
    return (x[1] + A1)^2 * (x[1] - A1)^2 + (x[2] + A2)^2 * (x[2] - A2)^2 + B1 * x[1] + B2 * x[2]
end

function ∇U_2D(x; A1=1.0, A2=1.2, B1=0.6, B2=0.3)
    ∇U1 = 2 * (x[1] + A1) * (x[1] - A1)^2 + 2 * (x[1] - A1) * (x[1] + A1)^2 + B1
    ∇U2 = 2 * (x[2] + A2) * (x[2] - A2)^2 + 2 * (x[2] - A2) * (x[2] + A2)^2 + B2
    return [∇U1, ∇U2]
end

function potential_data(x0, timesteps, dt, res; ϵ=√2)
    dim = length(x0)
    if dim == 1
        ∇U = ∇U_1D
    elseif dim == 2
        ∇U = ∇U_2D
    else
        error("Dimension not supported")
    end
    force(x) = -∇U(x)
    x = []
    x_temp = x0
    for i in ProgressBar(2:timesteps)
        rk4_step!(x_temp, dt, force)
        @inbounds x_temp .+= ϵ * randn(dim) * sqrt(dt)
        if i % res == 0
            push!(x, copy(x_temp))
        end
    end
    x = hcat(x...)
    # x = (x .- minimum(x)) ./ (maximum(x) - minimum(x))
    println("Autocorrelation: ", autocor(x')[2,:])
    @info "saving data for potential well"
    hfile = h5open(pwd() * "/data/potential_data_D$(dim).hdf5", "w")
    hfile["x"] = x
    hfile["dt"] = dt
    hfile["res"] = res
    close(hfile)
    @info "done saving data for potential well"
end

function lorenz96(u, F)
    n = length(u)
    du = similar(u)
    for i in 1:n
        du[i] = (u[mod1(i+1, n)] - u[mod1(i-2, n)]) * u[mod1(i-1, n)] - u[i] + F
    end
    return du
end

function simulate_lorenz96(n_steps, dt, F, u0, σ; seed=123, res=1)
    Random.seed!(seed)
    dim = length(u0)
    u = []
    push!(u, u0)
    uOld = u0
    for t in 2:n_steps
        du = lorenz96(uOld, F) * dt
        noise = σ * randn(dim) * sqrt(2dt)
        uNew = uOld .+ du .+ noise
        if t % res == 0
            push!(u, uNew)
        end
        uOld = uNew
    end
    return hcat(u...)
end




    
    
