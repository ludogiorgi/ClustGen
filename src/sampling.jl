# SAmpling with reverse diffusion method
function sample_reverse(Dim, nn, n_samples, n_diffs, σ, g)
    dt = 1.0 / n_diffs
    ens = zeros(Dim, n_samples)
    for i in 1:n_samples
        xOld = σ(1)*randn(Dim)
        for t in 1:n_diffs
            t_diff = (n_diffs - t + 1) / n_diffs
            s = σ(t_diff)
            score = nn([xOld..., t_diff]) ./ s
            xNew = xOld .+ score .* g(t_diff) .^2 .* dt .+ randn(Dim) .* sqrt(dt) .* g(t_diff)
            xOld = xNew
        end
        ens[:,i] = xOld
    end
    return ens
end

# Sampling with Langevin sampling method
function sample_langevin(T, dt, f, obs; seed=123, res=1, boundary=false)
    Random.seed!(seed)
    N = Int(T / dt)
    dim = length(obs[:,1])
    num_saved_steps = ceil(Int, N / res)
    x = zeros(dim, num_saved_steps)
    idx = 1
    x0 = obs[:,rand(1:length(obs[1,:]))]
    x_temp = x0
    count = 0
    for t in ProgressBar(2:N)
        rk4_step!(x_temp, dt, f)  
        x_temp += √2 * randn(dim) * sqrt(dt) 
        if boundary != false
            if any(x_temp .< boundary[1]) || any(x_temp .> boundary[2])
                x_temp = obs[:,rand(1:length(obs[1,:]))]
                count += 1
            end
        end
        if t % res == 0
            x[:, idx] = x_temp
            idx += 1
        end
    end    
    println("Number of boundary crossings: ", count)
    return x
end

# Sampling with Langevin sampling method
function sample_langevin_Σ(T, dt, f, obs, Σ; seed=123, res=1, boundary=false)
    Random.seed!(seed)
    N = Int(T / dt)
    dim = length(obs[:,1])
    num_saved_steps = ceil(Int, N / res)
    x = zeros(dim, num_saved_steps)
    idx = 1
    x0 = obs[:,rand(1:length(obs[1,:]))]
    x_temp = x0
    count = 0
    for t in ProgressBar(2:N)
        f_Σ2(x) = Σ^2 * f(x)
        rk4_step!(x_temp, dt, f_Σ2)  
        x_temp += Σ * randn(dim) * sqrt(2dt) 
        if boundary != false
            if any(x_temp .< boundary[1]) || any(x_temp .> boundary[2])
                x_temp = obs[:,rand(1:length(obs[1,:]))]
                count += 1
            end
        end
        if t % res == 0
            x[:, idx] = x_temp
            idx += 1
        end
    end    
    println("Number of boundary crossings: ", count)
    return x
end