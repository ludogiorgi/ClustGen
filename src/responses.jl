
function generate_numerical_response(model, model_pert, dim, dt, n_tau, n_therm, sigma, Obs, dim_Obs; n_ens=1000, resolution=1)

    δObs_ens = zeros(dim_Obs,n_tau,n_ens)
    for i in ProgressBar(1:n_ens)
        seed = abs(rand(Int))
        X0 = evolve(randn(dim), dt, n_therm, model, sigma; seed=seed, resolution=n_therm)[:,end] 
        seed = abs(rand(Int))
        X = evolve(X0, dt, n_tau*resolution, model, sigma; seed=seed, resolution=resolution) 
        X_pert = evolve(X0, dt, n_tau*resolution, model_pert, sigma; seed=seed, resolution=resolution) 
        δObs_ens[:,:,i] = Obs(X_pert) .- Obs(X)
    end
    δObs = mean(δObs_ens, dims=3)[:,:,1]

    return δObs
end

function generate_numerical_response3(model, u, dim, dt, n_tau, n_therm, sigma, Obs, dim_Obs; n_ens=1000, resolution=1)
    δObs_ens = zeros(dim_Obs, n_tau+1, n_ens)
    
    for i in ProgressBar(1:n_ens)
        seed = abs(rand(Int))
        X0 = evolve(randn(dim), dt, n_therm, model, sigma; seed=seed, resolution=n_therm)[:,end] 
        seed = abs(rand(Int))
        ϵ = u(X0)
        X0_pert = X0 .+ ϵ
        X = evolve(X0, dt, n_tau*resolution, model, sigma; seed=seed, resolution=resolution) 
        X_pert = evolve(X0_pert, dt, n_tau*resolution, model, sigma; seed=seed, resolution=resolution) 
        δObs_ens[:,:,i] = Obs(X_pert) .- Obs(X)
    end
    
    δObs = mean(δObs_ens, dims=3)[:,:,1]
    return δObs
end

function generate_numerical_response3(model, u, dim, dt, n_tau, n_therm, sigma1, sigma2, Obs, dim_Obs; n_ens=1000, resolution=1)
    δObs_ens = zeros(dim_Obs, n_tau+1, n_ens)
    
    for i in ProgressBar(1:n_ens)
        seed = abs(rand(Int))
        X0 = evolve(randn(dim), dt, n_therm, model, sigma1, sigma2; seed=seed, resolution=n_therm)[:,end] 
        seed = abs(rand(Int))
        ϵ = u(X0)
        X0_pert = X0 .+ ϵ
        X = evolve(X0, dt, n_tau*resolution, model, sigma1, sigma2; seed=seed, resolution=resolution) 
        X_pert = evolve(X0_pert, dt, n_tau*resolution, model, sigma1, sigma2; seed=seed, resolution=resolution) 
        δObs_ens[:,:,i] = Obs(X_pert) .- Obs(X)
    end
    
    δObs = mean(δObs_ens, dims=3)[:,:,1]
    return δObs
end

function generate_score_response(trj, u, div_u, f, score, dt, n_tau, Obs, dim_Obs)

    dim_trj, steps_trj = size(trj)
    B(x) = - div_u(x) .- (u(x)' * score(x))[1]
    R = zeros(dim_Obs,n_tau+1)
    δObs = zeros(dim_Obs,n_tau+1)
    for i in ProgressBar(1:n_tau+1)
        for j in 1:steps_trj-i+1
            R[:,i] .+= B(trj[:,j]) .* Obs(trj[:,j+i-1])
        end
        R[:, i] ./= (steps_trj - i + 1)
        if i > 1
            for j in 1:i-1
                δObs[:, i] .+= R[:, i - j + 1] * f(j * dt) * dt
            end
        end
    end

    return R, δObs
end