function rk4_step!(u, dt, f)
    k1 = f(u)
    k2 = f(u .+ 0.5 .* dt .* k1)
    k3 = f(u .+ 0.5 .* dt .* k2)
    k4 = f(u .+ dt .* k3)
    @inbounds u .= u .+ (dt / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
end

function computeSigma(X,
                      piVec,
                      Q,
                      gradLogp)

    (N, D) = size(X)
    @assert length(piVec) == N "piVec must have length N"
    @assert size(Q) == (N, N) "Q must be N×N"
    @assert size(gradLogp) == (N, D) "gradLogp must be N×D"
    
    # 1) Construct M (D×D)
    #    M[i,j] = sum_{n=1..N, m=1..N} X[m,i] * X[n,j] * piVec[n] * Q[m,n]
    M = zeros(D, D)
    for i in 1:D
        for j in 1:D
            s = 0.0
            for n in 1:N
                for m in 1:N
                    s += X[m, i] * X[n, j] * piVec[n] * Q[m, n]
                end
            end
            M[i, j] = s
        end
    end

    # 2) Construct V (D×D)
    #    V[j,k] = sum_{n=1..N} X[n,j] * piVec[n] * gradLogp[n,k]
    V = zeros(D, D)
    for j in 1:D
        for k in 1:D
            s = 0.0
            for n in 1:N
                s += X[n, j] * piVec[n] * gradLogp[n, k]
            end
            V[j, k] = s
        end
    end

    # 3) Compute Sigma * Sigma^T = M * (V^T)^(-1)
    #    => SigmaSigmaT = M * inv(V') 
    #    Then Sigma = sqrtm(SigmaSigmaT)

    SigmaSigmaT = M * inv(V')
    Sigma = sqrt(SigmaSigmaT)   # principal matrix square root

    return Sigma
end


function covariance(x1, x2; timesteps=length(x), progress = false)
    μ1 = mean(x1)
    μ2 = mean(x2)
    autocor = zeros(timesteps)
    progress ? iter = ProgressBar(1:timesteps) : iter = 1:timesteps
    for i in iter
        autocor[i] = mean(x1[i:end] .* x2[1:end-i+1]) - μ1 * μ2
    end
    return autocor
end


function covariance(g⃗1, g⃗2, Q::Eigen, timelist; progress=false)
   #  @assert all(real.(Q.values[1:end-1]) .< 0) "Did not pass an ergodic generator matrix"
    autocov = zeros(length(timelist))
    # Q  = V Λ V⁻¹
    Λ, V = Q
    p = real.(V[:, end] ./ sum(V[:, end]))
    v1 = V \ (p .* g⃗1)
    w2 = g⃗2' * V
    μ1 = sum(p .* g⃗1)
    μ2 = sum(p .* g⃗2)
    progress ? iter = ProgressBar(eachindex(timelist)) : iter = eachindex(timelist)
    for i in iter
        autocov[i] = real(w2 * (exp.(Λ .* timelist[i]) .* v1) - μ1 * μ2)
    end
    return autocov
end

covariance(g⃗1, g⃗2, Q, timelist; progress = false) = covariance(g⃗1, g⃗2, eigen(Q), timelist; progress = progress)