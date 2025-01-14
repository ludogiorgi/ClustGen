
function decorrelation_times(time_series::Array{Float64, 2}, threshold)
    D, N = size(time_series)
    
    if threshold isa Integer
        # If threshold is an integer, compute correlation only for the first 'threshold' time steps
        autocorr_times = fill(threshold, D, D)
        max_t = min(threshold, N)  # Safeguard in case threshold > N
        corr_series = zeros(D, D, max_t)
        for d1 in 1:D, d2 in 1:D
            for t in 1:max_t
                corr_series[d1, d2, t] = cor(time_series[d1, 1:end-t+1], time_series[d2, t:end])
            end
        end
        return autocorr_times, corr_series
    else
        # If threshold is a float, run the original logic
        autocorr_times = zeros(Int, D, D)
        for d1 in 1:D, d2 in 1:D
            for t in 2:N
                autocorr = cor(time_series[d1, 1:end-t+1], time_series[d2, t:end])
                if autocorr < threshold
                    autocorr_times[d1, d2] = t
                    break
                end
            end
        end
        corr_series = zeros(D, D, maximum(autocorr_times))
        for d1 in 1:D, d2 in 1:D
            for t in 1:autocorr_times[d1, d2]
                corr_series[d1, d2, t] = cor(time_series[d1, 1:end-t+1], time_series[d2, t:end])
            end
        end
        return autocorr_times, corr_series
    end
end

function rk4_step!(u, dt, f)
    k1 = f(u)
    k2 = f(u .+ 0.5 .* dt .* k1)
    k3 = f(u .+ 0.5 .* dt .* k2)
    k4 = f(u .+ dt .* k3)
    @inbounds u .= u .+ (dt / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
end

"""
    computeSigma(X, piVec, Q, gradLogp)

Compute the matrix Sigma (D×D) based on:
  - X:       an N×D matrix of data (row n is x^n in R^D).
  - piVec:   an N-element vector (π^n).
  - Q:       an N×N matrix.
  - gradLogp: an N×D matrix of gradients 
              (row n corresponds to ∇ ln p_S(x^n) in R^D).
Returns Sigma (D×D).
"""
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

function compute_corr(x_i, x_j, π, Q, τ)
    """
    Computes the expression:
    sum_{n=1}^N x_j^n π^n [sum_{m=1}^N x_i^m [exp(Q * τ)]_{mn}]
    
    Arguments:
    x_i  -- Array of size N, values for x_i^m
    x_j  -- Array of size N, values for x_j^n
    π    -- Array of size N, stationary distribution π^n
    Q    -- NxN matrix, the generator matrix Q
    τ    -- Scalar, the time lag τ
    
    Returns:
    result -- Scalar value of the computed expression
    """
    # Compute the matrix exponential exp(Q * τ)
    exp_Qτ = exp(Q * τ)
    
    # Center the input vectors
    x_i = x_i[:] .- mean(x_i)
    x_j = x_j[:] .- mean(x_j)
    
    # Compute the inner sum using matrix-vector multiplication
    inner_sum = exp_Qτ' * x_i
    
    # Compute the final result using element-wise multiplication and summation
    result = sum(x_j .* π .* inner_sum)
    
    return result / std(x_j) / std(x_i)
end