
function create_linear_interpolator(R_data::Matrix{Float64}, dt::Float64)
    num_lags, D = size(R_data)
    time_points = 0.0:dt:((num_lags - 1) * dt)
    return function(τ::Float64)
        if τ <= 0.0; return SVector{D,Float64}(R_data[1, :]); end
        if τ >= time_points[end]; return SVector{D,Float64}(R_data[end, :]); end
        idx = searchsortedlast(time_points, τ)
        if idx == 0 || idx >= num_lags; return SVector{D,Float64}(R_data[clamp(idx, 1, num_lags), :]); end
        t1, t2 = time_points[idx], time_points[idx+1]
        R1, R2 = SVector{D,Float64}(R_data[idx, :]), SVector{D,Float64}(R_data[idx+1, :])
        return R1 + (R2 - R1) * ((τ - t1) / (t2 - t1))
    end
end

function f_star(t::Float64, T::Float64, u::SVector, R_interpolators::Vector, λ::Vector)
    if t < 0 || t > T; return 0.0; end
    τ = T - t
    result = 0.0
    for i in 1:length(λ)
        result += λ[i] * dot(u, R_interpolators[i](τ))
    end
    return result
end

function compute_M_matrix(u::SVector, R_interpolators::Vector, T::Float64, N_OBS::Int)
    M = Matrix{Float64}(undef, N_OBS, N_OBS)
    # This can be parallelized, but for small N_OBS, the overhead might not be worth it.
    # Sticking with serial computation here for simplicity in the inner loop.
    for i in 1:N_OBS, j in i:N_OBS
        integrand(t) = dot(u, R_interpolators[i](T - t)) * dot(u, R_interpolators[j](T - t))
        integral_val, _ = quadgk(integrand, 0, T, rtol=1e-8)
        M[i, j] = integral_val
        if i != j; M[j, i] = integral_val; end
    end
    return M
end

function compute_C_matrices(R_interpolators::Vector, T::Float64, N_OBS::Int, D::Int)
    C_matrices = [Matrix{Float64}(undef, D, D) for _ in 1:N_OBS, _ in 1:N_OBS]
    ThreadsX.foreach(CartesianIndices((N_OBS, N_OBS))) do idx
        i, j = idx[1], idx[2]
        if i <= j
            integrand_matrix(t) = R_interpolators[i](T - t) * R_interpolators[j](T - t)'
            integral_mat, _ = quadgk(integrand_matrix, 0, T, rtol=1e-8)
            C_matrices[i, j] = integral_mat
            if i != j; C_matrices[j, i] = integral_mat'; end
        end
    end
    return C_matrices
end

function find_optimal_u(
    u_initial::SVector,
    R_interp::Vector,
    C_mats::Array{Matrix{Float64}, 2},
    delta_m::Vector{Float64},
    T::Float64,
    N::Int
)
    u_k = u_initial
    cost_history = Float64[]
    lambda_k = Vector{Float64}(undef, N)

    for k in 1:MAX_ITERATIONS
        M_matrix = compute_M_matrix(u_k, R_interp, T, N)
        M_reg = M_matrix + REGULARIZATION_ALPHA * I
        lambda_k = -(M_reg \ delta_m)
        cost = 0.5 * dot(lambda_k, M_matrix * lambda_k)
        push!(cost_history, cost)
        
        Q_matrix = sum(lambda_k[i] * lambda_k[j] * C_mats[i,j] for i in 1:N, j in 1:N)
        
        eigen_decomp = eigen(Symmetric(Q_matrix))
        idx_max_eig = argmax(eigen_decomp.values)
        u_next_raw = SVector{D, Float64}(eigen_decomp.vectors[:, idx_max_eig])

        if dot(u_k, u_next_raw) < 0
            u_next_raw = -u_next_raw
        end
        
        u_k_plus_1 = normalize((1 - STEP_SIZE) * u_k + STEP_SIZE * u_next_raw)
        
        change = 1.0 - abs(dot(u_k, u_k_plus_1))
        u_k = u_k_plus_1
        
        if change < CONVERGENCE_TOL
            break
        end
    end

    # Enforce sign convention: f*(0) >= 0
    f_at_zero = sum(lambda_k[i] * dot(u_k, R_interp[i](T)) for i in 1:N)
    if f_at_zero < 0
        u_k = -u_k
    end
    
    # Return final u, final λ, final cost, and the full history
    return (u_final=u_k, lambda_final=lambda_k, final_cost=last(cost_history), history=cost_history)
end