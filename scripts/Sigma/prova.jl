using LinearAlgebra
using LinearAlgebra: expm
using Optim

# -----------------------------------------------------------------------------
# Utility: flatten/unflatten off‑diagonal entries of S
# -----------------------------------------------------------------------------
function S_to_vec(S::AbstractMatrix{T}) where T
    n = size(S,1)
    s = Vector{T}(undef, n*(n-1))
    idx = 1
    for i in 1:n, j in 1:n
        if i != j
            s[idx] = S[i,j]
            idx += 1
        end
    end
    return s
end

function vec_to_S(s::AbstractVector{T}, n::Integer) where T
    S = zeros(T, n, n)
    idx = 1
    for i in 1:n, j in 1:n
        if i != j
            S[i,j] = s[idx]
            idx += 1
        end
    end
    return S
end

# -----------------------------------------------------------------------------
# Build Q = S - diag(sum(S, dims=2))
# -----------------------------------------------------------------------------
function Q_from_s(s, n)
    S = vec_to_S(s, n)
    D = sum(S, dims=2)              # column‐vector of row‐sums
    return S .- Diagonal(vec(D))    # Q_{ij}=S_{ij} for i≠j, Q_{ii}=-∑_{j≠i}S_{ij}
end

# -----------------------------------------------------------------------------
# Loss: sum of Frobenius‐norms squared between expm(Q t_k) and P_obs[k]
# -----------------------------------------------------------------------------
function loss(s, P_obs::Vector{Matrix{Float64}}, t_list::Vector{Float64})
    n = size(P_obs[1],1)
    Q = Q_from_s(s, n)
    J = 0.0
    for (P_k, t) in zip(P_obs, t_list)
        E = expm(Q * t)
        Δ = E .- P_k
        J += sum(abs2, Δ)
    end
    return J
end

# -----------------------------------------------------------------------------
# Main estimation routine
# -----------------------------------------------------------------------------
function estimate_generator(P_obs::Vector{Matrix{Float64}}, t_list::Vector{Float64})
    n = size(P_obs[1],1)
    # initial S: small random off‐diagonals, zero diagonal
    S0 = rand(n,n) .* (1 .- Matrix{Float64}(I, n, n))  # Convert I to explicit matrix
    s0 = S_to_vec(S0)

    # lower‐bounds = 0 for all off‐diagonals, upper=Inf
    lb = zeros(length(s0))
    ub = fill(Inf, length(s0))

    # box‐constrained Nelder–Mead
    res = optimize(
        s -> loss(s, P_obs, t_list),
        lb, ub, s0,
        Fminbox(NelderMead()),
        Optim.Options(iterations = 2000, show_trace = true)
    )

    s_opt = Optim.minimizer(res)
    return Q_from_s(s_opt, n)
end

# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
# (Replace the following with your actual data.)
n = 4
t_list = [1.0, 2.0, 3.0] 
# synthetic “observed” P(t) for testing: use expm(Q_true * t)
Q_true = [-0.8  0.3 0.3 0.2;
           0.1 -0.5 0.2 0.2;
           0.1  0.3 -1.0 0.6;
           0.2  0.2 0.1 -0.5]
P_obs = [exp(Q_true * t) for t in t_list]

# estimate Q
Q_est = estimate_generator(P_obs, t_list)

htm1 = heatmap(Q_true)
htm2 = heatmap(Q_est)
plot(htm1, htm2, layout=(1,2), size=(800,400), title="Estimated vs True Q", xlabel="i", ylabel="j")