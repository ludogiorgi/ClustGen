using Revise
using ClustGen
using StateSpacePartitions
using LinearAlgebra
using Random 
using ProgressBars
using Statistics
using KernelDensity
using HDF5
using Flux
using QuadGK
using BSON
using Plots
using StatsBase
using MarkovChainHammer
import MarkovChainHammer.Trajectory: ContinuousTimeEmpiricalProcess
import LaTeXStrings
##

function ∇U_c(x, t; A1=1.0, A2=1.0, B1=0.6, B2=0.3, C=1.0, D=0.0)
    # Conservative gradient terms
    ∇U1 = 2 * (x[1] + A1) * (x[1] - A1)^2 + 2 * (x[1] - A1) * (x[1] + A1)^2 + B1 + C * (x[1] * x[2])^2
    ∇U2 = 2 * (x[2] + A2) * (x[2] - A2)^2 + 2 * (x[2] - A2) * (x[2] + A2)^2 + B2 + C * (x[1] * x[2])^2
    
    # Non-conservative term (e.g., rotational flow)
    F1 = -D * x[2]
    F2 = D * x[1]
    
    # Total force
    return [-∇U1 + F1, -∇U2 + F2]
end

function ∇U_nc(x, t; A1=1.0, A2=1.0, B1=0.6, B2=0.3, C=1.0, D=3.0)
    # Conservative gradient terms
    ∇U1 = 2 * (x[1] + A1) * (x[1] - A1)^2 + 2 * (x[1] - A1) * (x[1] + A1)^2 + B1 + C * (x[1] * x[2])^2
    ∇U2 = 2 * (x[2] + A2) * (x[2] - A2)^2 + 2 * (x[2] - A2) * (x[2] + A2)^2 + B2 + C * (x[1] * x[2])^2
    
    # Non-conservative term (e.g., rotational flow)
    F1 = -D * x[2]
    F2 = D * x[1]
    
    # Total force
    return [-∇U1 + F1, -∇U2 + F2]
end


dt = 0.025
Σ_true = [1.0 0.5; 0.5 1.0]
sigma(x,t) = Σ_true
dim = 2
Nsteps = 10000000
obs_c = evolve(zeros(dim), dt, Nsteps, ∇U_c, sigma; timestepper=:rk4, resolution=1)
obs_nc = evolve(zeros(dim), dt, Nsteps, ∇U_nc, sigma; timestepper=:rk4, resolution=1)

plt1 = Plots.plot(obs_c[1,1:1000], obs_c[2,1:1000], label="Conservative")
plt2 = Plots.plot(obs_nc[1,1:1000], obs_nc[2,1:1000], label="Non-Conservative")
Plots.plot(plt1, plt2, layout=(1,2), size=(800,400), title="Trajectory in 2D Phase Space", xlabel="x1", ylabel="x2")
##

normalization = false
σ_value = 0.05

μ_c = repeat(obs_c[:,1:100:end], 1, 1)
μ_nc = repeat(obs_nc[:,1:100:end], 1, 1)

averages_c, centers_c, Nc_c, labels_c = f_tilde_labels(σ_value, μ_c; prob=0.1, do_print=true, conv_param=0.001, normalization=normalization)
averages_nc, centers_nc, Nc_nc, labels_nc = f_tilde_labels(σ_value, μ_nc; prob=0.1, do_print=true, conv_param=0.001, normalization=normalization)

Qc = generator(labels_c)
Qnc = generator(labels_nc)

Qc_c, Qc_nc = decomposition(Qc)
Qnc_c, Qnc_nc = decomposition(Qnc)

lc,vc = eigen(Qc)
lnc,vnc = eigen(Qnc)

Plots.plot(imag.(lc))
Plots.plot!(imag.(lnc))

Plots.plot(real.(lc))
Plots.plot!(real.(lnc))

##
tsteps = 81
res = 10

auto_obs_c = zeros(dim, tsteps)
auto_obs_nc = zeros(dim, tsteps)
auto_Qc = zeros(dim, tsteps)
auto_Qnc_c = zeros(dim, tsteps)

for i in 1:dim
    auto_obs_c[i,:] = autocovariance(obs_c[i,1:res:end]; timesteps=tsteps) 
    auto_obs_nc[i,:] = autocovariance(obs_nc[i,1:res:end]; timesteps=tsteps)
    auto_Qc[i,:] = autocovariance(centers_c[i,:], Qc, [0:dt*res:Int(res * (tsteps-1) * dt)...])
    auto_Qnc_c[i,:] = autocovariance(centers_nc[i,:], Qnc_c, [0:dt*res:Int(res * (tsteps-1) * dt)...])
end

Plots.plot(auto_obs_c[2,:])
Plots.plot!(auto_obs_nc[2,:])
Plots.plot!(auto_Qc[2,:])
Plots.plot!(auto_Qnc_c[2,:])

##


using LinearAlgebra
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
# Loss: sum of Frobenius‐norms squared between exp(Q t_k) and P_obs[k]
# -----------------------------------------------------------------------------
function loss(s, P_obs::Vector{Matrix{Float64}}, t_list::Vector{Float64})
    n = size(P_obs[1],1)
    Q = Q_from_s(s, n)
    J = 0.0
    for (P_k, t) in zip(P_obs, t_list)
        E = exp(Q * t)
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
t_list = [1, 2, 5, 10, 20] 
# synthetic "observed" P(t) for testing: use exp(Q_true * t)

P_obs = Matrix{Float64}[]
for t in t_list
    push!(P_obs, perron_frobenius(labels_nc; step=t)) 
end

# estimate Q
Q_est = estimate_generator(P_obs, Float64.(t_list))
auto_Q_est = zeros(dim, tsteps)

for i in 1:dim
    auto_Q_est[i,:] = autocovariance(centers_c[i,:], Q_est, [0:dt*res:Int(res * (tsteps-1) * dt)...])
end

Plots.plot(auto_obs_nc[2,:])
Plots.plot!(auto_Qnc_c[2,:])
Plots.plot!(auto_Q_est[2,:])