###############  FAST, TYPE-STABLE, PARALLEL VERSION  ###############
using Pkg; Pkg.activate("."); Pkg.instantiate()

using LinearAlgebra
using ClustGen
using Random
using QuadGK
using StatsBase
using Base.Threads
using MarkovChainHammer     # for evolve(...)
##

# ---------------- Triad Model System Definition ----------------

const L11 = -2.0; const L12 = 0.2; const L13 = 0.1; const g2 = 0.6; const g3 = 0.4
const s2_param = 1.2; const s3 = 0.8; const II = 1.0; const ϵ = 0.1

const a = L11 + ϵ * ( (II^2 * s2_param^2) / (2 * g2^2) - (L12^2) / g2 - (L13^2) / g3 )
const b = -2 * (L12 * II) / g2 * ϵ
const c = (II^2) / g2 * ϵ
const B = -(II * s2_param) / g2 * sqrt(ϵ)
const A = -(L12 * B) / II
const s_noise = (L13 * s3) / g3 * sqrt(ϵ)
const F_tilde = (A * B) / 2

# params: [a, b, c, F_tilde, A, B, s_noise]
const params_triad = [a, b, c, F_tilde, A, B, s_noise]

# Drift and noises (state in 1D vector form expected by evolve)
F(x, t; p=params_triad) = @inbounds [-p[4] + p[1]*x[1] + p[2]*x[1]^2 - p[3]*x[1]^3]
sigma1(x, t; p=params_triad) = @inbounds (p[5] - p[6]*x[1]) / √2
sigma2(x, t; p=params_triad) = @inbounds p[7] / √2

# ---------------- Score Functions & Derivatives ----------------
# Correct "true" score: s = 2f/σ^2 - ∂x ln σ^2 = [2(-F~+ax+bx^2-cx^3) + 2B(A-Bx)] / σ^2

function create_true_score_and_derivative_triad(p::AbstractVector{<:Real})
    a, b, c, Ftil, A_p, B_p, s_val = p
    den(u)  = @inbounds s_val^2 + (A_p - B_p*u)^2              # σ^2(u)
    f(u)    = @inbounds -Ftil + a*u + b*u^2 - c*u^3            # drift
    denp(u) = @inbounds 2*(A_p - B_p*u)*(-B_p)                 # (σ^2)'(u)
    num(u)  = @inbounds 2*f(u) + 2*B_p*(A_p - B_p*u)           # 2f - (σ^2)' = 2f + 2B(A-Bu)
    nump(u) = @inbounds 2*(a + 2*b*u - 3*c*u^2) + 2*B_p*(-B_p)

    score(x::Float64)::Float64 = num(x) / den(x)
    scorep(u::Float64)::Float64 = (nump(u)*den(u) - num(u)*denp(u)) / (den(u)^2)
    return score, scorep
end

# Linear (Gaussian) score around mean/var of a series
function create_linear_s_ds(x_t::AbstractVector{<:Real})
    μ = mean(x_t); σ2 = var(x_t)
    score(x::Float64)::Float64  = -(x - μ) / σ2
    scorep(::Float64)::Float64  = -1 / σ2
    return score, scorep
end

# Helper to coerce possibly array-returning score into Float64
_to_scalar(v) = v isa Number ? Float64(v) :
    (v isa AbstractArray && length(v) == 1 ? Float64(v[1]) :
        error("Score function returned non-scalar of size $(size(v))"))

# Build s'(x) by finite-differencing a black-box score and caching on a grid
function construct_divergence_score(s::Function; n_points::Int=500, xlims::Tuple{Real,Real}=(-5.0,5.0))
    s_scalar(x::Float64)::Float64 = _to_scalar(s(x))
    xgrid = Base.range(first(xlims), last(xlims), length=n_points)
    vals  = Vector{Float64}(undef, n_points)
    h = 1e-6
    @threads for i in eachindex(xgrid)
        x = xgrid[i]
        vals[i] = (s_scalar(x + h) - s_scalar(x - h)) / (2h)
    end
    stepx = Float64(step(xgrid))
    x0    = Float64(first(xgrid))
    # Fast linear interpolation
    function sprime_fast(x::Float64)::Float64
        if x <= xgrid[1]; return vals[1]; end
        if x >= xgrid[end]; return vals[end]; end
        t = (x - x0) / stepx + 1
        i = Int(floor(t))
        j = i + 1
        w = t - i
        @inbounds return (1-w)*vals[i] + w*vals[j]
    end
    return sprime_fast
end

# ---------------- Potential & Observables (analytic PDF) ----------------

function potential_triad(x::Float64, p::AbstractVector{<:Real})
    a_p, b_p, c_p, Ftil, A_p, B_p, s_p = p
    σ2(y) = @inbounds (A_p - B_p*y)^2 + s_p^2
    f(y)  = @inbounds -Ftil + a_p*y + b_p*y^2 - c_p*y^3
    I, _  = quadgk(y -> 2*f(y)/σ2(y), 0.0, x; rtol=1e-6)
    return -I + log(σ2(x))
end

p_unnorm_triad(x::Float64, p) = exp(-potential_triad(x, p))

function compute_observables_triad(p, observables::Vector{<:Function})
    bounds = (-6.0, 6.0)
    Z, _ = quadgk(x -> p_unnorm_triad(x, p), bounds...; rtol=1e-8)
    Z == 0.0 && error("Zero normalization, adjust bounds or params.")
    map(obs -> (quadgk(x -> obs(x)*p_unnorm_triad(x, p), bounds...; rtol=1e-8)[1] / Z), observables)
end

function compute_jacobian_triad(p, param_indices::Vector{Int}, observables; ε=1e-5)
    M, N = length(observables), length(param_indices)
    J = zeros(Float64, M, N)
    for (j, idx) in enumerate(param_indices)
        p_plus, p_minus = copy(p), copy(p)
        p_plus[idx]  += ε
        p_minus[idx] -= ε
        obs_p = compute_observables_triad(p_plus,  observables)
        obs_m = compute_observables_triad(p_minus, observables)
        @inbounds J[:, j] = (obs_p - obs_m) ./ (2ε)
    end
    return J
end

# ---------------- Fast precompute of s, s', and B-arrays ----------------

# Evaluate score and derivative once on the whole series (threaded)
function precompute_s_sprime(xs::AbstractVector{<:Real}, score::Function, sprime::Function)
    N = length(xs)
    svals  = Vector{Float64}(undef, N)
    sprims = Vector{Float64}(undef, N)
    @threads for i in eachindex(xs)
        x = Float64(xs[i])
        svals[i]  = score(x)
        sprims[i] = sprime(x)
    end
    return svals, sprims
end

# Build B_A, B_B, B_s in-place-friendly vectorized form
function build_B_arrays(xs::Vector{Float64},
                        svals::Vector{Float64},
                        sprims::Vector{Float64},
                        A::Float64, B::Float64, s_noise::Float64)
    @assert length(xs) == length(svals) == length(sprims)
    N = length(xs)
    tmp = similar(xs);  @. tmp = svals^2 + sprims

    BA = similar(xs);   @. BA = 2B*svals - (A - B*xs)*tmp
    BB = similar(xs);   @. BB = -2B - 2*svals*(-A + 2B*xs) - tmp*(-A*xs + B*xs^2)
    Bs = similar(xs);   @. Bs = -s_noise*tmp
    return BA, BB, Bs
end

# ---------------- Response Matrix from precomputed arrays ----------------

# Build observable time series once, then cross-cov vs B-arrays.
# Uses StatsBase.crosscov (demean=true) to preserve your original normalization.
function create_response_matrix_from_arrays(xs::Vector{Float64}, dt::Float64,
                                            observables::Vector{<:Function},
                                            B_arrays::NTuple{3,Vector{Float64}};
                                            max_lag_time::Float64=30.0)
    K = length(observables)
    lag_idx = 0:floor(Int, max_lag_time/dt)
    nlag = length(lag_idx)
    R  = zeros(Float64, K, 3)
    Rt = zeros(Float64, K, 3, nlag)

    # Precompute observables over the series (threaded)
    obs_ts = Vector{Vector{Float64}}(undef, K)
    @threads for k in 1:K
        f = observables[k]
        y = Vector{Float64}(undef, length(xs))
        @inbounds @simd for i in eachindex(xs)
            y[i] = f(xs[i])
        end
        obs_ts[k] = y
    end

    # Correlations (threaded over observables)
    BA, BB, Bs = B_arrays
    allB = (BA, BB, Bs)

    @threads for k in 1:K
        tsA = obs_ts[k]
        for (j, tsB) in enumerate(allB)
            covs = .-crosscov(tsA, tsB, lag_idx; demean=true)  # GFDT sign
            @inbounds Rt[k, j, :] = covs
            # trapezoid integral over lags
            integ = dt * (sum(covs) - 0.5*(covs[1] + covs[end]))
            R[k, j] = integ
        end
    end
    return R, Rt, (lag_idx .* dt)
end

# Type-stable wrapper; try batched evaluation if available
function eval_score_series(raw, xs::Vector{Float64})
    try
        y = raw(xs)                         # if implementation supports vector input
        return Float64.(y)
    catch
        return map(x -> Float64(_to_scalar(raw(x))), xs)
    end
end

##
# ---------------- Main Execution ----------------

println("Generating baseline time series...")
dt_sim  = 0.01
N_steps = 20_000_000  # heavy; consider decimation once validated
ts = evolve([0.0], dt_sim, N_steps, F, sigma1, sigma2; timestepper=:rk4)[1, :]
println("Data generation complete.")

# Observables (edit as needed)
observables = [x -> x^3]   # currently one observable; add x->x, x->x^2 for mean/variance

# ---- A) TRUE score path (exact) ----
println("\n--- Computing R_true (fast precompute path) ---")
true_s, true_sp = create_true_score_and_derivative_triad(params_triad)
svals_true, sprims_true = precompute_s_sprime(ts, true_s, true_sp)
BA_true, BB_true, Bs_true = build_B_arrays(ts, svals_true, sprims_true, params_triad[5], params_triad[6], params_triad[7])
R_true, R_true_t, lags = create_response_matrix_from_arrays(ts, dt_sim, observables, (BA_true, BB_true, Bs_true))

# ---- B) LINEAR (Gaussian) score path ----
println("\n--- Computing R_linear (fast precompute path) ---")
lin_s, lin_sp = create_linear_s_ds(ts)
svals_lin, sprims_lin = precompute_s_sprime(ts, lin_s, lin_sp)
BA_lin, BB_lin, Bs_lin = build_B_arrays(ts, svals_lin, sprims_lin, params_triad[5], params_triad[6], params_triad[7])
R_linear, R_linear_t, _ = create_response_matrix_from_arrays(ts, dt_sim, observables, (BA_lin, BB_lin, Bs_lin))

# ---- C) CLUSTERED (KGMM) score path (optional) ----
println("\n--- Computing R_clustered (fast precompute path) ---")
kgmm = calculate_score_kgmm(reshape(ts, 1, :); σ_value=0.05, clustering_prob=0.0005, verbose=false)
raw_score = kgmm.score_function

svals_cl = eval_score_series(raw_score, ts)
sprime_cl = construct_divergence_score(x->Float64(_to_scalar(raw_score(x))))
sprims_cl = [sprime_cl(x) for x in ts]     # vectorized interp, very fast
BA_cl, BB_cl, Bs_cl = build_B_arrays(ts, svals_cl, sprims_cl, params_triad[5], params_triad[6], params_triad[7])
R_clustered, R_clustered_t, _ = create_response_matrix_from_arrays(ts, dt_sim, observables, (BA_cl, BB_cl, Bs_cl))

# ---- D) Analytic Jacobian for comparison ----
println("\n--- Computing J_analytical ---")
param_indices = [5, 6, 7]  # A, B, s_noise
J_analytical = compute_jacobian_triad(params_triad, param_indices, observables; ε=1e-3)

println("\n--- COMPARISON ---")
println("R_linear   = ", R_linear)
println("R_clustered = ", R_clustered)
println("R_true     = ", R_true)
println("J_analytical= ", J_analytical)
println("-------------------------------------------------\n")

# Example: inspect one response kernel (obs #1 vs param #3 = s_noise)
# (Uncomment plotting only if you need it — plotting packages are heavy.)
# using Plots
# idx_obs = 1
# idx_param = 3
# plot(lags, vec(R_true_t[idx_obs, idx_param, :]), label="True")
# plot!(lags, vec(R_linear_t[idx_obs, idx_param, :]), label="Linear")
# plot!(xlabel="lag", ylabel="kernel")

#################################################################
