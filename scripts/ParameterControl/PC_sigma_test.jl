
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using ClustGen          # `evolve` integrator
using KernelDensity     # KDE for visual PDF comparison
using Statistics
using Plots
##
# -------------------------------------------------------------
#  Model definition
# -------------------------------------------------------------

# Drift parameters (cubic polynomial + constant forcing)
a = -0.0222; b = -0.2; c = 0.0494; F̃ = 0.6

# Constant noise amplitude  σ = s0 / √2 so that g² = σ²
s0    = 0.7071
σ₀    = s0 / √2          # g(x) = σ₀  (constant)
g2    = σ₀^2             # g²

# Relative perturbation on the diffusion term
δ = 0.1

# --- Drift (scalar) ---------------------------------------------------------
F(x) = F̃ + a*x + b*x^2 - c*x^3              # scalar function
F_vec(x, t) = (F(x[1]),)                    # ClustGen expects a tuple / small vector

# --- Constant diffusion -----------------------------------------------------
σ_vec(x, t)      = σ₀                       # original noise (scalar)
σ_pert_vec(x, t) = (1 + δ)*σ₀              # perturbed noise amplitude

# -------------------------------------------------------------
#  Analytical ∂ₓ log p and PF‑ODE drift correction
# -------------------------------------------------------------
# For constant diffusion (1‑D):   ∂ₓ log p(x) =  2 F(x) / g²
∂logp(x) = 2*F(x) / g2                      # analytical gradient

# First‑order drift correction   δf(x) = −δ g² ∂ₓ log p(x) = −2δ F(x)
δf(x) = -2*δ*F(x)

# Deterministic PF‑ODE drift (no stochastic term)
F_PF_vec(x, t) = (F(x[1]) + δf(x[1]),)      # = (1−2δ)·F(x)

# -------------------------------------------------------------
#  Integration settings
# -------------------------------------------------------------

dt_sim = 0.01          # internal SDE timestep
res    = 10            # keep every `res` steps
Nsteps = 20_000_000    # length of simulation (feel free to shorten while testing)

println("Simulating SDE with perturbed diffusion …")
obs_σpert = evolve([0.0], dt_sim, Nsteps, F_vec, σ_pert_vec;
                   timestepper = :euler, resolution = res)[1, :]

println("Simulating PF‑ODE with modified drift …")
obs_PF = evolve([0.0], dt_sim, Nsteps, F_PF_vec, σ_vec;
                timestepper = :euler, resolution = res)[1, :]

# -------------------------------------------------------------
#  Density estimation & visual comparison
# -------------------------------------------------------------
println("Estimating PDFs …")
kd_σpert = kde(obs_σpert)
kd_PF    = kde(obs_PF)

xgrid = range(minimum(vcat(kd_σpert.x, kd_PF.x)),
              maximum(vcat(kd_σpert.x, kd_PF.x)); length=1000)

plot(xgrid, pdf(kd_σpert, xgrid); label="Perturbed σ", linewidth=2)
plot!(xgrid, pdf(kd_PF,    xgrid); label="PF‑ODE",     linewidth=2)
plot!(xlabel="x", ylabel="Density", title="PDF comparison: σ‑perturbed vs PF‑ODE")
