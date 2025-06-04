# MAIN SCRIPT — Score Function Training from Clustering (formerly run_experiments)

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using MarkovChainHammer
using ClustGen
using KernelDensity
using HDF5
using Flux
using BSON
using Plots
using LinearAlgebra
using ProgressBars
using Distributions
using QuadGK
using LaTeXStrings
using StatsBase


########### USEFUL FUNCTIONS ###########
function F(x, t, σ, ε ; µ=10.0, ρ=28.0, β=8/3)
    dx = x[1] * (1 - x[1]^2) + (σ / ε) * x[3]
    dy1 = µ/ε^2 * (x[3] - x[2])
    dy2 = 1/ε^2 * (x[2] * (ρ - x[4]) - x[3])
    dy3 = 1/ε^2 * (x[2] * x[3] - β * x[4])
    return [dx, dy1, dy2, dy3]
end

function sigma(x, t; noise = 0.0)
    sigma1 = noise
    sigma2 = noise
    sigma3 = noise
    sigma4 = noise #Added: This is for the 4th variable
    return [sigma1, sigma2, sigma3, sigma4]
end

function normalize_f(f, x, t, M, S)
    return f(x .* S .+ M, t) .* S
end
########### END USEFUL FUNCTIONS ###########




# Parameters
fix_initial_state = false
σ=0.08
ε=0.5
save_figs = false
dim = 4 # Number of dimensions in the system

########## 1. Simulate System ##########
dt = 0.01
Nsteps = 100000000
f(x, t) = F(x, t, σ, ε)
obs_nn = evolve(randn(4), dt, Nsteps, f, sigma; timestepper=:rk4, resolution=10)
#obs_uncorr = obs_nn[1:1, 1:1:end]

########## 2. Normalize and autocovariance ##########
M = mean(obs_nn, dims=2)[1]
S = std(obs_nn, dims=2)[1]
obs = (obs_nn[1:1,:] .- M) ./ S

autocov_obs = autocovariance(obs[1, 1:100000]; timesteps=500)
kde_obs = kde(obs[1, :])

autocov_obs_nn = zeros(4, 100)

for i in 1:4
    autocov_obs_nn[i, :] = autocovariance(obs_nn[i, :]; timesteps=100)
end

D_eff = dt * (0.5 * autocov_obs_nn[3, 1] + sum(autocov_obs_nn[3, 2:end-1]) + 0.5 * autocov_obs_nn[3, end])
D_eff = 0.3
@show D_eff

# plt_12 = plot(autocov_obs_nn[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of x")

#compute tau from autocovariance 
# function estimate_tau(y, dt; threshold=0.2)
#     y_centered = y .- mean(y)
#     acf = autocor(y_centered)
#     for i in 2:length(acf)
#         if abs(acf[i]) < threshold
#             return (i - 1) * dt, acf
#         end
#     end
#     return (length(acf) - 1) * dt, acf
# end
# # Applichiamolo alla terza variabile (y₂)
# τ_y2, acf_y2 = estimate_tau(obs_nn[3, :], dt)


#training and clustering parameters 
σ_value=0.05
prob=0.001
conv_param=0.02
n_epochs=5000
batch_size=16


########## 3. Clustering ##########
averages, centers, Nc, labels = f_tilde_labels(σ_value, obs[:,1:10:end]; prob=prob, do_print=false, conv_param=conv_param, normalization=false)
inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)

########## 4. Score Functions ##########

#analytic score function
f1(x,t) = x .- x.^3
score_true(x, t) = normalize_f(f1, x, t, M, S)

#learned score function
#kde_x = kde(obs_nn[1, :])
centers_sorted_indices = sortperm(centers[1, :])
centers_sorted = centers[:, centers_sorted_indices][:]
scores = .- averages[:, centers_sorted_indices][:] ./ σ_value

########## 5. Train NN ##########
@time nn, losses = train(inputs_targets, n_epochs, batch_size, [1, 50, 25, 1];
    opt=Flux.Adam(0.001), activation=swish, last_activation=identity,
    use_gpu=false)

nn_clustered_cpu = nn |> cpu
score_clustered(x) = .- nn_clustered_cpu(reshape(Float32[x...], :, 1))[:] ./ σ_value

########## 6. Compute PDF ##########
function true_pdf_normalized(x)
    x_phys = x .* S[1] .+ M[1]
    U = .-0.5 .* x_phys.^2 .+ 0.25 .* x_phys.^4
    p = exp.(-2 .* U ./ D_eff)
    return p ./ S[1]
end

xax = [-1.25:0.005:1.25...]
xax_2 = [-1.6:0.02:1.6...]
interpolated_score = [score_clustered(xax[i])[1] for i in eachindex(xax)]
true_score = [2 * score_true(xax[i], 0.0)[1] / D_eff for i in eachindex(xax)]
pdf_interpolated_norm = compute_density_from_score(xax_2, score_clustered)
pdf_true = true_pdf_normalized(xax_2)
scale_factor = maximum(kde_x.density) / maximum(pdf_true)
pdf_true .*= scale_factor

########## 6b. Compare y2 distribution with Gaussian ##########
# y2_samples = obs_nn[3, 200:end] #occhio a 200:end qui
# kde_y2 = kde(y2_samples)
# μ_y2 = mean(y2_samples)
# σ_y2 = std(y2_samples)
# gauss_y2(x) = pdf(Normal(μ_y2, σ_y2), x)
# xax_y2 = kde_y2.x
# dx = xax_y2[2] - xax_y2[1]
# pdf_kde = kde_y2.density ./ (sum(kde_y2.density) * dx)
# pdf_gaussian_y2 = [gauss_y2(x) for x in xax_y2]
# pdf_gaussian_y2 ./= sum(pdf_gaussian_y2) * dx


########## 6c. Compute ACF from Langvein equation dx = s(x) +  \sqrt(2)ξ(t) without Phi ##########
# score_only_xt(x, t) = score_clustered(x)
# sigma_plain(x, t) = [sqrt(2.0);;]  

# trj_score_only = evolve([0.0], dt, Nsamples, score_only_xt, sigma_plain;
#                         timestepper=:euler, resolution=1)
# auto_score_only = autocovariance(trj_score_only[1, 1:res:end]; timesteps=tsteps)


########## Phi calculation ##########
dt = 0.1
#rate matrix
Q = generator(labels; dt=dt)*0.16
P_steady = steady_state(Q)

#test if Q approximates well the dynamics
tsteps = 51
res = 10

auto_obs = autocovariance(obs[1:res:end]; timesteps=tsteps) 
auto_Q = autocovariance(centers[1,:], Q, [0:dt*res:Int(res * (tsteps-1) * dt)...])


plt = Plots.plot(auto_obs)
plt = Plots.plot!(auto_Q)

#compute the score function
gradLogp = - averages ./ σ_value


#compute Phi and Σ
M_Q = centers * Q * (centers *Diagonal(P_steady))'
V_Q = gradLogp * (centers * Diagonal(P_steady))'
Φ = (M_Q * inv(V_Q))[1,1]
Σ = sqrt(Φ)


########## Test effective dynamics ##########
score_clustered_xt(x, t) = Φ * score_clustered(x)
# score_clustered_xt(x, t) = begin
#     s = score_clustered(x)
#     drift = Φ * s
#     if any(isnan.(drift)) || any(abs.(drift) .> 1e4)
#         @warn "🚨 Drift exploded" x s drift
#     end
#     return drift
# end

 



# Diffusion coefficient √(2Φ)
#τ_y2 = 2
#sigma_Langevin(x, t) = Σ / sqrt(2 * τ_y2) questa servirà poin
sigma_Langevin(x, t) = Σ 
# @show τ_y2
# @show Σ
# @show sigma_Langevin
# @show maximum(abs, Σ / sqrt(2 * τ_y2))

# Simulate Langevin dynamics
Nsamples = 10000000
dt = 0.001
trj_langevin = evolve([0.0], dt, Nsamples, score_clustered_xt, sigma_Langevin;
                      timestepper=:euler, resolution=100)

dt = 0.1
# PDF of Langevin trajectory
kde_langevin = kde(trj_langevin[1, :])


# Autocovariance of Langevin trajectory vs observed
auto_langevin = autocovariance(trj_langevin[1, 1:res:end]; timesteps=tsteps)

########## 7. Plotting ##########
Plots.default(fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10, legendfontsize=10)
plotlyjs()



#Plot PDF
p_pdf = plot(kde_obs.x, kde_obs.density, label="Observed", lw=2, color=:red)
plot!(p_pdf, kde_langevin.x, kde_langevin.density, label="Langevin", lw=2, color=:blue)
xlabel!("x"); ylabel!("Density"); title!("PDF comparison")
plot!(p_pdf, xax_2, pdf_true; label="PDF analytic", linewidth=2, linestyle=:dash, color=:lime)
# plot!(p_pdf, xax_2, pdf_interpolated_norm; label="PDF learned", linewidth=2,color=:cyan)


#Plot autocovariance
p_acf = plot(auto_obs, label="", lw=2, color=:red)
xlabel!("Lag"); ylabel!("Autocorrelation"); title!("Autocorrelation: NN vs Observed")
plot!(p_acf, auto_langevin, label="", lw=2, color=:blue)
xlabel!("Time steps"); ylabel!("Autocorrelation"); title!("Autocorrelation comparison")

plot!(p_pdf, [NaN], [NaN], xlabel="\n\n")
plt = plot(p_pdf, p_acf, layout=(2, 1), size=(800, 600))
display(plt)
#plot pdf pdf_gaussian_y2
# p_y2 = plot(kde_y2.x, pdf_kde; label="PDF of y2(t)", xlabel="y2", ylabel="Density", title="Distribution of y2(t)", linewidth=2)
# plot!(p_y2, xax_y2, pdf_gaussian_y2; label="Gaussian fit", linewidth=2)

#Plot Score
p_score = scatter(centers_sorted, scores; color=:blue, alpha=0.2, label="Cluster centers",
    xlims=(-1.3, 1.3), ylims=(-5, 5), xlabel="𝑥", ylabel="Score(𝑥)", title="Score Function Estimate")
plot!(p_score, xax, interpolated_score; label="NN interpolation", linewidth=2, color=:red)
plot!(p_score, xax, true_score; label="Score analytic", linewidth=2, color=:lime)

########## 8. Save ##########
if save_figs
    base_path = "/Users/giuliodelfelice/Desktop/MIT"
    test_folder = joinpath(base_path, "TESTS")
    mkpath(test_folder)
    savefig(p_score, joinpath(test_folder, "Interpolation.pdf"))
    savefig(p_pdf, joinpath(test_folder, "PDFs.pdf"))
    savefig(p_loss, joinpath(test_folder, "loss_plot.pdf"))
    savefig(p_acf, joinpath(test_folder, "y2_pdf.pdf"))

else
    display(p_score)
    display(p_pdf)
    display(p_loss)
    display(p_acf)
end










########### VERIFICA ###########
tsteps = 100
res = 10

acf_real = autocovariance(obs[1:res:end]; timesteps=tsteps)
acf_sim = autocovariance(trj_langevin[1, 1:res:end]; timesteps=tsteps)

acf_real_norm = acf_real ./ acf_real[1]
acf_sim_norm = acf_sim ./ acf_sim[1]

using LsqFit

function exp_model(t, p)
    A, τ = p
    return A .* exp.(-t ./ τ)
end

lags = collect(0:tsteps-1)
fit_result = curve_fit(exp_model, lags, acf_sim_norm, [1.0, 1.0])  # init: A=1, τ=1
params = fit_result.param
A_fit, τ_sim = params
@show A_fit, τ_sim

τ_real, _ = estimate_tau(obs_nn[1, :], dt)
@show τ_real

println("τ_real  = ", τ_real)
println("τ_sim   = ", τ_sim)
println("Φ_eff_real = ", 1 / τ_real)
println("Φ_eff_sim  = ", 1 / τ_sim)


#Plot for verification
plot(lags, acf_real_norm, label="Real", lw=2, color=:blue)
plot!(lags, acf_sim_norm, label="Simulated", lw=2, color=:red)
plot!(lags, exp_model(lags, params), label="Fit A·exp(-t/τ)", lw=2, linestyle=:dash, color=:black)
xlabel!("Lag")
ylabel!("Autocorrelation")
title!("ACF: real vs simulated")


                      
