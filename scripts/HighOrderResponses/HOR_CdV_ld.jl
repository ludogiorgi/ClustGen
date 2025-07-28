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

##

# Pre-computed coefficients function
function compute_coeffs(m, b, beta, gamma)
    π = pi
    
    # Eq. 2.7 from the image
    α = (8 * sqrt(2) / π) * (m^2 / (4m^2 - 1)) * ((b^2 + m^2 - 1) / (b^2 + m^2))
    β = (beta * b^2) / (b^2 + m^2)
    δ = (64 * sqrt(2) / (15π)) * ((b^2 - m^2 + 1) / (b^2 + m^2))
    γ̃ = gamma * (4m / (4m^2 - 1)) * (sqrt(2) * b / π)
    ε = 16 * sqrt(2) / (5π)
    γ = gamma * (4m^3 / (4m^2 - 1)) * (sqrt(2) * b / (π * (b^2 + m^2)))

    return (α=α, β=β, δ=δ, γ̃=γ̃, ε=ε, γ=γ)
end

b = 1.6
beta = 1.25
gamma = 0.2
C = 0.1
x1_star = 0.95
x4_star = -0.76095

coeffs1 = compute_coeffs(1.0, b, beta, gamma)
coeffs2 = compute_coeffs(2.0, b, beta, gamma)

# Extract all coefficients ahead of time
α1, β1, δ1, γ̃1, ε, γ1 = coeffs1.α, coeffs1.β, coeffs1.δ, coeffs1.γ̃, coeffs1.ε, coeffs1.γ
α2, β2, δ2, γ̃2, _, γ2 = coeffs2.α, coeffs2.β, coeffs2.δ, coeffs2.γ̃, coeffs2.ε, coeffs2.γ

# Print all calculated coefficients
println("\n=== Model Coefficients ===")
println("\nParameters:")
println("b = $b, beta = $beta, gamma = $gamma, C = $C")
println("x1_star = $x1_star, x4_star = $x4_star")

println("\nFor m = 1:")
println("α1 = $α1")
println("β1 = $β1")
println("δ1 = $δ1")
println("γ̃1 = $γ̃1")
println("γ1 = $γ1")

println("\nFor m = 2:")
println("α2 = $α2")
println("β2 = $β2")
println("δ2 = $δ2")
println("γ̃2 = $γ̃2")
println("γ2 = $γ2")

println("\nShared coefficient:")
println("ε = $ε")


##

# Create a function that returns an optimized F with pre-computed coefficients
function create_optimized_F(b=1.6, beta=1.25, gamma=0.2, C=0.1, x1_star=0.95, x4_star=-0.76095)
    # Pre-compute coefficients only once
    coeffs1 = compute_coeffs(1.0, b, beta, gamma)
    coeffs2 = compute_coeffs(2.0, b, beta, gamma)
    
    # Extract all coefficients ahead of time
    α1, β1, δ1, γ̃1, ε, γ1 = coeffs1.α, coeffs1.β, coeffs1.δ, coeffs1.γ̃, coeffs1.ε, coeffs1.γ
    α2, β2, δ2, γ̃2, _, γ2 = coeffs2.α, coeffs2.β, coeffs2.δ, coeffs2.γ̃, coeffs2.ε, coeffs2.γ
    
    # Return a closure that captures the pre-computed coefficients
    return function F_fast(x, t)
        # Unpack state variables
        x1, x2, x3, x4, x5, x6 = x
        
        # Compute the derivatives using pre-calculated coefficients
        dx1 = γ̃1 * x3 - C * (x1 - x1_star)
        dx2 = -(α1 * x1 - β1) * x3 - C * x2 - δ1 * x4 * x6
        dx3 = (α1 * x1 - β1) * x2 - γ1 * x1 - C * x3 + δ1 * x4 * x5
        dx4 = γ̃2 * x6 - C * (x4 - x4_star) + ε * (x2 * x6 - x3 * x5)
        dx5 = -(α2 * x1 - β2) * x6 - C * x5 - δ2 * x4 * x3
        dx6 = (α2 * x1 - β2) * x5 - γ2 * x4 - C * x6 + δ2 * x4 * x2
        
        return [dx1, dx2, dx3, dx4, dx5, dx6]
    end
end

F_fast = create_optimized_F()

function sigma(x, t; noise = 0.01)
    return noise
end

function normalize_f(f, x, t, M, S)
    return f(x .* S .+ M, t) .* S
end

dim = 6
dt = 1.0
Nsteps = 1000000
obs_nn = evolve(0.1 .* randn(dim), dt, Nsteps, F_fast, sigma; resolution = 1)

M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn[:,1000:end] .- M) ./ S


# Plots.plot(obs_nn[2,1000:1:2000])
# ##
# kde_true_12 = kde((obs[1,:], obs[2,:]))
# kde_true_13 = kde((obs[1,:], obs[3,:]))
# kde_true_14 = kde((obs[1,:], obs[4,:]))

# plt1 = Plots.heatmap(kde_true_12.x, kde_true_12.y, kde_true_12.density, xlabel="X", ylabel="Y", title="True PDF")
# plt2 = Plots.heatmap(kde_true_13.x, kde_true_13.y, kde_true_13.density, xlabel="X", ylabel="Y", title="True PDF")
# plt3 = Plots.heatmap(kde_true_14.x, kde_true_14.y, kde_true_14.density, xlabel="X", ylabel="Y", title="True PDF")
# Plots.plot(plt1, plt2, plt3, layout=(1, 3), size=(1200, 400))

# ##
# kde_true_1 = kde(obs[1,:])
# kde_true_2 = kde(obs[2,:])
# kde_true_3 = kde(obs[3,:])
# kde_true_4 = kde(obs[4,:])
# kde_true_5 = kde(obs[5,:])
# kde_true_6 = kde(obs[6,:])

# plt1 = Plots.plot(kde_true_1.x, kde_true_1.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
# plt2 = Plots.plot(kde_true_2.x, kde_true_2.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
# plt3 = Plots.plot(kde_true_3.x, kde_true_3.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
# plt4 = Plots.plot(kde_true_4.x, kde_true_4.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
# plt5 = Plots.plot(kde_true_5.x, kde_true_5.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
# plt6 = Plots.plot(kde_true_6.x, kde_true_6.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")

# Plots.plot(plt1, plt2, plt3, plt4, plt5, plt6, layout=(2, 3), size=(1200, 800))

# ##

autocov_obs = zeros(dim, 300)
for i in 1:dim
    autocov_obs[i,:] = autocovariance(obs[i,1:100000]; timesteps=300)
end

autocov_obs_mean = mean(autocov_obs, dims=1)

plotly()
Plots.plot(autocov_obs_mean[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")
##
obs_uncorr = obs[:, 1:1:end]

plotly()
Plots.scatter(obs_uncorr[1,1:100:100000], obs_uncorr[2,1:100:100000], obs_uncorr[3,1:100:100000], markersize=1, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

normalization = false
σ_value = 0.05

averages, _, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.000025, do_print=true, conv_param=0.002, normalization=normalization)

if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
end

targets_norm = [norm(averages[:,i]) for i in eachindex(centers[1,:])]
Plots.scatter(centers[1,:], centers[2,:], marker_z=targets_norm, color=:viridis)

##
#################### TRAINING WITH CLUSTERING LOSS ####################

@time nn_clustered, loss_clustered = train(inputs_targets, 300, 128, [dim, 128, 64, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value
plt = Plots.plot(loss_clustered)
#savefig(plt, "figures/HOR_figures/CdV_clustered_loss.png")

##
cluster_loss = check_loss(obs_uncorr[:, 1000:1000:end], nn_clustered_cpu, σ_value)
##
#################### TRAINING WITH VANILLA LOSS ####################

@time nn_vanilla, loss_vanilla = train(obs_uncorr[:,1:10:end], 300, 128, [dim, 128, 64, dim], σ_value; use_gpu=true, opt=Adam(0.001))
nn_vanilla_cpu = nn_vanilla |> cpu
score_vanilla(x) = .- nn_vanilla_cpu(Float32.([x...])) ./ σ_value
plt = Plots.plot(loss_vanilla)
savefig(plt, "figures/HOR_figures/CdV_vanilla_loss.png")
##
#################### SAMPLES GENERATION ####################

score_gen(x) = score_clustered(x)

score_gen_xt(x,t) = score_gen(x)
sigma_I(x,t) = 1.0

trj_clustered = evolve(zeros(dim), 0.002*dt, 100000000, score_gen_xt, sigma_I; timestepper=:euler, resolution=10, boundary=[-100,100])
# trj_score = evolve([0.0, 0.0], dt, 1000000, score_true, sigma_I; timestepper=:rk4, resolution=10, boundary=[-100,100])

kde_clustered_1 = kde(trj_clustered[1,:])
kde_true_1 = kde(obs[1,:])
kde_clustered_2 = kde(trj_clustered[2,:])
kde_true_2 = kde(obs[2,:])
kde_clustered_3 = kde(trj_clustered[3,:])
kde_true_3 = kde(obs[3,:])
kde_clustered_4 = kde(trj_clustered[4,:])
kde_true_4 = kde(obs[4,:])
kde_clustered_5 = kde(trj_clustered[5,:])
kde_true_5 = kde(obs[5,:])
kde_clustered_6 = kde(trj_clustered[6,:])
kde_true_6 = kde(obs[6,:])

##
gr()
plt1 = Plots.plot(kde_true_1.x, kde_true_1.density, label="Observed", xlabel="X", ylabel="Density", title="PDF 1")
plt1 = Plots.plot!(kde_clustered_1.x, kde_clustered_1.density, label="Generated", xlabel="X", ylabel="Density")
plt2 = Plots.plot(kde_true_2.x, kde_true_2.density, label="Observed", xlabel="X", ylabel="Density", title="PDF 2")
plt2 = Plots.plot!(kde_clustered_2.x, kde_clustered_2.density, label="Generated", xlabel="X", ylabel="Density")
plt3 = Plots.plot(kde_true_3.x, kde_true_3.density, label="Observed", xlabel="X", ylabel="Density", title="PDF 3")
plt3 = Plots.plot!(kde_clustered_3.x, kde_clustered_3.density, label="Generated", xlabel="X", ylabel="Density")
plt4 = Plots.plot(kde_true_4.x, kde_true_4.density, label="Observed", xlabel="X", ylabel="Density", title="PDF 4")
plt4 = Plots.plot!(kde_clustered_4.x, kde_clustered_4.density, label="Generated", xlabel="X", ylabel="Density")
plt5 = Plots.plot(kde_true_5.x, kde_true_5.density, label="Observed", xlabel="X", ylabel="Density", title="PDF 5")
plt5 = Plots.plot!(kde_clustered_5.x, kde_clustered_5.density, label="Generated", xlabel="X", ylabel="Density")
plt6 = Plots.plot(kde_true_6.x, kde_true_6.density, label="Observed", xlabel="X", ylabel="Density", title="PDF 6")
plt6 = Plots.plot!(kde_clustered_6.x, kde_clustered_6.density, label="Generated", xlabel="X", ylabel="Density")
plt = Plots.plot(plt1, plt2, plt3, plt4, plt5, plt6, layout=(2, 3), size=(1200, 800))
# savefig(plt, "figures/HOR_figures/CdV_univariate_clustered.png")
##
# gr()
# # Compute bivariate PDFs for consecutive variables
# kde_true_12 = kde((obs[1,:], obs[2,:]))
# kde_clustered_12 = kde((trj_clustered[1,1:100:end], trj_clustered[2,1:100:end]))
# kde_true_13 = kde((obs[1,:], obs[3,:]))
# kde_clustered_13 = kde((trj_clustered[1,1:100:end], trj_clustered[3,1:100:end]))
# kde_true_14 = kde((obs[1,:], obs[4,:]))
# kde_clustered_14 = kde((trj_clustered[1,1:100:end], trj_clustered[4,1:100:end]))
# kde_true_15 = kde((obs[1,:], obs[5,:]))
# kde_clustered_15 = kde((trj_clustered[1,1:100:end], trj_clustered[5,1:100:end]))
# kde_true_16 = kde((obs[1,:], obs[6,:]))
# kde_clustered_16 = kde((trj_clustered[1,1:100:end], trj_clustered[6,1:100:end]))

# plt1 = Plots.heatmap(kde_true_12.x, kde_true_12.y, kde_true_12.density, xlabel="X", ylabel="Y", title="True PDF 1-2")
# plt2 = Plots.heatmap(kde_clustered_12.x, kde_clustered_12.y, kde_clustered_12.density, xlabel="X", ylabel="Y", title="Generated PDF 1-2")
# plt3 = Plots.heatmap(kde_true_13.x, kde_true_13.y, kde_true_13.density, xlabel="X", ylabel="Y", title="True PDF 1-3")
# plt4 = Plots.heatmap(kde_clustered_13.x, kde_clustered_13.y, kde_clustered_13.density, xlabel="X", ylabel="Y", title="Generated PDF 1-3")
# plt5 = Plots.heatmap(kde_true_14.x, kde_true_14.y, kde_true_14.density, xlabel="X", ylabel="Y", title="True PDF 1-4")
# plt6 = Plots.heatmap(kde_clustered_14.x, kde_clustered_14.y, kde_clustered_14.density, xlabel="X", ylabel="Y", title="Generated PDF 1-4")
# plt7 = Plots.heatmap(kde_true_15.x, kde_true_15.y, kde_true_15.density, xlabel="X", ylabel="Y", title="True PDF 1-5")
# plt8 = Plots.heatmap(kde_clustered_15.x, kde_clustered_15.y, kde_clustered_15.density, xlabel="X", ylabel="Y", title="Generated PDF 1-5")
# plt9 = Plots.heatmap(kde_true_16.x, kde_true_16.y, kde_true_16.density, xlabel="X", ylabel="Y", title="True PDF 1-6")
# plt10 = Plots.heatmap(kde_clustered_16.x, kde_clustered_16.y, kde_clustered_16.density, xlabel="X", ylabel="Y", title="Generated PDF 1-6")
# plt = Plots.plot(plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8, plt9, plt10, layout=(5, 2), size=(1200, 1200))
# # savefig(plt, "figures/HOR_figures/CdV_bivariate5.png")
# ##

using ClustGen
f(t) = 1.0

res_trj = 1
steps_trj = 100000
trj = obs[:,1:res_trj:steps_trj*res_trj]

ϵ = 0.01

function u(x)
    U = zeros(dim)
    U[1] = ϵ                          
    return U
end

div_u(x) = 0.0
invC0 = inv(cov(obs'))
score_qG(x) = - invC0* (x)

dim_Obs = 6
n_tau = 40

R_num, δObs_num = zeros(4, dim_Obs, n_tau+1), zeros(4, dim_Obs, n_tau+1)
R_lin, δObs_lin = zeros(4, dim_Obs, n_tau+1), zeros(4, dim_Obs, n_tau+1)
R_gen, δObs_gen = zeros(4, dim_Obs, n_tau+1), zeros(4, dim_Obs, n_tau+1)

R_num[1,:,:], R_num[2,:,:], R_num[3,:,:], R_num[4,:,:] = generate_numerical_response_HO(F_fast, u, dim, dt, n_tau, 1000, sigma, M; n_ens=100000, resolution=res_trj, timestepper=:rk4)

for i in 1:4
    Obs(x) = x .^i
    R_lin[i,:,:], δObs_lin[i,:,:] = generate_score_response(trj, u, div_u, f, score_qG, res_trj*dt, n_tau, Obs, dim_Obs)
    R_gen[i,:,:], δObs_gen[i,:,:] = generate_score_response(trj, u, div_u, f, score_gen, res_trj*dt, n_tau, Obs, dim_Obs)
end

##

R0_gen = zeros(dim,dim)
for j in 1:steps_trj
    R0_gen .+= trj[:,j] * score_gen(trj[:,j])'
end
R0_gen ./= (steps_trj)
invR0_gen = .-inv(R0_gen)
    
R_gen_hack = zeros(4, dim, n_tau+1)
for i in 1:4
    for j in 1:n_tau+1
        R_gen_hack[i,:,j] = R_gen[i,:,j]'*invR0_gen 
    end
end

##
############## OBSERVATIONS GENERATION ####################

obs_trj = evolve(zeros(dim), dt, 100000, F_fast, sigma; timestepper=:rk4)
obs_trj = (obs_trj .- M) ./ S
score_trj = evolve(zeros(dim), 0.01*dt, 10000000, score_gen_xt, sigma_I; timestepper=:rk4, resolution=100)
##
# Save the results to an HDF5 file

save_variables_to_hdf5("data/HOR_data/CdV_results_ld_0.05-0.00001-128-300_ld.h5", Dict(
    "R_num" => R_num,
    "R_lin" => R_lin,
    "R_gen" => R_gen,
    "S" => S,
    "M" => M,
    "obs_trj" => obs_trj,
    "score_trj" => score_trj,
    "σ_value" => σ_value,
    "averages" => averages,
    "centers" => centers,
    "Nc" => Nc,
    "kde_clustered_1_x" => [kde_clustered_1.x...],
    "kde_clustered_1_density" => kde_clustered_1.density,
    "kde_true_1_x" => [kde_true_1.x...],
    "kde_true_1_density" => kde_true_1.density,
    "kde_clustered_2_x" => [kde_clustered_2.x...],
    "kde_clustered_2_density" => kde_clustered_2.density,
    "kde_true_2_x" => [kde_true_2.x...],
    "kde_true_2_density" => kde_true_2.density,
    "kde_clustered_3_x" => [kde_clustered_3.x...],
    "kde_clustered_3_density" => kde_clustered_3.density,
    "kde_true_3_x" => [kde_true_3.x...],
    "kde_true_3_density" => kde_true_3.density,
    "kde_clustered_4_x" => [kde_clustered_4.x...],
    "kde_clustered_4_density" => kde_clustered_4.density,
    "kde_true_4_x" => [kde_true_4.x...],
    "kde_true_4_density" => kde_true_4.density,
    "kde_clustered_5_x" => [kde_clustered_5.x...],
    "kde_clustered_5_density" => kde_clustered_5.density,
    "kde_true_5_x" => [kde_true_5.x...],
    "kde_true_5_density" => kde_true_5.density,
    "kde_clustered_6_x" => [kde_clustered_6.x...],
    "kde_clustered_6_density" => kde_clustered_6.density,
    "kde_true_6_x" => [kde_true_6.x...],
    "kde_true_6_density" => kde_true_6.density,
    "dt" => dt,
    "Nsteps" => Nsteps,
    "ϵ" => ϵ,
    "res_trj" => res_trj,
    "n_tau" => n_tau
))

##

results = read_variables_from_hdf5("data/HOR_data/CdV_results_ld_0.05-0.00001-128-300_ld.h5")
# Extract all variables from the results dictionary
R_num = results["R_num"]
R_lin = results["R_lin"]
R_gen = results["R_gen"]
S = results["S"]
M = results["M"]
obs_trj = results["obs_trj"]
score_trj = results["score_trj"]
σ_value = results["σ_value"]
averages = results["averages"]
centers = results["centers"]
Nc = results["Nc"]
kde_clustered_1_x = results["kde_clustered_1_x"]
kde_clustered_1_density = results["kde_clustered_1_density"]
kde_true_1_x = results["kde_true_1_x"]
kde_true_1_density = results["kde_true_1_density"]
kde_clustered_2_x = results["kde_clustered_2_x"]
kde_clustered_2_density = results["kde_clustered_2_density"]
kde_true_2_x = results["kde_true_2_x"]
kde_true_2_density = results["kde_true_2_density"]
kde_clustered_3_x = results["kde_clustered_3_x"]
kde_clustered_3_density = results["kde_clustered_3_density"]
kde_true_3_x = results["kde_true_3_x"]
kde_true_3_density = results["kde_true_3_density"]
kde_clustered_4_x = results["kde_clustered_4_x"]
kde_clustered_4_density = results["kde_clustered_4_density"]
kde_true_4_x = results["kde_true_4_x"]
kde_true_4_density = results["kde_true_4_density"]
kde_clustered_5_x = results["kde_clustered_5_x"]
kde_clustered_5_density = results["kde_clustered_5_density"]
kde_true_5_x = results["kde_true_5_x"]
kde_true_5_density = results["kde_true_5_density"]
kde_clustered_6_x = results["kde_clustered_6_x"]
kde_clustered_6_density = results["kde_clustered_6_density"]
kde_true_6_x = results["kde_true_6_x"]
kde_true_6_density = results["kde_true_6_density"]
dt = results["dt"]
Nsteps = results["Nsteps"]
ϵ = results["ϵ"]
res_trj = results["res_trj"]
n_tau = results["n_tau"]

##

using GLMakie
using Distributions  # For the standard normal PDF

# Create figure with 6×5 layout (added PDF column)
fig = Figure(resolution=(1800, 2200), font="CMU Serif", fontsize=24)

# Define common elements
colors = [:blue, :black, :red]
labels = ["Numerical", "Gaussian", "KGMM"]
time_axis = 0:dt*res_trj:n_tau*dt*res_trj

# Create standard normal PDF function for the "linear" model
normal_dist = Normal(0, 1)
gaussian_pdf(x) = pdf.(normal_dist, x)

# Create axes array - now 5 columns (PDFs + 4 moments)
axes = Matrix{Axis}(undef, 6, 5)

# Column titles
titles = ["PDF", "1st moment", "2nd moment", 
          "3rd moment", "4th moment"]
# Row labels
ylabels = ["x₁", "x₂", "x₃", 
           "x₄", "x₅", "x₆"]

# Calculate y-axis limits for each column of response functions (columns 2-5)
y_limits = zeros(4, 2)  # [moment, min/max]
for j in 1:4  # For each moment
    all_values = []
    for i in 1:6  # For each variable
        # Collect all values that will be plotted for this moment
        push!(all_values, R_num[j,i,:]./ϵ ./ S[i]^(j-1))
        push!(all_values, R_lin[j,i,:]./ϵ)
        push!(all_values, R_gen[j,i,:]./ϵ)
    end
    
    # Combine all values and find min/max
    combined = vcat(all_values...)
    valid_values = filter(isfinite, combined)  # Remove any Inf or NaN
    if !isempty(valid_values)
        y_limits[j, 1] = minimum(valid_values)  # Min value for this moment
        y_limits[j, 2] = maximum(valid_values)  # Max value for this moment
        
        # Add a small padding (5%) to the limits for better visualization
        range = y_limits[j, 2] - y_limits[j, 1]
        y_limits[j, 1] -= 0.05 * range
        y_limits[j, 2] += 0.05 * range
    end
end

# Create subplots
for i in 1:6
    # First column: PDF plots
    axes[i,1] = Axis(fig[i,1],
        xlabel = (i == 6) ? "Value" : "",
        ylabel = ylabels[i],
        title = (i == 1) ? titles[1] : "",
        titlesize = 36,
        xlabelsize = 28,
        ylabelsize = 28,
        xticklabelsize = 24,
        yticklabelsize = 24,
        limits = ((-6,6), nothing)  # Set x-limits
    )
    
    # Get the appropriate KDE data based on variable index
    true_x = eval(Symbol("kde_true_$(i)_x"))
    true_density = eval(Symbol("kde_true_$(i)_density"))
    clustered_x = eval(Symbol("kde_clustered_$(i)_x"))
    clustered_density = eval(Symbol("kde_clustered_$(i)_density"))
    
    # Create Gaussian PDF with same x range as the true data
    gauss_x = LinRange(-6, 6, 100)
    gauss_y = gaussian_pdf.(gauss_x)
    
    # Scale the Gaussian to match magnitude of true PDF for better visibility
    scale_factor = maximum(true_density) / maximum(gauss_y)
    gauss_y_scaled = gauss_y .* scale_factor
    
    # Plot PDFs
    lines!(axes[i,1], true_x, true_density, color=colors[1], linewidth=3)
    lines!(axes[i,1], gauss_x, gauss_y_scaled, color=colors[2], linewidth=3)
    lines!(axes[i,1], clustered_x, clustered_density, color=colors[3], linewidth=3)

    # Response function plots (now in columns 2-5)
    for j in 1:4
        response_col = j + 1  # Column index in the figure (PDF is col 1)
        axes[i,response_col] = Axis(fig[i,response_col],
            xlabel = (i == 6) ? "Time lag" : "",
            ylabel = "",  # Only first column gets y labels now
            title = (i == 1) ? titles[response_col] : "",
            titlesize = 36,
            xlabelsize = 28,
            ylabelsize = 28,
            xticklabelsize = 24,
            yticklabelsize = 24,
            limits = (nothing, (y_limits[j, 1], y_limits[j, 2]))
        )

        # Plot response data
        lines!(axes[i,response_col], time_axis, R_num[j,i,:]./ϵ ./ S[i]^(j-1), color=colors[1], linewidth=3)
        lines!(axes[i,response_col], time_axis, R_lin[j,i,:]./ϵ, color=colors[2], linewidth=3)
        lines!(axes[i,response_col], time_axis, R_gen[j,i,:]./ϵ, color=colors[3], linewidth=3)
        
        # Add grid lines for readability
        axes[i,response_col].xgridvisible = true
        axes[i,response_col].ygridvisible = true
    end
end

# Add unified legend at the bottom
Legend(fig[7, :],
    [LineElement(color=c, linewidth=3, linestyle=:solid)
     for c in colors],
    labels,
    "Methods",
    orientation = :horizontal,
    titlesize = 32,
    labelsize = 28
)

# Adjust spacing
colgap!(fig.layout, 20)
rowgap!(fig.layout, 20)
# Add more bottom margin for A4 printing
fig.layout[8, :] = GridLayout(height=20)

# Save figure
save("figures/HOR_figures/CdV_responses_ylims_ld.png", fig, px_per_unit=2)  # Higher DPI for better print quality

fig

##
######################################### FIGURE WITHOUT YLIMITS #########################################

using GLMakie
using Distributions  # For the standard normal PDF

# Create figure with 6×5 layout (added PDF column)
fig = Figure(resolution=(1800, 2200), font="CMU Serif", fontsize=24)

# Define common elements
colors = [:blue, :black, :red]
labels = ["Numerical", "Linear", "KGMM"]
time_axis = 0:dt*res_trj:n_tau*dt*res_trj

# Create standard normal PDF function for the "linear" model
normal_dist = Normal(0, 1)
gaussian_pdf(x) = pdf.(normal_dist, x)

# Create axes array - now 5 columns (PDFs + 4 moments)
axes = Matrix{Axis}(undef, 6, 5)

# Column titles
titles = ["PDF", "1st moment", "2nd moment", 
          "3rd moment", "4th moment"]
# Row labels
ylabels = ["x₁", "x₂", "x₃", 
           "x₄", "x₅", "x₆"]

# Calculate y-axis limits for each column of response functions (columns 2-5)
y_limits = zeros(4, 2)  # [moment, min/max]
for j in 1:4  # For each moment
    all_values = []
    for i in 1:6  # For each variable
        # Collect all values that will be plotted for this moment
        push!(all_values, R_num[j,i,:]./ϵ ./ S[i]^(j-1))
        push!(all_values, R_lin[j,i,:]./ϵ)
        push!(all_values, R_gen[j,i,:]./ϵ)
    end
    
    # Combine all values and find min/max
    combined = vcat(all_values...)
    valid_values = filter(isfinite, combined)  # Remove any Inf or NaN
    if !isempty(valid_values)
        y_limits[j, 1] = minimum(valid_values)  # Min value for this moment
        y_limits[j, 2] = maximum(valid_values)  # Max value for this moment
        
        # Add a small padding (5%) to the limits for better visualization
        range = y_limits[j, 2] - y_limits[j, 1]
        y_limits[j, 1] -= 0.05 * range
        y_limits[j, 2] += 0.05 * range
    end
end

# Create subplots
for i in 1:6
    # First column: PDF plots
    axes[i,1] = Axis(fig[i,1],
        xlabel = (i == 6) ? "Value" : "",
        ylabel = ylabels[i],
        title = (i == 1) ? titles[1] : "",
        titlesize = 36,
        xlabelsize = 28,
        ylabelsize = 28,
        xticklabelsize = 24,
        yticklabelsize = 24
    )
    
    # Get the appropriate KDE data based on variable index
    true_x = eval(Symbol("kde_true_$(i)_x"))
    true_density = eval(Symbol("kde_true_$(i)_density"))
    clustered_x = eval(Symbol("kde_clustered_$(i)_x"))
    clustered_density = eval(Symbol("kde_clustered_$(i)_density"))
    
    # Create Gaussian PDF with same x range as the true data
    gauss_x = LinRange(minimum(true_x), maximum(true_x), 100)
    gauss_y = gaussian_pdf.(gauss_x)
    
    # Scale the Gaussian to match magnitude of true PDF for better visibility
    scale_factor = maximum(true_density) / maximum(gauss_y)
    gauss_y_scaled = gauss_y .* scale_factor
    
    # Plot PDFs
    lines!(axes[i,1], true_x, true_density, color=colors[1], linewidth=3, xlims!=(-6,6))
    lines!(axes[i,1], gauss_x, gauss_y_scaled, color=colors[2], linewidth=3, xlims!=(-6,6))
    lines!(axes[i,1], clustered_x, clustered_density, color=colors[3], linewidth=3, xlims!=(-6,6))
    
    # Response function plots (now in columns 2-5)
    for j in 1:4
        response_col = j + 1  # Column index in the figure (PDF is col 1)
        axes[i,response_col] = Axis(fig[i,response_col],
            xlabel = (i == 6) ? "Time lag" : "",
            ylabel = "",  # Only first column gets y labels now
            title = (i == 1) ? titles[response_col] : "",
            titlesize = 36,
            xlabelsize = 28,
            ylabelsize = 28,
            xticklabelsize = 24,
            yticklabelsize = 24
            # limits parameter removed to allow automatic y-axis scaling
        )

        # Plot response data
        lines!(axes[i,response_col], time_axis, R_num[j,i,:]./ϵ ./ S[i]^(j-1), color=colors[1], linewidth=3)
        lines!(axes[i,response_col], time_axis, R_lin[j,i,:]./ϵ, color=colors[2], linewidth=3)
        lines!(axes[i,response_col], time_axis, R_gen[j,i,:]./ϵ, color=colors[3], linewidth=3)
        
        # Add grid lines for readability
        axes[i,response_col].xgridvisible = true
        axes[i,response_col].ygridvisible = true
    end
end

# Add unified legend at the bottom
Legend(fig[7, :],
    [LineElement(color=c, linewidth=3, linestyle=:solid)
     for c in colors],
    labels,
    "Methods",
    orientation = :horizontal,
    titlesize = 32,
    labelsize = 28
)

# Adjust spacing
colgap!(fig.layout, 20)
rowgap!(fig.layout, 20)
# Add more bottom margin for A4 printing
fig.layout[8, :] = GridLayout(height=20)

# Save figure
save("figures/HOR_figures/CdV_responses.png", fig, px_per_unit=2)  # Higher DPI for better print quality

fig