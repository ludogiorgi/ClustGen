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
using Random
using QuadGK
using GLMakie
##

# Triad Model parameters
L11 = -2.0
L12 = 0.2
L13 = 0.1
g2 = 0.6
g3 = 0.4
s2 = 1.2
s3 = 0.8
II = 1.0
ϵ = 0.1

# Coefficients of the reduced model
a = L11 + ϵ * ( (II^2 * s2^2) / (2 * g2^2) - (L12^2) / g2 - (L13^2) / g3 )
b = -2 * (L12 * II) / (g2) * ϵ
c = (II^2) / (g2) * ϵ
B = -(II * s2) / (g2) * sqrt(ϵ)
A = -(L12 * B) / II
F_tilde = (A * B) / 2   
s = (L13 * s3) / g3 * sqrt(ϵ)

function F(x,t; F_tilde=F_tilde)
    u = x[1]
    return [-F_tilde + a * u + b * u^2 - c * u^3]
end

function sigma1(x,t)
    return (A - B * x[1])/√2
end

function sigma2(x,t)
    return s/√2
end

function score_true(x)
    u = x[1]
    return [2 * ((A*B/2) + (a-B^2)*u + b*u^2 - c*u^3) / (s^2+(A-B*u)^2)]
end

function normalize_f(f, x, M, S)
    return (f(x .* S .+ M) .* S)[:]
end

dim = 1
dt = 0.01
Nsteps = 20000000
obs_nn = evolve([0.0], dt, Nsteps, F, sigma1, sigma2; timestepper=:euler)
M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S

autocov_obs = zeros(dim, 300)
for i in 1:dim
    autocov_obs[i,:] = autocovariance(obs[i,1:1000000]; timesteps=300)
end

kde_obs = kde(obs[200:end])

autocov_obs_mean = mean(autocov_obs, dims=1)

plt1 = Plots.plot(autocov_obs_mean[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")
plt2 = Plots.plot(kde_obs.x, kde_obs.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")

Plots.plot(plt1, plt2, layout=(2, 1), size=(800, 800))

##
obs_uncorr = obs[:, 1:1:end]

Plots.scatter(obs_uncorr[1,1:10000], markersize=2, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

function score_true_norm(x)
    return normalize_f(score_true, x, M, S)
end

normalization = false
σ_value = 0.02

averages, averages_residual, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.002, do_print=true, conv_param=0.1, normalization=normalization)

if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
    inputs_targets_residual, M_averages_values_residual, m_averages_values_residual = generate_inputs_targets(averages_residual, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
    inputs_targets_residual = generate_inputs_targets(averages_residual, centers, Nc; normalization=false)
end

centers_sorted_indices = sortperm(centers[1,:])
centers_sorted = centers[:,centers_sorted_indices][:]
scores = .- averages[:,centers_sorted_indices][:] ./ σ_value
scores_true = [score_true_norm(centers_sorted[i])[1] for i in eachindex(centers_sorted)]

Plots.scatter(centers_sorted, scores, color=:blue)
Plots.plot!(centers_sorted, scores_true, color=:red)
##
#################### TRAINING WITH CLUSTERING LOSS ####################
inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
normalization = false
σ_value = 0.02

@time nn_clustered, loss_clustered = train(inputs_targets, 2000, 32, [dim, 50, 25, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value

Plots.plot(loss_clustered)

##
#################### VECTOR FIELDS ####################

xax = [centers_sorted[1]:0.005:centers_sorted[end]...]

s_true = [score_true_norm(xax[i])[1] for i in eachindex(xax)]
s_gen = [score_clustered(xax[i])[1] for i in eachindex(xax)]

Plots.scatter(centers_sorted, scores, color=:blue)
Plots.plot!(xax, s_true, label="True", xlabel="X", ylabel="Force", title="Forces")
Plots.plot!(xax, s_gen, label="Learned")

##
#################### SAMPLES GENERATION ####################

score_gen(x) = score_clustered(x)
score_clustered_xt(x,t) = score_clustered(x)
score_true_xt(x,t) = score_true_norm(x)
sigma_I(x,t) = 1.0

trj_clustered = evolve([0.0], 0.1*dt,Nsteps, score_clustered_xt, sigma_I; timestepper=:euler, resolution=10)
trj_true = evolve([0.0], 0.1*dt, Nsteps, score_true_xt, sigma_I; timestepper=:euler, resolution=10)

kde_clustered = kde(trj_clustered[:])
kde_obs = kde(obs[:])
kde_true = kde(obs[:])

Plots.plot(kde_clustered.x, kde_clustered.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
Plots.plot!(kde_obs.x, kde_obs.density, label="Learned", xlabel="X", ylabel="Density", title="Learned PDF")
Plots.plot!(kde_true.x, kde_true.density, label="True", xlabel="X", ylabel="Density", title="True PDF")
##
############## OBSERVATIONS GENERATION ####################

obs_trj = evolve([0.0], dt, 10000, F, sigma1, sigma2; timestepper=:rk4)[:]
obs_trj = (obs_trj .- M) ./ S
score_trj = evolve([0.0], 0.1*dt, 100000, score_clustered_xt, sigma_I; timestepper=:rk4, resolution=10)[:]

Plots.plot(obs_trj)
Plots.plot!(score_trj)

##
#################### Responses ####################

using ClustGen
f(t) = 1.0
score_gen(x) = score_clustered(x)

res_trj = 5
steps_trj = 500000
trj = obs[:,1:res_trj:steps_trj*res_trj]

ϵ = 1.0

function u(x)
    U = zeros(dim)
    U[1] = ϵ                          
    return U
end

div_u(x) = 0.0
invC0 = inv(cov(obs[:,:,1]'))
score_qG(x) = - invC0* (x)

dim_Obs = 1
n_tau = 200

# R_num, δObs_num = zeros(4, dim_Obs, n_tau+1), zeros(4, dim_Obs, n_tau+1)
R_lin, δObs_lin = zeros(4, dim_Obs, n_tau+1), zeros(4, dim_Obs, n_tau+1)
R_gen, δObs_gen = zeros(4, dim_Obs, n_tau+1), zeros(4, dim_Obs, n_tau+1)
R_true, δObs_true = zeros(4, dim_Obs, n_tau+1), zeros(4, dim_Obs, n_tau+1)

# R_num[1,:,:], R_num[2,:,:], R_num[3,:,:], R_num[4,:,:] = generate_numerical_response_HO(F, u, dim, 0.1*dt, n_tau, 10*1000, sigma1, sigma2, M; n_ens=1000, resolution=10*res_trj, timestepper=:euler)

for i in 1:4
    Obs(x) = x .^i
    R_lin[i,:,:], δObs_lin[i,:,:] = generate_score_response(trj, u, div_u, f, score_qG, res_trj*dt, n_tau, Obs, dim_Obs)
    R_gen[i,:,:], δObs_gen[i,:,:] = generate_score_response(trj, u, div_u, f, score_gen, res_trj*dt, n_tau, Obs, dim_Obs)
    R_true[i,:,:], δObs_true[i,:,:] = generate_score_response(trj, u, div_u, f, score_true_norm, res_trj*dt, n_tau, Obs, dim_Obs)
end

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

plt1 = Plots.plot(R_lin[1,1,:])
plt1 = Plots.plot!(R_gen[1,1,:])
plt1 = Plots.plot!(R_true[1,1,:])
plt2 = Plots.plot(R_lin[2,1,:])
plt2 = Plots.plot!(R_gen[2,1,:])
plt2 = Plots.plot!(R_true[2,1,:])
plt3 = Plots.plot(R_lin[3,1,:])
plt3 = Plots.plot!(R_gen[3,1,:])
plt3 = Plots.plot!(R_true[3,1,:])
plt4 = Plots.plot(R_lin[4,1,:])
plt4 = Plots.plot!(R_gen[4,1,:])
plt4 = Plots.plot!(R_true[4,1,:])
Plots.plot(plt1, plt2, plt3, plt4, layout=(2, 2), size=(800, 800))

##
############## OBSERVATIONS GENERATION ####################

obs_trj = evolve(zeros(dim), dt, 100000, F, sigma1, sigma2; timestepper=:rk4)
obs_trj = (obs_trj .- M) ./ S
score_trj = evolve(zeros(dim), dt, 100000, score_clustered_xt, sigma_I; timestepper=:rk4, resolution=1)

##

# Save results to HDF5 file for 1D system
save_variables_to_hdf5("data/HOR_data/reduced_results.h5", Dict(
    "R_true" => R_true,
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
    "kde_clustered_1_x" => [kde_clustered.x...],
    "kde_clustered_1_density" => kde_clustered.density,
    "kde_true_1_x" => [kde_true.x...],
    "kde_true_1_density" => kde_true.density,
    "dt" => dt,
    "Nsteps" => Nsteps,
    "ϵ" => ϵ,
    "res_trj" => res_trj,
    "n_tau" => n_tau
))

##

results = read_variables_from_hdf5("data/HOR_data/reduced_results.h5")
# Extract all variables from the results dictionary
R_true = results["R_true"]
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
dt = results["dt"]
Nsteps = results["Nsteps"]
ϵ = results["ϵ"]
res_trj = results["res_trj"]
n_tau = results["n_tau"]

##

# R_true[2,1,1] = 0.01
# R_gen[2,1,1] = 0.0
# R_gen[2,1,:] .-= 0.005

using GLMakie
using Distributions  # For the standard normal PDF

# Create figure with 1×5 layout (1D system, PDF + 4 moments)
fig = Figure(resolution=(1800, 600), font="CMU Serif", fontsize=24)

# Define common elements
colors = [:blue, :black, :red]
labels = ["True", "Linear", "Generative"]
time_axis = 0:dt*res_trj:n_tau*dt*res_trj

# Create standard normal PDF function for the "linear" model
normal_dist = Normal(0, 1)
gaussian_pdf(x) = pdf.(normal_dist, x)

# Create axes array - 5 columns (PDF + 4 moments)
axes = Vector{Axis}(undef, 5)

# Column titles
titles = ["PDF", "1st moment", "2nd moment", 
          "3rd moment", "4th moment"]

# Calculate y-axis limits for each response function (columns 2-5)
y_limits = zeros(4, 2)  # [moment, min/max]
for j in 1:4  # For each moment
    all_values = []
    # Collect all values that will be plotted for this moment
    push!(all_values, R_true[j,1,:]./ϵ)
    push!(all_values, R_lin[j,1,:]./ϵ)
    push!(all_values, R_gen[j,1,:]./ϵ)
    
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

# First column: PDF plot
axes[1] = Axis(fig[1,1],
    xlabel = "Value",
    ylabel = "x",
    title = titles[1],
    titlesize = 36,
    xlabelsize = 28,
    ylabelsize = 28,
    xticklabelsize = 24,
    yticklabelsize = 24,
    limits = (-3, 5, nothing, nothing),
)

# Get the KDE data
true_x = kde_true_1_x
true_density = kde_true_1_density
clustered_x = kde_clustered_1_x
clustered_density = kde_clustered_1_density

# Create Gaussian PDF with same x range as the true data
gauss_x = LinRange(minimum(true_x), maximum(true_x), 100)
gauss_y = gaussian_pdf.(gauss_x)

# Scale the Gaussian to match magnitude of true PDF for better visibility
scale_factor = maximum(true_density) / maximum(gauss_y)
gauss_y_scaled = gauss_y .* scale_factor

# Plot PDF
lines!(axes[1], true_x, true_density, color=colors[1], linewidth=3)
lines!(axes[1], gauss_x, gauss_y_scaled, color=colors[2], linewidth=3)
lines!(axes[1], clustered_x, clustered_density, color=colors[3], linewidth=3)

# Response function plots (columns 2-5)
for j in 1:4
    response_col = j + 1  # Column index in the figure (PDF is col 1)
    axes[response_col] = Axis(fig[1,response_col],
        xlabel = "Time lag",
        ylabel = "",
        title = titles[response_col],
        titlesize = 36,
        xlabelsize = 28,
        ylabelsize = 28,
        xticklabelsize = 24,
        yticklabelsize = 24,
        limits = (nothing, (y_limits[j, 1], y_limits[j, 2]))
    )

    # Plot response data
    lines!(axes[response_col], time_axis, R_true[j,1,:]./ϵ , color=colors[1], linewidth=3)
    lines!(axes[response_col], time_axis, R_lin[j,1,:]./ϵ, color=colors[2], linewidth=3)
    lines!(axes[response_col], time_axis, R_gen[j,1,:]./ϵ, color=colors[3], linewidth=3)
    
    # Add grid lines for readability
    axes[response_col].xgridvisible = true
    axes[response_col].ygridvisible = true
end

# Add unified legend at the bottom
Legend(fig[2, :],
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
# Add more bottom margin for better layout
fig.layout[3, :] = GridLayout(height=20)

# Save figure
# save("figures/HOR_figures/reduced_responses_with_pdfs.png", fig, px_per_unit=2)  # Higher DPI for better print quality

fig

##
n_tau = 200
ϵ = 0.1*F_tilde
F_pert(x,t) = F(x,t;F_tilde=F_tilde+ϵ)
R_num, δObs_num = zeros(4, dim_Obs, n_tau+1), zeros(4, dim_Obs, n_tau+1)
δObs_num[1,:,:], δObs_num[2,:,:], δObs_num[3,:,:], δObs_num[4,:,:] = generate_numerical_response_f_HO(F, F_pert, dim, dt, n_tau, 2000, sigma1, sigma2, M[1]; n_ens=1000, resolution=res_trj, timestepper=:euler)
##
##
#################### Responses ####################

f(t) = 1.0

res_trj = 5
steps_trj = 500000
trj = obs[:,1:res_trj:steps_trj*res_trj]

u(x) = - ϵ
div_u(x) = 0.0
invC0 = inv(cov(obs[:,:,1]'))
score_qG(x) = - invC0*x

dim_Obs = 1

R_lin, δObs_lin = zeros(4, dim_Obs, n_tau+1), zeros(4, dim_Obs, n_tau+1)
R_gen, δObs_gen = zeros(4, dim_Obs, n_tau+1), zeros(4, dim_Obs, n_tau+1)
R_true, δObs_true = zeros(4, dim_Obs, n_tau+1), zeros(4, dim_Obs, n_tau+1)

for i in 1:4
    Obs(x) = x .^i
    R_lin[i,:,:], δObs_lin[i,:,:] = generate_score_response(trj, u, div_u, f, score_qG, res_trj*dt, n_tau, Obs, dim_Obs)
    R_gen[i,:,:], δObs_gen[i,:,:] = generate_score_response(trj, u, div_u, f, score_gen, res_trj*dt, n_tau, Obs, dim_Obs)
    R_true[i,:,:], δObs_true[i,:,:] = generate_score_response(trj, u, div_u, f, score_true_norm, res_trj*dt, n_tau, Obs, dim_Obs)
end
##
plt1 = Plots.plot(R_lin[1,1,:])
plt1 = Plots.plot!(R_gen[1,1,:])
plt1 = Plots.plot!(R_true[1,1,:])
plt2 = Plots.plot(R_lin[2,1,:])
plt2 = Plots.plot!(R_gen[2,1,:])
plt2 = Plots.plot!(R_true[2,1,:])
plt3 = Plots.plot(R_lin[3,1,:])
plt3 = Plots.plot!(R_gen[3,1,:])
plt3 = Plots.plot!(R_true[3,1,:])
plt4 = Plots.plot(R_lin[4,1,:])
plt4 = Plots.plot!(R_gen[4,1,:])
plt4 = Plots.plot!(R_true[4,1,:])
Plots.plot(plt1, plt2, plt3, plt4, layout=(2, 2), size=(800, 800))
##
plt1 = Plots.plot(δObs_lin[1,1,:])
plt1 = Plots.plot!s(δObs_num[1,1,:])
plt1 = Plots.plot!(δObs_true[1,1,:])
plt2 = Plots.plot(δObs_lin[2,1,:])
plt2 = Plots.plot!(δObs_num[2,1,:] ./ S[1])
plt2 = Plots.plot!(δObs_true[2,1,:])
plt3 = Plots.plot(δObs_lin[3,1,:])
plt3 = Plots.plot!(δObs_num[3,1,:] ./ S[1]^2)
plt3 = Plots.plot!(δObs_true[3,1,:])
plt4 = Plots.plot(δObs_lin[4,1,:])
plt4 = Plots.plot!(δObs_num[4,1,:] ./ S[1]^3)
plt4 = Plots.plot!(δObs_true[4,1,:])
Plots.plot(plt1, plt2, plt3, plt4, layout=(2, 2), size=(800, 800))

##
Nsteps = 5000000
ϵ = F_tilde
F_pert(x,t) = F(x,t;F_tilde=F_tilde+ϵ)
obs_pert_nn = evolve([0.0], dt, Nsteps, F_pert, sigma1, sigma2; timestepper=:euler)
obs_pert = (obs_pert_nn .- M) ./ S

kde_obs_pert = kde(obs_pert[:])

Plots.plot(kde_obs_pert.x, kde_obs_pert.density, label="Perturbed", xlabel="X", ylabel="Density", title="Observed PDF")
Plots.plot!(kde_obs.x, kde_obs.density, label="Unperturbed", xlabel="X", ylabel="Density", title="Learned PDF")
##

order = 4
moments_unpert = [1.0] 
moments_pert = [1.0]
moments_gen = [1.0]  
moments_lin = [1.0]

for k in 1:order
    push!(moments_unpert, mean(obs[:] .^ k))
    push!(moments_pert, mean(obs_pert[:] .^ k))
    push!(moments_gen, moments_unpert[k+1] - ϵ * δObs_true[k,1,end])
    push!(moments_lin, moments_unpert[k+1] - ϵ * δObs_lin[k,1,end])
end

println(moments_unpert)
println(moments_pert)
println(moments_gen)
println(moments_lin)

##
pdf_func, xs_gen, ys_gen, best_λ = maxent_distribution(moments_gen, bounds=(-3.0, 7.0), grid_size=500)
pdf_func, xs_lin, ys_lin, best_λ = maxent_distribution(moments_lin, bounds=(-3.0, 7.0), grid_size=500)


# Plot the result
Plots.plot(xs_gen, ys_gen, label="gen")
Plots.plot!(xs_lin, ys_lin, label="lin")
Plots.plot!(kde_obs_pert.x, kde_obs_pert.density, label="pert")
Plots.plot!(kde_obs.x, kde_obs.density, label="unpert")

##

# Triad Model parameters
L11 = -2.0
L12 = 0.2
L13 = 0.1
g2 = 0.6
g3 = 0.4
s2 = 1.2
s3 = 0.8
II = 1.0
ϵ = 0.1

# Coefficients of the reduced model
a = L11 + ϵ * ( (II^2 * s2^2) / (2 * g2^2) - (L12^2) / g2 - (L13^2) / g3 )
b = -2 * (L12 * II) / (g2) * ϵ
c = (II^2) / (g2) * ϵ
B = -(II * s2) / (g2) * sqrt(ϵ)
A = -(L12 * B) / II
F_tilde = (A * B) / 2   
s = (L13 * s3) / g3 * sqrt(ϵ)

function p(u, Ftilde)
    # ArcTan term
    arctan_term = 2 * (
        -A^2*b*B + A^3*c - B^3*Ftilde + b*B*s^2 + 
        A*(-a*B^2 + B^4 - 3*c*s^2)
    ) * atan((A - B*u)/s)

    # Log term
    log_term = s * (
        (A - B*u)*(-2*b*B + 5*A*c + B*c*u) + 
        (2*A*b*B + a*B^2 - B^4 - 3*A^2*c + c*s^2) * log(s^2 + (A - B*u)^2)
    )

    return exp((1 / (B^4 * s)) * (arctan_term + log_term))
end


p_unpert(u) = p(u, F_tilde)




# Plots.plot(xax, p_unpert.(xax), label="Unperturbed", xlabel="X", ylabel="Density", title="Unperturbed PDF")
N_unpert, _ = quadgk(u -> p_unpert(u), p_inf, m_inf, rtol=1e-8)
mean_p, _ = quadgk(u -> u * p_unpert(u), p_inf, m_inf, rtol=1e-8)
mean_p /= N_unpert
var_p, _ = quadgk(u -> (u - mean_p)^2 * p_unpert(u), p_inf, m_inf, rtol=1e-8)
var_p /= N_unpert

function compute_moments(m_inf, p_inf, ϵ; n_moments=4)
    p_pert(u) = p(u, F_tilde+ϵ)
    N_pert, _ = quadgk(u -> p_pert(u), p_inf, m_inf, rtol=1e-8)
    moments = zeros(n_moments)   
    for k in 1:n_moments
        integral, _ = quadgk(u -> (u-mean_p)^k * p_pert(u), p_inf, m_inf, rtol=1e-8)
        moments[k] = integral / N_pert  / sqrt(var_p^k)
    end
    
    return moments 
end

ϵ_vals = [0.0:0.005:0.15...]

p_inf, m_inf = -7.0, 7.0

moments_pert = zeros(length(ϵ_vals), n_moments)

for (i, ϵ) in enumerate(ϵ_vals)  
    moments_pert[i,:] = compute_moments(m_inf, p_inf, ϵ)
end

plt1 = Plots.plot(ϵ_vals, moments_pert[:,1], label="Perturbed", title="1st moment", xlabel="ϵ", ylabel="Moment", lw=2)
plt2 = Plots.plot(ϵ_vals, moments_pert[:,2], label="Perturbed", title="2nd moment", xlabel="ϵ", ylabel="Moment", lw=2)
plt3 = Plots.plot(ϵ_vals, moments_pert[:,3], label="Perturbed", title="3rd moment", xlabel="ϵ", ylabel="Moment", lw=2)
plt4 = Plots.plot(ϵ_vals, moments_pert[:,4], label="Perturbed", title="4th moment", xlabel="ϵ", ylabel="Moment", lw=2)

plt = Plots.plot(plt1, plt2, plt3, plt4, layout=(2, 2), size=(800, 800))
Plots.savefig(plt, "figures/HOR_figures/reduced_mult_moments.png")