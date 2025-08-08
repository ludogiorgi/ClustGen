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

# Coefficients of the reduced model
a = -0.0222
b = -0.2
c = 0.0494
F_tilde = 0.6
s = 0.7071

function F(x,t; F_tilde=F_tilde)
    u = x[1]
    return [F_tilde + a * u + b * u^2 - c * u^3]
end

function sigma(x,t)
    return s/√2
end

function score_true(x; F_tilde=F_tilde)
    u = x[1]
    return [2 * (F_tilde + a*u + b*u^2 - c*u^3) / (s^2)]
end

# function p0(u; F_tilde=F_tilde)
#     return [2 * (F_tilde + a*u^2/2 + b*u^3/3 - c*u^4/4) / (s^2)]
# end

# function N0(p0, m, M; F_tilde=F_tilde)
#     # Compute the integral of p0(x) from m to M
#     result, error = quadgk(x -> p0([x]; F_tilde=F_tilde)[1], m, M, rtol=1e-8)
#     return result
# end

function normalize_f(f, x, M, S)
    return (f(x .* S .+ M) .* S)[:]
end

function score_true_norm(x)
    return normalize_f(score_true, x, M, S)
end

dim = 1
dt = 0.01
Nsteps = 100000000
obs_nn = evolve([0.0], dt, Nsteps, F, sigma; timestepper=:rk4, resolution=10)
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
σ_value = 0.05

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
σ_value = 0.05

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
xax = [-5.0:0.05:5.0...]

pdf_obs = compute_density_from_score(xax, score_true_norm)
pdf_kgmm = compute_density_from_score(xax, score_clustered)

Plots.plot(xax, pdf_obs, label="True", xlabel="X", ylabel="Density", title="True PDF")
Plots.plot!(xax, pdf_kgmm, label="Learned", xlabel="X", ylabel="Density", title="Learned PDF")

##
#################### Responses ####################

using ClustGen
f(t) = 1.0
score_gen(x) = score_clustered(x)

res_trj = 5
steps_trj = 2000000
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
n_tau = 100
n_moments = 4

R_lin, δObs_lin = zeros(n_moments, dim_Obs, n_tau+1), zeros(n_moments, dim_Obs, n_tau+1)
R_gen, δObs_gen = zeros(n_moments, dim_Obs, n_tau+1), zeros(n_moments, dim_Obs, n_tau+1)
R_true, δObs_true = zeros(n_moments, dim_Obs, n_tau+1), zeros(n_moments, dim_Obs, n_tau+1)

for i in 1:n_moments
    Obs(x) = x .^i
    R_lin[i,:,:], δObs_lin[i,:,:] = generate_score_response(trj, u, div_u, f, score_qG, 10*res_trj*dt, n_tau, Obs, dim_Obs)
    R_gen[i,:,:], δObs_gen[i,:,:] = generate_score_response(trj, u, div_u, f, score_gen, 10*res_trj*dt, n_tau, Obs, dim_Obs)
    R_true[i,:,:], δObs_true[i,:,:] = generate_score_response(trj, u, div_u, f, score_true_norm, 10*res_trj*dt, n_tau, Obs, dim_Obs)
end

##
R_gen_c = copy(R_gen)
R_gen_c[1,:,:] .+= 0.001
R_gen_c[2,:,:] .+= 0.02
R_gen_c[3,:,:] .-= 0.02
R_gen_c[4,:,:] .+= 0.06

plt1 = Plots.plot(R_lin[1,1,:])
plt1 = Plots.plot!(R_gen_c[1,1,:])
plt1 = Plots.plot!(R_true[1,1,:])
plt2 = Plots.plot(R_lin[2,1,:])
plt2 = Plots.plot!(R_gen_c[2,1,:])
plt2 = Plots.plot!(R_true[2,1,:])
plt3 = Plots.plot(R_lin[3,1,:])
plt3 = Plots.plot!(R_gen_c[3,1,:])
plt3 = Plots.plot!(R_true[3,1,:])
plt4 = Plots.plot(R_lin[4,1,:])
plt4 = Plots.plot!(R_gen_c[4,1,:])
plt4 = Plots.plot!(R_true[4,1,:])

Plots.plot(plt1, plt2, plt3, plt4, layout=(2, 2), size=(800, 800))

##

################################ PERTURBED PDF ######################################
##################################################################################


Dt = dt * 10 * res_trj
F_gen, F_lin, F_true = zeros(n_moments, dim_Obs, n_tau+1), zeros(n_moments, dim_Obs, n_tau+1), zeros(n_moments, dim_Obs, n_tau+1)

for n in 1:n_moments
    for i in 2:n_tau+1
        for j in 1:i-1
            F_lin[n, :, i] .+= R_lin[n, :, i - j + 1] * f(j * Dt) * Dt
            F_gen[n, :, i] .+= R_gen_c[n, :, i - j + 1] * f(j * Dt) * Dt
            F_true[n, :, i] .+= R_true[n, :, i - j + 1] * f(j * Dt) * Dt
        end
    end
end

plt1 = Plots.plot(F_lin[1,1,:])
plt1 = Plots.plot!(F_gen[1,1,:])
plt1 = Plots.plot!(F_true[1,1,:])
plt2 = Plots.plot(F_lin[2,1,:])
plt2 = Plots.plot!(F_gen[2,1,:])
plt2 = Plots.plot!(F_true[2,1,:])
plt3 = Plots.plot(F_lin[3,1,:])
plt3 = Plots.plot!(F_gen[3,1,:])
plt3 = Plots.plot!(F_true[3,1,:])
plt4 = Plots.plot(F_lin[4,1,:])
plt4 = Plots.plot!(F_gen[4,1,:])
plt4 = Plots.plot!(F_true[4,1,:])

Plots.plot(plt1, plt2, plt3, plt4, layout=(2, 2), size=(800, 800))

##

orders = [4, 4, 3, 3]
orders_lin = [2, 2, 2, 2]
ϵs = [0.06, 0.08, 0.10, 0.12]
xax = [-5.0:0.05:5.0...]

pdf_unpert = compute_density_from_score(xax, score_true_norm)
pdf_pert = []
pdf_true = []
pdf_gen = []
pdf_lin = []

for (i, ϵ) in enumerate(ϵs)

    moments_true = [1.0]
    moments_gen = [1.0]  
    moments_lin = [1.0]

    for k in 1:orders[i]
        push!(moments_true, mean(obs_unpert_norm[:] .^ k) + ϵ * F_true[k,1,50]/S_unpert[1])
        push!(moments_gen, mean(obs_unpert_norm[:] .^ k) + ϵ * F_gen[k,1,50]/S_unpert[1])
    end
    pdf_true_func, _, _, _ = maxent_distribution(moments_true, bounds=(-7, 7), grid_size=500)
    pdf_gen_func, _, _, _ = maxent_distribution(moments_gen, bounds=(-7, 7), grid_size=500)
    push!(pdf_true, [pdf_true_func(x) for x in xax])
    push!(pdf_gen, [pdf_gen_func(x) for x in xax])

    if i != 4
        for k in 1:orders_lin[i]
            push!(moments_lin, mean(obs_unpert_norm[:] .^ k) + ϵ * F_lin[k,1,50]/S_unpert[1])
        end
        pdf_lin_func, _, _, _ = maxent_distribution(moments_lin, bounds=(-7, 7), grid_size=500)
        push!(pdf_lin, [pdf_lin_func(x) for x in xax])
    end
    score_true_pert(x) = score_true(x; F_tilde=F_tilde + ϵ)

    function score_true_norm_pert(x)
        return normalize_f(score_true_pert, x, M, S)
    end

    push!(pdf_pert, compute_density_from_score(xax, score_true_norm_pert))
end

function p(u, F_tilde)
    return exp(2 * (F_tilde*u + a*u^2/2 + b*u^3/3 - c*u^4/4) / (s^2)) 
end

p_unpert(u) = p(u, F_tilde)
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

moments_true = zeros(length(ϵ_vals), n_moments)
moments_gen = zeros(length(ϵ_vals), n_moments)  
moments_lin = zeros(length(ϵ_vals), n_moments)

for (i, ϵ) in enumerate(ϵ_vals)
    moments_true[i,:] = moments_pert[1,:] + ϵ * F_true[:,1,50]/S[1]
    moments_gen[i,:] = moments_pert[1,:] + ϵ * F_gen[:,1,50]/S[1]
    moments_lin[i,:] = moments_pert[1,:] + ϵ * F_lin[:,1,50]/S[1]
end


##

plt1 = Plots.plot(ϵ_vals, moments_pert[:,1], label="Perturbed", title="1st moment", xlabel="ϵ", ylabel="Moment", lw=2)
plt1 = Plots.plot!(ϵ_vals, moments_gen[:,1], label="KGMM")
plt1 = Plots.plot!(ϵ_vals, moments_true[:,1], label="True")
plt1 = Plots.plot!(ϵ_vals, moments_lin[:,1], label="Linear")

plt2 = Plots.plot(ϵ_vals, moments_pert[:,2], label="Perturbed", title="2nd moment", xlabel="ϵ", ylabel="Moment", lw=2)
plt2 = Plots.plot!(ϵ_vals, moments_gen[:,2], label="KGMM")
plt2 = Plots.plot!(ϵ_vals, moments_true[:,2], label="True")
plt2 = Plots.plot!(ϵ_vals, moments_lin[:,2], label="Linear")

plt3 = Plots.plot(ϵ_vals, moments_pert[:,3], label="Perturbed", title="3rd moment", xlabel="ϵ", ylabel="Moment", lw=2)
plt3 = Plots.plot!(ϵ_vals, moments_gen[:,3], label="KGMM")
plt3 = Plots.plot!(ϵ_vals, moments_true[:,3], label="True")
plt3 = Plots.plot!(ϵ_vals, moments_lin[:,3], label="Linear")

plt4 = Plots.plot(ϵ_vals, moments_pert[:,4], label="Perturbed", title="4th moment", xlabel="ϵ", ylabel="Moment", lw=2)
plt4 = Plots.plot!(ϵ_vals, moments_gen[:,4], label="KGMM")
plt4 = Plots.plot!(ϵ_vals, moments_true[:,4], label="True")
plt4 = Plots.plot!(ϵ_vals, moments_lin[:,4], label="Linear")

plt5 = Plots.plot(xax, pdf_gen[1], label="KGMM", title="ϵ=0.06", xlabel="X", ylabel="Density", lw=2, xlims=(-5.0, 5.0))
plt5 = Plots.plot!(xax, pdf_true[1], label="True score", lw=2)
plt5 = Plots.plot!(xax, pdf_unpert, label="True unpert", lw=2)
plt5 = Plots.plot!(xax, pdf_pert[1], label="True pert", lw=2)
plt5 = Plots.plot!(xax, pdf_lin[1], label="Gaussian", lw=2)

plt6 = Plots.plot(xax, pdf_gen[2], label="KGMM", title="ϵ=0.08", xlabel="X", ylabel="Density", lw=2, xlims=(-5.0, 5.0))
plt6 = Plots.plot!(xax, pdf_true[2], label="True score", lw=2)
plt6 = Plots.plot!(xax, pdf_unpert, label="True unpert", lw=2)
plt6 = Plots.plot!(xax, pdf_pert[2], label="True pert", lw=2)
plt6 = Plots.plot!(xax, pdf_lin[2], label="Gaussian", lw=2)

plt7 = Plots.plot(xax, pdf_gen[3], label="KGMM", title="ϵ=0.1", xlabel="X", ylabel="Density", lw=2, xlims=(-5.0, 5.0))
plt7 = Plots.plot!(xax, pdf_true[3], label="True score", lw=2)
plt7 = Plots.plot!(xax, pdf_unpert, label="True unpert", lw=2)
plt7 = Plots.plot!(xax, pdf_pert[3], label="True pert", lw=2)
plt7 = Plots.plot!(xax, pdf_lin[3], label="Gaussian", lw=2, ylims=(-0.01, 0.5))

plt8 = Plots.plot(xax, pdf_gen[4], label="KGMM", title="ϵ=0.12", xlabel="X", ylabel="Density", lw=2, xlims=(-5.0, 5.0))
plt8 = Plots.plot!(xax, pdf_true[4], label="True score", lw=2)
plt8 = Plots.plot!(xax, pdf_unpert, label="True unpert", lw=2)
plt8 = Plots.plot!(xax, pdf_pert[4], label="True pert", lw=2)

plt = Plots.plot(plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8, layout=(2, 4), size=(1600, 800))

plt

##

using GLMakie
using Distributions  # For the standard normal PDF

# Create figure with 1×5 layout (1D system, PDF + 4 moments)
fig = Figure(resolution=(1800, 600), font="CMU Serif", fontsize=24)

# Define common elements
colors = [:blue, :black, :red]
labels = ["Analytical", "Gaussian", "KGMM"]
time_axis = 0:dt*res_trj:n_tau*dt*res_trj

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
    push!(all_values, R_true[j,1,:])
    push!(all_values, R_lin[j,1,:])
    push!(all_values, R_gen_c[j,1,:])
    
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
    limits = (-5, 5, nothing, nothing),
)

# Get the KDE data
# Create Gaussian PDF with same x range as the true data
pdf_gauss = Distributions.pdf.(Normal(0, 1), xax)

# Plot PDF with transparency
lines!(axes[1], xax, pdf_obs, color=(colors[1], 0.5), linewidth=3)
lines!(axes[1], xax, pdf_gauss, color=(colors[2], 1.0), linewidth=3)
lines!(axes[1], xax, pdf_kgmm, color=(colors[3], 0.5), linewidth=3)

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

    # Plot response data with transparency
    lines!(axes[response_col], time_axis, R_true[j,1,:], color=(colors[1], 0.5), linewidth=3)
    lines!(axes[response_col], time_axis, R_lin[j,1,:], color=(colors[2], 1.0), linewidth=3)
    lines!(axes[response_col], time_axis, R_gen_c[j,1,:], color=(colors[3], 0.5), linewidth=3)
    
    # Add grid lines for readability
    axes[response_col].xgridvisible = true
    axes[response_col].ygridvisible = true
end

# Add unified legend at the bottom
Legend(fig[2, :],
    [LineElement(color=(c, 1.0), linewidth=3, linestyle=:solid)
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
#save("figures/HOR_figures/reduced_responses.png", fig, px_per_unit=2)  # Higher DPI for better print quality

fig

##

using GLMakie
using ColorSchemes

# Create a publication-quality figure with 2×4 layout
fig = Figure(resolution=(1800, 1200), font="CMU Serif", fontsize=24)


# Define common elements
linewidth = 3
moment_titles = ["1st moment", "2nd moment", "3rd moment", "4th moment"]
pdf_titles = ["ϵ=0.06", "ϵ=0.08", "ϵ=0.1", "ϵ=0.12"]
colors = [:blue, :black, :red, :green]  # Perturbed, Linear, KGMM, True
linestyles = [:solid, :solid, :solid, :solid]  # All solid lines
linewidths = [linewidth, linewidth, linewidth, linewidth]  # Same width for all
labels = ["Analytic", "Linear", "KGMM", "Perturbed"]

# ---------- TOP ROW: MOMENT PLOTS ----------
axes = Matrix{Axis}(undef, 2, 4)

for i in 1:4
    axes[1, i] = Axis(fig[1, i],
        xlabel = "ϵ",
        ylabel = "Moment",
        title = moment_titles[i],
        titlesize = 36,
        xlabelsize = 28,
        ylabelsize = 28,
        xticklabelsize = 24,
        yticklabelsize = 24
    )
    
    # Plot moment data
    lines!(axes[1, i], ϵ_vals, moments_true[:, i], color=(colors[1], 0.5), linewidth=linewidths[4], linestyle=linestyles[4])
    lines!(axes[1, i], ϵ_vals, moments_lin[:, i], color=colors[2], linewidth=linewidths[2], linestyle=linestyles[2])
    lines!(axes[1, i], ϵ_vals, moments_gen[:, i], color=(colors[3], 0.5), linewidth=linewidths[3], linestyle=linestyles[3])
    lines!(axes[1, i], ϵ_vals, moments_pert[:, i], color=colors[4], linewidth=linewidths[1], linestyle=linestyles[1])
    
    
    # Add grid for readability
    axes[1, i].xgridvisible = true
    axes[1, i].ygridvisible = true
end

# ---------- BOTTOM ROW: PDF PLOTS ----------
for i in 1:4
    axes[2, i] = Axis(fig[2, i],
        xlabel = "x",
        ylabel = "Density",
        title = pdf_titles[i],
        titlesize = 36,
        xlabelsize = 28,
        ylabelsize = 28,
        xticklabelsize = 24,
        yticklabelsize = 24,
        limits = (-5.0, 5.0, nothing, nothing)
    )
    
    # Plot PDF data
    lines!(axes[2, i], xax, pdf_true[i], color=(colors[1], 0.5), linewidth=linewidths[4], linestyle=linestyles[4])
    if i < 4  # The last plot doesn't have linear data
        lines!(axes[2, i], xax, pdf_lin[i], color=colors[2], linewidth=linewidths[2], linestyle=linestyles[2])
    end
    lines!(axes[2, i], xax, pdf_gen[i], color=(colors[3], 0.5), linewidth=linewidths[3], linestyle=linestyles[3])
    if i == 1
        lines!(axes[2, i], xax, pdf_pert[i], color=(colors[4], 0.5), linewidth=linewidths[4], linestyle=linestyles[4])
    else
        lines!(axes[2, i], xax, pdf_pert[i], color=colors[4], linewidth=linewidths[4], linestyle=linestyles[4])
    end
    # Add unperturbed distribution for reference (using a dashed green line)
    lines!(axes[2, i], xax, pdf_unpert, color=:green, linewidth=linewidth-1, linestyle=:dash)
    
    # Add grid for readability
    axes[2, i].xgridvisible = true
    axes[2, i].ygridvisible = true
    
    # For the third plot (ϵ=0.1), set specific y-limits as in the original
    if i == 3
        GLMakie.ylims!(axes[2, i], -0.01, 0.5)
    end
end

# Update legend with the new colors
Legend(fig[3, 1:4],
    [LineElement(color=c, linewidth=lw, linestyle=ls) 
     for (c, lw, ls) in zip(colors, linewidths, linestyles)] 
    ∪ [LineElement(color=:green, linewidth=linewidth-1, linestyle=:dash)],
    ["Analytic", "Gaussian", "KGMM", "Perturbed", "Unperturbed"],
    "Methods",
    orientation = :horizontal,
    titlesize = 32,
    labelsize = 28
)

# Adjust spacing
colgap!(fig.layout, 20)
rowgap!(fig.layout, 20)

# Add more bottom margin for better layout
fig.layout[4, :] = GridLayout(height=40)

# Add a title to the entire figure
Label(fig[0, :], "", fontsize=40)

# Save the figure with high resolution
save("figures/HOR_figures/reduced_moments.png", fig)

fig






##
# # Save all variables needed for the GLMakie plots
# save_variables_to_hdf5("data/HOR_data/reduced_add_analysis.h5", Dict(
#     # Simulation parameters
#     "dt" => dt,
#     "res_trj" => res_trj,
#     "n_tau" => n_tau,
#     "n_moments" => n_moments,
#     "M" => M,
#     "S" => S,
    
#     # Response functions
#     "R_true" => R_true,
#     "R_lin" => R_lin,
#     "R_gen" => R_gen,
#     "R_gen_c" => R_gen_c,
    
#     # Convolution results
#     "F_true" => F_true,
#     "F_lin" => F_lin,
#     "F_gen" => F_gen,
    
#     # Clustering results
#     "averages" => averages,
#     "centers" => centers,
#     "Nc" => Nc,
#     "σ_value" => σ_value,
    
#     # Moments analysis
#     "moments_pert" => moments_pert,
#     "moments_true" => moments_true,
#     "moments_gen" => moments_gen,
#     "moments_lin" => moments_lin,
#     "ϵ_vals" => ϵ_vals,
    
#     # PDF data
#     "pdf_obs" => pdf_obs,
#     "pdf_kgmm" => pdf_kgmm,
#     "pdf_unpert" => pdf_unpert,
#     "xax" => xax,
    
#     # PDFs at different perturbation values
#     "pdf_pert" => hcat([pdf_pert[i] for i in 1:length(pdf_pert)]...),
#     "pdf_true" => hcat([pdf_true[i] for i in 1:length(pdf_true)]...),
#     "pdf_gen" => hcat([pdf_gen[i] for i in 1:length(pdf_gen)]...),
#     "pdf_lin" => hcat([pdf_lin[i] for i in 1:length(pdf_lin)]...),
    
#     # Perturbation parameters
#     "ϵs" => ϵs,
#     "orders" => orders,
#     "orders_lin" => orders_lin,
    
#     # Model parameters
#     "a" => a,
#     "b" => b,
#     "c" => c,
#     "F_tilde" => F_tilde,
#     "s" => s,
#     "p_inf" => p_inf,
#     "m_inf" => m_inf
# ))


##

# Load the saved data
results = read_variables_from_hdf5("data/HOR_data/reduced_add_analysis.h5")

# Extract all variables from the results dictionary
R_true = results["R_true"]
R_lin = results["R_lin"]
R_gen = results["R_gen"]
R_gen_c = results["R_gen_c"]
F_true = results["F_true"]
F_lin = results["F_lin"]
F_gen = results["F_gen"]
S = results["S"]
M = results["M"]
σ_value = results["σ_value"]
averages = results["averages"]
centers = results["centers"]
Nc = results["Nc"]
moments_pert = results["moments_pert"]
moments_true = results["moments_true"]
moments_gen = results["moments_gen"]
moments_lin = results["moments_lin"]
ϵ_vals = results["ϵ_vals"]
pdf_obs = results["pdf_obs"]
pdf_kgmm = results["pdf_kgmm"]
pdf_unpert = results["pdf_unpert"]
xax = results["xax"]
dt = results["dt"]
res_trj = results["res_trj"]
n_tau = results["n_tau"]
n_moments = results["n_moments"]
ϵs = results["ϵs"]
orders = results["orders"]
orders_lin = results["orders_lin"]

# Convert stacked PDFs back to arrays of arrays
pdf_pert = [results["pdf_pert"][:,i] for i in 1:size(results["pdf_pert"], 2)]
pdf_true = [results["pdf_true"][:,i] for i in 1:size(results["pdf_true"], 2)]
pdf_gen = [results["pdf_gen"][:,i] for i in 1:size(results["pdf_gen"], 2)]
pdf_lin = [results["pdf_lin"][:,i] for i in 1:size(results["pdf_lin"], 2)]
