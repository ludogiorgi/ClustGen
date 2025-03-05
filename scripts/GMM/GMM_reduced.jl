using Pkg
Pkg.activate(".")
Pkg.instantiate()
##

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

function F(x,t)
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
    return f(x .* S .+ M) .* S
end

function pdf_score(x, s)
    u = x[1]
    unnorm(u_val) = begin
        I, _ = quadgk(v -> s(u),
                      0, u_val)
        exp(-2 * I)
    end
    norm, _ = quadgk(unnorm, -Inf, Inf)
    
    return unnorm(u) / norm
end

dim = 1
dt = 0.01
Nsteps = 10000000
obs_nn = evolve([0.0], dt, Nsteps, F, sigma1, sigma2; timestepper=:rk4)
M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S

autocov_obs = zeros(dim, 300)
for i in 1:dim
    autocov_obs[i,:] = autocovariance(obs[i,:]; timesteps=300)
end

kde_obs = kde(obs[200:end])

autocov_obs_mean = mean(autocov_obs, dims=1)

plt1 = Plots.plot(autocov_obs_mean[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")
plt2 = Plots.plot(kde_obs.x, kde_obs.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")

Plots.plot(plt1, plt2, layout=(2, 1), size=(800, 800))

##
obs_uncorr = obs[:, 10:1:end]

Plots.scatter(obs_uncorr[1,1:10000], markersize=2, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

function score_true_norm(x)
    return normalize_f(score_true, x, M, S)
end

normalization = false
σ_value = 0.01

Nc
averages, averages_residual, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.002, do_print=true, conv_param=0.001, normalization=normalization)

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
scores_true = [-score_true_norm(centers_sorted[i])[1] for i in eachindex(centers_sorted)]

Plots.scatter(centers_sorted, scores, color=:blue)
Plots.plot!(centers_sorted, scores_true, color=:red)
##
#################### TRAINING WITH CLUSTERING LOSS ####################

@time nn_clustered, loss_clustered = train(inputs_targets, 2000, 32, [dim, 128, 64, dim]; use_gpu=true, activation=swish, last_activation=identity)
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

Plots.plot(xax, s_true, label="True", xlabel="X", ylabel="Force", title="Forces")
Plots.plot!(xax, s_gen, label="Learned")

##
#################### SAMPLES GENERATION ####################

score_clustered_xt(x,t) = score_clustered(x)
score_true_xt(x,t) = score_true_norm(x)
sigma_I(x,t) = 1.0

trj_clustered = evolve([0.0], dt, Nsteps, score_clustered_xt, sigma_I; timestepper=:rk4, resolution=1)
trj_true = evolve([0.0], dt, Nsteps, score_true_xt, sigma_I; timestepper=:rk4, resolution=1)


kde_clustered = kde(trj_clustered[:])
kde_true = kde(trj_true[:])

Plots.plot(kde_clustered.x, kde_clustered.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
Plots.plot!(kde_true.x, kde_true.density, label="True", xlabel="X", ylabel="Density", title="True PDF")

kde_clustered
##  
# Create directory structure if it doesn't exist
mkpath("data/GMM_data")

# Write data to HDF5 file
h5open("data/GMM_data/reduced.h5", "w") do file
    # Write scalar values
    write(file, "dt", dt)
    write(file, "Nsteps", Nsteps)
    write(file, "Nc", Nc)
    
    # Write arrays
    write(file, "averages", averages)
    write(file, "averages_residual", averages_residual)
    write(file, "centers", centers)
    write(file, "kde_clustered_x", [kde_clustered.x...])
    write(file, "kde_clustered_density", kde_clustered.density)
    write(file, "kde_true_x", [kde_true.x...])
    write(file, "kde_true_density", kde_true.density)
    write(file, "xax", xax)
    write(file, "s_true", s_true)
    write(file, "s_gen", s_gen)
    write(file, "obs_uncorr_subset", obs_uncorr[:,1:100000])
end

println("Data saved to data/GMM_data/reduced.h5")
##

# Code to read the data
function read_reduced_data(filename="data/GMM_data/reduced.h5")
    data = Dict()
    h5open(filename, "r") do file
        # Read scalar values
        data["dt"] = read(file, "dt")
        data["Nsteps"] = read(file, "Nsteps")
        data["Nc"] = read(file, "Nc")
        
        # Read arrays
        data["averages"] = read(file, "averages")
        data["averages_residual"] = read(file, "averages_residual")
        data["centers"] = read(file, "centers")
        data["kde_clustered_x"] = read(file, "kde_clustered_x")
        data["kde_clustered_density"] = read(file, "kde_clustered_density")
        data["kde_true_x"] = read(file, "kde_true_x")
        data["kde_true_density"] = read(file, "kde_true_density")
        data["xax"] = read(file, "xax")
        data["s_true"] = read(file, "s_true")
        data["s_gen"] = read(file, "s_gen")
        data["obs_uncorr_subset"] = read(file, "obs_uncorr_subset")
    end
    return data
end

data = read_reduced_data()

# Extract all variables from the data dictionary
dt = data["dt"]
Nsteps = data["Nsteps"]
Nc = data["Nc"]
averages = data["averages"]
averages_residual = data["averages_residual"]
centers = data["centers"]
kde_clustered_x = data["kde_clustered_x"]
kde_clustered_density = data["kde_clustered_density"]
kde_true_x = data["kde_true_x"]
kde_true_density = data["kde_true_density"]
xax = data["xax"]
s_true = data["s_true"]
s_gen = data["s_gen"]
obs_uncorr_subset = data["obs_uncorr_subset"]

##


fig = Figure(resolution=(1200, 300), font="CMU Serif")

# Define common elements
colors = [:red, :blue]
labels = ["True", "GMM"]

# Create subplots
ax0 = Axis(fig[1,1], 
    xlabel="t", ylabel="x",
    title="Observations",
    titlesize=20,
    xlabelsize=16, ylabelsize=16)

ax1 = Axis(fig[1,2], 
    xlabel="x", ylabel="Score",
    title="Scores",
    titlesize=20,
    xlabelsize=16, ylabelsize=16)

ax2 = Axis(fig[1,3], 
    xlabel="x", ylabel="PDF",
    title="PDFs",
    titlesize=20,
    xlabelsize=16, ylabelsize=16)

# Plot data
n_obs = min(10000, size(obs_uncorr_subset, 2))
lines!(ax0, dt:dt:dt*n_obs, obs_uncorr_subset[1, 1:n_obs], color=:black, linewidth=1)

lines!(ax1, xax, s_true, color=colors[1], linewidth=2)
lines!(ax1, xax, s_gen, color=colors[2], linewidth=2)

lines!(ax2, kde_true_x, kde_true_density, color=colors[1], linewidth=2)
lines!(ax2, kde_clustered_x, kde_clustered_density, color=colors[2], linewidth=2)

# Add a more compact legend
Legend(fig[1, :], 
    [LineElement(color=c, linewidth=2) for c in colors],
    labels,
    orientation=:horizontal,
    tellheight=false,
    tellwidth=false,
    halign=:right,
    valign=:top,
    margin=(10, 10, 10, 10))

# Adjust spacing
colgap!(fig.layout, 20)

save("figures/GMM_figures/reduced.png", fig)

fig