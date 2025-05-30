using Pkg
Pkg.activate(".")
Pkg.instantiate()

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
using Plots
using QuadGK
using BSON
using GLMakie
using GLMakie: xlims!, ylims!
using StatsBase
using MarkovChainHammer
import MarkovChainHammer.Trajectory: ContinuousTimeEmpiricalProcess
import LaTeXStrings
using ColorSchemes
##


function F(x, t; A1=1.0, A2=1.2, B1=0.6, B2=0.3)
    ∇U1 = 2 * (x[1] + A1) * (x[1] - A1)^2 + 2 * (x[1] - A1) * (x[1] + A1)^2 + B1
    ∇U2 = 2 * (x[2] + A2) * (x[2] - A2)^2 + 2 * (x[2] - A2) * (x[2] + A2)^2 + B2
    return [-∇U1, -∇U2]
end

function sigma(x, t)
    return [1.0, 1.0]
end

function normalize_f(f, x, t, M, S)
    return f(x .* S .+ M, t) .* S
end

dt = 0.05
dim = 2

Nsteps = 100000000
obs_nn = evolve([0.0, 0.0], dt, Nsteps, F, sigma; resolution = 10)

M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S

autocov_obs = zeros(dim, 300)
for i in 1:dim
    autocov_obs[i,:] = autocovariance(obs[i,:]; timesteps=300)
end

autocov_obs_mean = mean(autocov_obs, dims=1)

Plots.plot(autocov_obs_mean[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")

##
obs_uncorr = obs[:, 1:1:end]

gr()
Plots.scatter(obs_uncorr[1,1:1:10000], obs_uncorr[2,1:1:10000], markersize=1, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

normalization = false
σ_value = 0.05

averages, _, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.001, do_print=true, conv_param=0.005, normalization=normalization)

if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
end

targets_norm = [norm(averages[:,i]) for i in eachindex(centers[1,:])]
Plots.scatter(centers[1,:], centers[2,:], marker_z=targets_norm, color=:viridis)
##
function score_true(x, t)
    return normalize_f(F, x, t, M, S)
end


averages_true = hcat([score_true(centers[:,i], 0.0) for i in 1:Nc]...)
averages_gen =  .- averages ./ σ_value # hcat([score_clustered(averages[:,i]) for i in eachindex(centers[1,:])]...)
diffs_norm = [norm(averages_true[:,i] - averages_gen[:,i]) for i in eachindex(centers[1,:])]

plt1 = Plots.scatter(averages_gen[1,:], averages_gen[2,:], label="Clustered", xlabel="X", ylabel="Y", title="Averages")
plt1 = Plots.scatter!(averages_true[1,:], averages_true[2,:], label="True", xlabel="X", ylabel="Y", title="Averages")

plt2 = Plots.scatter(centers[1,:], averages_gen[1,:], label="Clustered", xlabel="X", ylabel="score X", title="Averages")
plt2 = Plots.scatter!(centers[1,:], averages_true[1,:], label="True", xlabel="X", ylabel="score X", title="Averages")

plt3 = Plots.scatter(centers[2,:], averages_gen[2,:], label="Clustered", xlabel="Y", ylabel="score Y", title="Averages")
plt3 = Plots.scatter!(centers[2,:], averages_true[2,:], label="True", xlabel="Y", ylabel="score Y", title="Averages")

plt4 = Plots.scatter(centers[1,:], centers[2,:], marker_z=diffs_norm, legend=false, xlabel="X", ylabel="Y", title="Averages")

Plots.plot(plt1, plt2, plt3, plt4)

##
#################### TRAINING WITH CLUSTERING LOSS ####################

@time nn_clustered, loss_clustered = train(inputs_targets, 1000, 32, [dim, 128, 64, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value
Plots.plot(loss_clustered)


##
#################### SAMPLES GENERATION ####################

score_clustered_xt(x,t) = score_clustered(x)
sigma_I(x,t) = 1.0

trj_clustered = evolve([0.0, 0.0], 0.1*dt, 10000000, score_clustered_xt, sigma_I; timestepper=:rk4, resolution=10, boundary=[-100,100])
# trj_score = evolve([0.0, 0.0], dt, 1000000, score_true, sigma_I; timestepper=:rk4, resolution=10, boundary=[-100,100])

kde_clustered_x = kde(trj_clustered[1,:])
kde_true_x = kde(obs[1,:])

kde_clustered_y = kde(trj_clustered[2,:])
kde_true_y = kde(obs[2,:])

gr()
plt1 = Plots.plot(kde_clustered_x.x, kde_clustered_x.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
plt1 = Plots.plot!(kde_true_x.x, kde_true_x.density, label="True", xlabel="X", ylabel="Density", title="True PDF")

plt2 = Plots.plot(kde_clustered_y.x, kde_clustered_y.density, label="Observed", xlabel="Y", ylabel="Density", title="Observed PDF")
plt2 = Plots.plot!(kde_true_y.x, kde_true_y.density, label="True", xlabel="Y", ylabel="Density", title="True PDF")

Plots.plot(plt1, plt2, layout=(2, 1), size=(800, 600))
##
############## OBSERVATIONS GENERATION ####################

obs_trj = evolve([0.0, 0.0], dt, 10000, F, sigma; timestepper=:rk4)
obs_trj = (obs_trj .- M) ./ S
score_trj = evolve([0.0, 0.0], 0.1*dt, 100000, score_clustered_xt, sigma_I; timestepper=:rk4, resolution=10)
##

# Create vector field data
n_grid = 50
d_grid = 1/10
c_grid = [((n_grid+1)*d_grid)/2, ((n_grid+1)*d_grid)/2]

x = range(-c_grid[1], stop=c_grid[1], length=n_grid)
y = range(-c_grid[2], stop=c_grid[2], length=n_grid)

u_true = zeros(n_grid, n_grid)
v_true = zeros(n_grid, n_grid)
u_clustered = zeros(n_grid, n_grid)
v_clustered = zeros(n_grid, n_grid)

for i in 1:n_grid
    for j in 1:n_grid
        u_true[j, i], v_true[j, i] = score_true([x[i], y[j]], 0.0)
        u_clustered[j, i], v_clustered[j, i] = score_clustered([x[i], y[j]])
    end
end

# Calculate vector field magnitudes
mag_true = sqrt.(u_true.^2 .+ v_true.^2)
mag_clustered = sqrt.(u_clustered.^2 .+ v_clustered.^2)

# Calculate bivariate KDEs
kde_obs_xy = kde((obs[1,:], obs[2,:]))
kde_clustered_xy = kde((trj_clustered[1,:], trj_clustered[2,:]))

##

# Create directory structure if it doesn't exist
mkpath("data/GMM_data")

# Write data to HDF5 file
h5open("data/GMM_data/2D_potential.h5", "w") do file
    # Write scalar values and parameters
    write(file, "dt", dt)
    write(file, "dim", dim)
    write(file, "Nsteps", Nsteps)
    write(file, "Nc", Nc)
    
    # Write vector field grid data
    write(file, "x", collect(x))
    write(file, "y", collect(y))
    
    # Write force field data
    write(file, "u_true", u_true)
    write(file, "v_true", v_true)
    write(file, "u_clustered", u_clustered)
    write(file, "v_clustered", v_clustered)
    
    # Write KDE data for univariate distributions
    write(file, "kde_true_x_x", collect(kde_true_x.x))
    write(file, "kde_true_x_density", collect(kde_true_x.density))
    write(file, "kde_clustered_x_x", collect(kde_clustered_x.x))
    write(file, "kde_clustered_x_density", collect(kde_clustered_x.density))
    
    write(file, "kde_true_y_x", collect(kde_true_y.x))
    write(file, "kde_true_y_density", collect(kde_true_y.density))
    write(file, "kde_clustered_y_x", collect(kde_clustered_y.x))
    write(file, "kde_clustered_y_density", collect(kde_clustered_y.density))
    
    # Write bivariate KDE data
    write(file, "kde_obs_xy_x", collect(kde_obs_xy.x))
    write(file, "kde_obs_xy_y", collect(kde_obs_xy.y))
    write(file, "kde_obs_xy_density", kde_obs_xy.density)
    
    write(file, "kde_clustered_xy_x", collect(kde_clustered_xy.x))
    write(file, "kde_clustered_xy_y", collect(kde_clustered_xy.y))
    write(file, "kde_clustered_xy_density", kde_clustered_xy.density)
    
    # Write original data samples
    write(file, "obs_trj", obs_trj)
    write(file, "score_trj", score_trj)
    
    # Write statistics
    write(file, "M", M)
    write(file, "S", S)
    
    # Write averages and centers
    write(file, "averages", averages)
    write(file, "centers", centers)
    write(file, "averages_true", averages_true)
    write(file, "averages_gen", averages_gen)
end

println("Data saved to data/GMM_data/2D_potential.h5")
##
# Function to read the data without using KDEResult
function read_2D_potential_data(filename="data/GMM_data/2D_potential.h5")
    data = Dict()
    h5open(filename, "r") do file
        # Read scalar values
        data["dt"] = read(file, "dt")
        data["dim"] = read(file, "dim")
        data["Nsteps"] = read(file, "Nsteps")
        data["Nc"] = read(file, "Nc")
        
        # Read vector field grid data
        data["x"] = read(file, "x")
        data["y"] = read(file, "y")
        
        # Read force field data
        data["u_true"] = read(file, "u_true")
        data["v_true"] = read(file, "v_true")
        data["u_clustered"] = read(file, "u_clustered")
        data["v_clustered"] = read(file, "v_clustered")
        
        # Read KDE data for univariate distributions as separate components
        data["kde_true_x_x"] = read(file, "kde_true_x_x")
        data["kde_true_x_density"] = read(file, "kde_true_x_density")
        data["kde_clustered_x_x"] = read(file, "kde_clustered_x_x")
        data["kde_clustered_x_density"] = read(file, "kde_clustered_x_density")
        
        data["kde_true_y_x"] = read(file, "kde_true_y_x")
        data["kde_true_y_density"] = read(file, "kde_true_y_density")
        data["kde_clustered_y_x"] = read(file, "kde_clustered_y_x")
        data["kde_clustered_y_density"] = read(file, "kde_clustered_y_density")
        
        # Read bivariate KDE data as separate components
        data["kde_obs_xy_x"] = read(file, "kde_obs_xy_x")
        data["kde_obs_xy_y"] = read(file, "kde_obs_xy_y")
        data["kde_obs_xy_density"] = read(file, "kde_obs_xy_density")
        
        data["kde_clustered_xy_x"] = read(file, "kde_clustered_xy_x")
        data["kde_clustered_xy_y"] = read(file, "kde_clustered_xy_y")
        data["kde_clustered_xy_density"] = read(file, "kde_clustered_xy_density")
        
        # Read original data samples
        data["obs_trj"] = read(file, "obs_trj")
        data["score_trj"] = read(file, "score_trj")
        
        # Read statistics
        data["M"] = read(file, "M")
        data["S"] = read(file, "S")
        
        # Read averages and centers
        data["averages"] = read(file, "averages")
        data["centers"] = read(file, "centers")
        data["obs_trj"] = read(file, "obs_trj")
        data["averages_gen"] = read(file, "averages_gen")
    end
    return data
end

# Read the data back
data = read_2D_potential_data()

# Extract variables for use in plotting
dt = data["dt"]
dim = data["dim"]
Nsteps = data["Nsteps"]
Nc = data["Nc"]
x = data["x"]
y = data["y"]
u_true = data["u_true"]
v_true = data["v_true"]
u_clustered = data["u_clustered"]
v_clustered = data["v_clustered"]

# Use the KDE data directly without KDEResult
kde_true_x_x = data["kde_true_x_x"]
kde_true_x_density = data["kde_true_x_density"]
kde_clustered_x_x = data["kde_clustered_x_x"]
kde_clustered_x_density = data["kde_clustered_x_density"]

kde_true_y_x = data["kde_true_y_x"]
kde_true_y_density = data["kde_true_y_density"]
kde_clustered_y_x = data["kde_clustered_y_x"]
kde_clustered_y_density = data["kde_clustered_y_density"]

kde_obs_xy_x = data["kde_obs_xy_x"]
kde_obs_xy_y = data["kde_obs_xy_y"]
kde_obs_xy_density = data["kde_obs_xy_density"]

kde_clustered_xy_x = data["kde_clustered_xy_x"]
kde_clustered_xy_y = data["kde_clustered_xy_y"]
kde_clustered_xy_density = data["kde_clustered_xy_density"]

obs_sample = data["obs_trj"]
trj_clustered_sample = data["score_trj"]
M = data["M"]
S = data["S"]
averages = data["averages"]
centers = data["centers"]
averages_true = data["averages"]
averages_gen = data["averages_gen"]

println("Data loaded successfully")

##


# Create main figure with wider aspect ratio for 3x8 layout 
# (wider to accommodate colorbars without using insert!)
fig = GLMakie.Figure(size=(2500, 1400), fontsize=28)

# Create layout for 3x8 grid that includes space for colorbars from the start
grid = fig[1:3, 1:8] = GLMakie.GridLayout(3, 8)

# Create the new time series plots in the top row with equal sizes
ax_time_x = GLMakie.Axis(grid[1, 1:4], 
    xlabel="t", ylabel="x",
    title="x Time Series",
    titlesize=36, xlabelsize=32, ylabelsize=32)

ax_time_y = GLMakie.Axis(grid[1, 5:8], 
    xlabel="t", ylabel="y",
    title="y Time Series",
    titlesize=36, xlabelsize=32, ylabelsize=32)

# Force field plots (first two columns)
ax_vf_true = GLMakie.Axis(grid[2, 1:2], 
    xlabel="x", ylabel="y",
    title="True Score",
    titlesize=36, xlabelsize=32, ylabelsize=32)

ax_vf_clustered = GLMakie.Axis(grid[3, 1:2], 
    xlabel="x", ylabel="y",
    title="KGMM Score",
    titlesize=36, xlabelsize=32, ylabelsize=32)

# PDFs (columns 4-5)
ax_pdf_x = GLMakie.Axis(grid[2, 4:5],
    xlabel="x", ylabel="PDF",
    title="Univariate x PDFs",
    titlesize=36, xlabelsize=32, ylabelsize=32)

ax_pdf_y = GLMakie.Axis(grid[3, 4:5],
    xlabel="y", ylabel="PDF", 
    title="Univariate y PDFs",
    titlesize=36, xlabelsize=32, ylabelsize=32)

# Bivariate PDFs (last two columns)
ax_true_xy = GLMakie.Axis(grid[2, 6:7],
    xlabel="x", ylabel="y",
    title="Bivariate True PDF",
    titlesize=36, xlabelsize=32, ylabelsize=32)

ax_clustered_xy = GLMakie.Axis(grid[3, 6:7],
    xlabel="x", ylabel="y",
    title="Bivariate KGMM PDF",
    titlesize=36, xlabelsize=32, ylabelsize=32)

# Colorbars in dedicated columns
force_bar = GLMakie.Colorbar(grid[2:3, 3], 
              colormap=:viridis, 
              limits=(0, vf_vmax),
              labelsize=32,
              vertical=true,
              width=20)

pdf_bar = GLMakie.Colorbar(grid[2:3, 8], 
              colormap=:viridis, 
              limits=(0, pdf_vmax),
              labelsize=32,
              vertical=true,
              width=20)

# Adjust column widths 
GLMakie.colsize!(grid, 3, GLMakie.Relative(0.05))  # Narrow column for force colorbar
GLMakie.colsize!(grid, 8, GLMakie.Relative(0.05))  # Narrow column for PDF colorbar
GLMakie.colsize!(grid, 1, GLMakie.Relative(0.15))  # Force field column 1
GLMakie.colsize!(grid, 2, GLMakie.Relative(0.15))  # Force field column 2
GLMakie.colsize!(grid, 4, GLMakie.Relative(0.15))  # PDF column 1
GLMakie.colsize!(grid, 5, GLMakie.Relative(0.15))  # PDF column 2
GLMakie.colsize!(grid, 6, GLMakie.Relative(0.15))  # Bivariate column 1
GLMakie.colsize!(grid, 7, GLMakie.Relative(0.15))  # Bivariate column 2

# Plot time series data in the top row
# Select a subset of time points for better visualization
time_subset = 1:1:1000
time_points = collect(time_subset) .* dt

# X component time series
GLMakie.lines!(ax_time_x, time_points, obs_sample[1, time_subset], 
       color=:red, linewidth=1, label="True")
GLMakie.lines!(ax_time_x, time_points, trj_clustered_sample[1, time_subset], 
       color=:blue, linewidth=1, label="KGMM")

# Y component time series  
GLMakie.lines!(ax_time_y, time_points, obs_sample[2, time_subset], 
       color=:red, linewidth=1)
GLMakie.lines!(ax_time_y, time_points, trj_clustered_sample[2, time_subset], 
       color=:blue, linewidth=1)

# Set same x-axis limits for both time series plots
time_xlims = (minimum(time_points), maximum(time_points))
GLMakie.xlims!(ax_time_x, time_xlims)
GLMakie.xlims!(ax_time_y, time_xlims)

# Vector field plots
x_points = repeat(x, outer=length(y))
y_points = repeat(y, inner=length(x))

u_true_flat = vec(u_true')
v_true_flat = vec(v_true')
u_clustered_flat = vec(u_clustered')
v_clustered_flat = vec(v_clustered')

mag_true_flat = sqrt.(u_true_flat.^2 .+ v_true_flat.^2)
mag_clustered_flat = sqrt.(u_clustered_flat.^2 .+ v_clustered_flat.^2)

scale = 1.0
u_true_norm = u_true_flat ./ max.(mag_true_flat, 1e-10) .* scale
v_true_norm = v_true_flat ./ max.(mag_true_flat, 1e-10) .* scale
u_clustered_norm = u_clustered_flat ./ max.(mag_clustered_flat, 1e-10) .* scale
v_clustered_norm = v_clustered_flat ./ max.(mag_clustered_flat, 1e-10) .* scale

# Use the same max value for both vector fields for consistent coloring
vf_vmax = maximum(mag_clustered_flat)

# Arrow plots with identical colormap range
arrow_true = GLMakie.arrows!(ax_vf_true, x_points, y_points, u_true_norm, v_true_norm, 
       arrowsize=1,
       linewidth=1,
       color=mag_true_flat,
       colormap=:viridis,
       colorrange=(0, vf_vmax))

arrow_clustered = GLMakie.arrows!(ax_vf_clustered, x_points, y_points, u_clustered_norm, v_clustered_norm, 
       arrowsize=1,
       linewidth=1,
       color=mag_clustered_flat,
       colormap=:viridis,
       colorrange=(0, vf_vmax))

# Univariate PDFs - add legend to BOTH plots
GLMakie.lines!(ax_pdf_x, kde_true_x_x, kde_true_x_density, 
       color=:red, linewidth=2, label="True")
GLMakie.lines!(ax_pdf_x, kde_clustered_x_x, kde_clustered_x_density, 
       color=:blue, linewidth=2, label="KGMM")
GLMakie.axislegend(ax_pdf_x, position=:lt, labelsize=32)

GLMakie.lines!(ax_pdf_y, kde_true_y_x, kde_true_y_density, 
       color=:red, linewidth=2, label="True")
GLMakie.lines!(ax_pdf_y, kde_clustered_y_x, kde_clustered_y_density, 
       color=:blue, linewidth=2, label="KGMM")

# Bivariate PDFs with consistent limits
# Use the same max value for both PDFs for consistent coloring
pdf_vmax = max(maximum(kde_obs_xy_density), maximum(kde_clustered_xy_density))

# Individual heatmaps with consistent color ranges
hm1 = GLMakie.heatmap!(ax_true_xy, 
         kde_obs_xy_x, kde_obs_xy_y, kde_obs_xy_density,
         colormap=:viridis, 
         colorrange=(0, pdf_vmax))

hm2 = GLMakie.heatmap!(ax_clustered_xy, 
         kde_clustered_xy_x, kde_clustered_xy_y, kde_clustered_xy_density,
         colormap=:viridis, 
         colorrange=(0, pdf_vmax))

# Set limits for bivariate plots if needed
# Use the same limits for consistent comparison
x_limits = (minimum(kde_obs_xy_x), maximum(kde_obs_xy_x))
y_limits = (minimum(kde_obs_xy_y), maximum(kde_obs_xy_y))
GLMakie.xlims!(ax_true_xy, x_limits)
GLMakie.ylims!(ax_true_xy, y_limits)
GLMakie.xlims!(ax_clustered_xy, x_limits)
GLMakie.ylims!(ax_clustered_xy, y_limits)

# Adjust spacing between plots
GLMakie.colgap!(grid, 10)  # Horizontal spacing
GLMakie.rowgap!(grid, 15)  # Vertical spacing

# Add a title
GLMakie.Label(fig[0, 1:8], text="", fontsize=40, font=:bold)

# Make sure the figures directory exists
mkpath("figures/GMM_figures")

# Save the figure
GLMakie.save("figures/GMM_figures/2D_potential.png", fig)

fig