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

function F(x, t; F0=6.0, nu=1.0, c=10.0, b=10.0, Nk=4, Nj=10)
    # Coupling constant: c1 = c/b
    c1 = c / b

    # Allocate arrays for the derivatives of the slow and fast variables.
    dx = zeros(Nk)
    dy = zeros(Nk, Nj)
    
    # Extract the slow variables xₖ from the state vector.
    x_slow = x[1:Nk]
    
    # Extract the fast variables yₖ,ⱼ.
    # They are assumed to be stored after the slow variables.
    # Reshape them into an Nk×Nj matrix, where the k-th row corresponds to the block for xₖ.
    x_fast = reshape(x[Nk+1:end], (Nj, Nk))'  # now x_fast[k, j] corresponds to yₖ,ⱼ

    # Compute the forcing for the slow variables (Eq. 10):
    # dxₖ/dt = - xₖ₋₁ (xₖ₋₂ - xₖ₊₁) - nu*xₖ + F0 + c1 * (sum of fast variables for mode k)
    for k in 1:Nk
        # Use periodic boundary conditions:
        # For index arithmetic in Julia (1-indexed):
        im1 = mod(k - 2, Nk) + 1  # index for xₖ₋₁
        im2 = mod(k - 3, Nk) + 1  # index for xₖ₋₂
        ip1 = mod(k, Nk) + 1      # index for xₖ₊₁
        dx[k] = - x_slow[im1]*(x_slow[im2] - x_slow[ip1]) - nu*x_slow[k] + F0 + c1*sum(x_fast[k, :])
    end

    # Compute the forcing for the fast variables (Eq. 11):
    # dyₖ,ⱼ/dt = - c*b * yₖ,ⱼ₊₁ (yₖ,ⱼ₊₂ - yₖ,ⱼ₋₁) - c*nu*yₖ,ⱼ + c1*xₖ
    for k in 1:Nk
        for j in 1:Nj
            # Periodic indices in the fast sub-block:
            jm1 = mod(j - 2, Nj) + 1   # index for yₖ,ⱼ₋₁
            jp1 = mod(j, Nj) + 1       # index for yₖ,ⱼ₊₁
            jp2 = mod(j + 1, Nj) + 1     # index for yₖ,ⱼ₊₂
            dy[k, j] = - c*b * x_fast[k, jp1]*(x_fast[k, jp2] - x_fast[k, jm1]) -
                       c*nu * x_fast[k, j] + c1*x_slow[k]
        end
    end

    # Combine the slow and fast derivatives into a single vector.
    # The slow derivatives come first, then the fast derivatives (flattened in row-major order).
    return vcat(dx, vec(transpose(dy)))
end


function sigma(x, t; noise = 0.2)
    return noise
end

function normalize_f(f, x, t, M, S)
    return f(x .* S .+ M, t) .* S
end

dim = 4
dt = 0.005
Nsteps = 20000000
obs_nn = evolve(0.01 .* randn(44), dt, Nsteps, F, sigma; resolution = 2)[1:4,:]

M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S

# kde_true_12 = kde((obs[1,:], obs[2,:]))
# kde_true_13 = kde((obs[1,:], obs[3,:]))
# kde_true_14 = kde((obs[1,:], obs[4,:]))

# plt1 = Plots.heatmap(kde_true_12.x, kde_true_12.y, kde_true_12.density, xlabel="X", ylabel="Y", title="True PDF")
# plt2 = Plots.heatmap(kde_true_13.x, kde_true_13.y, kde_true_13.density, xlabel="X", ylabel="Y", title="True PDF")
# plt3 = Plots.heatmap(kde_true_14.x, kde_true_14.y, kde_true_14.density, xlabel="X", ylabel="Y", title="True PDF")
# Plots.plot(plt1, plt2, plt3, layout=(1, 3), size=(1200, 400))
# ##

autocov_obs = zeros(dim, 300)
for i in 1:dim
    autocov_obs[i,:] = autocovariance(obs[i,:]; timesteps=300)
end

autocov_obs_mean = mean(autocov_obs, dims=1)

plotly()
Plots.plot(autocov_obs_mean[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")

##
obs_uncorr = obs[:, 1:1:end]

plotly()
Plots.scatter(obs_uncorr[1,1:100:end], obs_uncorr[2,1:100:end], obs_uncorr[3,1:100:end], markersize=1, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

normalization = false
σ_value = 0.08

averages, _, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.0002, do_print=true, conv_param=0.002, normalization=normalization)

if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
end

targets_norm = [norm(averages[:,i]) for i in eachindex(centers[1,:])]
Plots.scatter(centers[1,:], centers[2,:], marker_z=targets_norm, color=:viridis)

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
#################### SAMPLES GENERATION ####################

score_clustered_xt(x,t) = score_clustered(x)
sigma_I(x,t) = 1.0

trj_clustered = evolve(zeros(dim), 0.1*dt, 10000000, score_clustered_xt, sigma_I; timestepper=:rk4, resolution=2, boundary=[-5,5])
# trj_score = evolve([0.0, 0.0], dt, 1000000, score_true, sigma_I; timestepper=:rk4, resolution=10, boundary=[-100,100])

kde_clustered_1 = kde(trj_clustered[1,:])
kde_true_1 = kde(obs[1,:])

kde_clustered_2 = kde(trj_clustered[2,:])
kde_true_2 = kde(obs[2,:])

kde_clustered_3 = kde(trj_clustered[3,:])
kde_true_3 = kde(obs[3,:])

kde_clustered_4 = kde(trj_clustered[4,:])
kde_true_4 = kde(obs[4,:])

kde_clustered_y = (kde_clustered_1.density .+ kde_clustered_2.density .+ kde_clustered_3.density .+ kde_clustered_4.density) ./ 4
kde_clustered_x = ([kde_clustered_1.x...] .+ [kde_clustered_2.x...] .+ [kde_clustered_3.x...] .+ [kde_clustered_4.x...]) ./ 4

kde_true_y = (kde_true_1.density .+ kde_true_2.density .+ kde_true_3.density .+ kde_true_4.density) ./ 4
kde_true_x = ([kde_true_1.x...] .+ [kde_true_2.x...] .+ [kde_true_3.x...] .+ [kde_true_4.x...]) ./ 4

Plots.plot(kde_clustered_x, kde_clustered_y, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
Plots.plot!(kde_true_x, kde_true_y, label="True", xlabel="X", ylabel="Density", title="True PDF")

##
# Compute bivariate PDFs for consecutive variables
kde_true_12 = kde((obs[1,:], obs[2,:]))
kde_clustered_12 = kde((trj_clustered[1,:], trj_clustered[2,:]))

kde_true_23 = kde((obs[2,:], obs[3,:]))
kde_clustered_23 = kde((trj_clustered[2,:], trj_clustered[3,:]))

kde_true_34 = kde((obs[3,:], obs[4,:]))
kde_clustered_34 = kde((trj_clustered[3,:], trj_clustered[4,:]))

kde_true_41 = kde((obs[4,:], obs[1,:]))
kde_clustered_41 = kde((trj_clustered[4,:], trj_clustered[1,:]))

# Compute bivariate PDFs for variables with one in between
kde_true_13 = kde((obs[1,:], obs[3,:]))
kde_clustered_13 = kde((trj_clustered[1,:], trj_clustered[3,:]))

kde_true_24 = kde((obs[2,:], obs[4,:]))
kde_clustered_24 = kde((trj_clustered[2,:], trj_clustered[4,:]))

# Compute average PDFs for consecutive variables
kde_true_consecutive_density = (kde_true_12.density + kde_true_23.density + kde_true_34.density + kde_true_41.density) ./ 4
kde_clustered_consecutive_density = (kde_clustered_12.density + kde_clustered_23.density + kde_clustered_34.density + kde_clustered_41.density) ./ 4

# Use one of the grids for plotting (they should be similar)
kde_consecutive_x = kde_true_12.x
kde_consecutive_y = kde_true_12.y

# Compute average PDFs for variables with one in between
kde_true_skip_density = (kde_true_13.density + kde_true_24.density) ./ 2
kde_clustered_skip_density = (kde_clustered_13.density + kde_clustered_24.density) ./ 2

# Use one of the grids for plotting
kde_skip_x = kde_true_13.x
kde_skip_y = kde_true_13.y

# Plot the results
plt1 = Plots.heatmap(kde_consecutive_x, kde_consecutive_y, kde_true_consecutive_density, 
                     xlabel="X", ylabel="Y", title="True PDF (Consecutive)")
plt2 = Plots.heatmap(kde_consecutive_x, kde_consecutive_y, kde_clustered_consecutive_density, 
                    xlabel="X", ylabel="Y", title="Generated PDF (Consecutive)")
plt3 = Plots.heatmap(kde_skip_x, kde_skip_y, kde_true_skip_density, 
                    xlabel="X", ylabel="Y", title="True PDF (Skip-One)")
plt4 = Plots.heatmap(kde_skip_x, kde_skip_y, kde_clustered_skip_density, 
                    xlabel="X", ylabel="Y", title="Generated PDF (Skip-One)")
Plots.plot(plt1, plt2, plt3, plt4, layout=(2, 2), size=(800, 800))

##
############## OBSERVATIONS GENERATION ####################

obs_trj = evolve(0.01 .* randn(36), dt, 20000, F, sigma; resolution = 2)[1:4,:]
obs_trj = ((obs_trj .- M) ./ S)[1:2,:]
score_trj = evolve(0.01 .* randn(4), dt, 20000, score_clustered_xt, sigma_I; resolution = 2)[1:2,:]

##

function save_lorenz96_plot_data(filename="data/GMM_data/lorenz96_2.h5")
    # Create directory structure if it doesn't exist
    mkpath(dirname(filename))
    
    # Calculate density_max for color scaling
    density_max = max(
        maximum(kde_true_consecutive_density), 
        maximum(kde_clustered_consecutive_density),
        maximum(kde_true_skip_density),
        maximum(kde_clustered_skip_density)
    )
    
    # Convert any range objects to arrays
    kde_true_x_array = collect(kde_true_x)
    kde_true_y_array = collect(kde_true_y)
    kde_clustered_x_array = collect(kde_clustered_x)
    kde_clustered_y_array = collect(kde_clustered_y)
    
    kde_consecutive_x_array = collect(kde_consecutive_x)
    kde_consecutive_y_array = collect(kde_consecutive_y)
    
    kde_skip_x_array = collect(kde_skip_x)
    kde_skip_y_array = collect(kde_skip_y)
    
    h5open(filename, "w") do file
        # Univariate PDFs
        write(file, "kde_true_x", kde_true_x_array)
        write(file, "kde_true_y", kde_true_y_array)
        write(file, "kde_clustered_x", kde_clustered_x_array)
        write(file, "kde_clustered_y", kde_clustered_y_array)
        
        # Bivariate consecutive PDFs
        write(file, "kde_consecutive_x", kde_consecutive_x_array)
        write(file, "kde_consecutive_y", kde_consecutive_y_array)
        write(file, "kde_true_consecutive_density", kde_true_consecutive_density)
        write(file, "kde_clustered_consecutive_density", kde_clustered_consecutive_density)
        
        # Bivariate skip PDFs
        write(file, "kde_skip_x", kde_skip_x_array)
        write(file, "kde_skip_y", kde_skip_y_array)
        write(file, "kde_true_skip_density", kde_true_skip_density)
        write(file, "kde_clustered_skip_density", kde_clustered_skip_density)
        
        # Color scaling
        write(file, "density_max", density_max)
        
        # Save additional metadata
        write(file, "dt", dt)
        write(file, "dim", dim)
        write(file, "σ_value", σ_value)
        
        # Save sample data points (for reference)
        write(file, "obs_trj", obs_trj)
        write(file, "score_trj", score_trj)
        
        # Save normalization parameters
        write(file, "M", M)
        write(file, "S", S)
    end
    
    println("Plot data saved to $filename")
end

function read_lorenz96_plot_data(filename="data/GMM_data/lorenz96.h5")
    data = Dict()
    
    h5open(filename, "r") do file
        # Univariate PDFs
        data["kde_true_x"] = read(file, "kde_true_x")
        data["kde_true_y"] = read(file, "kde_true_y")
        data["kde_clustered_x"] = read(file, "kde_clustered_x")
        data["kde_clustered_y"] = read(file, "kde_clustered_y")
        
        # Bivariate consecutive PDFs
        data["kde_consecutive_x"] = read(file, "kde_consecutive_x")
        data["kde_consecutive_y"] = read(file, "kde_consecutive_y")
        data["kde_true_consecutive_density"] = read(file, "kde_true_consecutive_density")
        data["kde_clustered_consecutive_density"] = read(file, "kde_clustered_consecutive_density")
        
        # Bivariate skip PDFs
        data["kde_skip_x"] = read(file, "kde_skip_x")
        data["kde_skip_y"] = read(file, "kde_skip_y")
        data["kde_true_skip_density"] = read(file, "kde_true_skip_density")
        data["kde_clustered_skip_density"] = read(file, "kde_clustered_skip_density")
        
        # Color scaling
        data["density_max"] = read(file, "density_max")
        
        # Read metadata
        data["dt"] = read(file, "dt")
        data["dim"] = read(file, "dim")
        data["σ_value"] = read(file, "σ_value")
        
        # Read sample data
        data["obs_trj"] = read(file, "obs_trj")
        data["score_trj"] = read(file, "score_trj")
        
        # Read normalization parameters
        data["M"] = read(file, "M")
        data["S"] = read(file, "S")
    end
    
    println("Plot data loaded from $filename")
    return data
end

# Save the data
save_lorenz96_plot_data()

##

function create_publication_plots(kde_true_x, kde_true_y, kde_clustered_x, kde_clustered_y,
    kde_consecutive_x, kde_consecutive_y, kde_true_consecutive_density,
    kde_clustered_consecutive_density, kde_skip_x, kde_skip_y,
    kde_true_skip_density, kde_clustered_skip_density, obs_trj, score_trj, dt)

# Increased size to accommodate the new row
fig = Figure(resolution=(2250, 2250), fontsize=32)

# Create a 3-row layout
main_layout = fig[1:3, 1:3] = GridLayout()

# Row 1: Time series plots (new addition)
time_panel = main_layout[1, 1:3] = GridLayout()

# Rows 2-3: Original layout with univariate and bivariate plots
left_panel = main_layout[2:3, 1] = GridLayout()
right_panel = main_layout[2:3, 2:3] = GridLayout()

# Create time series axes for x[1] and x[2]
time_ax1 = Axis(time_panel[1, 1:2], 
xlabel="t",
ylabel="x[k]",
title="x[k] Time Series",
titlesize=36,
xlabelsize=32,
ylabelsize=32)

# Plot time series data
n_points = 10000
time_vector = (1:2:n_points) .* dt .* 2

# First component
lines!(time_ax1, time_vector, obs_trj[1, 1:2:n_points], 
color=:red, linewidth=1, label="True")
lines!(time_ax1, time_vector, score_trj[1, 1:2:n_points], 
color=:blue, linewidth=1, label="GMM")
# axislegend(time_ax1, position=:rt, framevisible=true, bgcolor=:white, labelsize=22)

# Create univariate axis that spans the left half
univariate_ax = Axis(left_panel[1, 1], 
xlabel="x[k]",
ylabel="PDF",
title="Univariate PDF",
titlesize=36,
xlabelsize=32,
ylabelsize=32)

# Create colorbar for all heatmaps
density_max = max(
maximum(kde_true_consecutive_density), 
maximum(kde_clustered_consecutive_density),
maximum(kde_true_skip_density),
maximum(kde_clustered_skip_density)
)

# Univariate plot
lines!(univariate_ax, kde_true_x, kde_true_y, color=:red, linewidth=2, label="True")
lines!(univariate_ax, kde_clustered_x, kde_clustered_y, color=:blue, linewidth=2, label="KGMM")
axislegend(univariate_ax, position=:rt, framevisible=true, bgcolor=:white, labelsize=32)

# Create a 2×2 grid of heatmaps in the right half
heatmap_axes = [
Axis(right_panel[1, 1], titlesize=36, xlabelsize=32, ylabelsize=32),
Axis(right_panel[1, 2], titlesize=36, xlabelsize=32, ylabelsize=32),
Axis(right_panel[2, 1], titlesize=36, xlabelsize=32, ylabelsize=32),
Axis(right_panel[2, 2], titlesize=36, xlabelsize=32, ylabelsize=32)
]

# Set labels and titles for heatmaps
heatmap_axes[1].title = "True (x[k]-x[k+1]) PDF"
heatmap_axes[2].title = "KGMM (x[k]-x[k+1]) PDF"
heatmap_axes[3].title = "True (x[k]-x[k+2]) PDF"
heatmap_axes[4].title = "KGMM (x[k]-x[k+2]) PDF"

for ax in heatmap_axes
ax.xlabel = "x[k]"
end

heatmap_axes[1].ylabel = "x[k+1]"
heatmap_axes[2].ylabel = "x[k+1]"
heatmap_axes[3].ylabel = "x[k+2]"
heatmap_axes[4].ylabel = "x[k+2]"

# Create heatmaps
hm1 = GLMakie.heatmap!(heatmap_axes[1], kde_consecutive_x, kde_consecutive_y, kde_true_consecutive_density, 
colormap=:viridis, colorrange=(0, density_max))
hm2 = GLMakie.heatmap!(heatmap_axes[2], kde_consecutive_x, kde_consecutive_y, kde_clustered_consecutive_density, 
colormap=:viridis, colorrange=(0, density_max))
hm3 = GLMakie.heatmap!(heatmap_axes[3], kde_skip_x, kde_skip_y, kde_true_skip_density, 
colormap=:viridis, colorrange=(0, density_max))
hm4 = GLMakie.heatmap!(heatmap_axes[4], kde_skip_x, kde_skip_y, kde_clustered_skip_density, 
colormap=:viridis, colorrange=(0, density_max))

# Add a single colorbar for all heatmaps with larger font
Colorbar(fig[2:3, 4], limits=(0, density_max), colormap=:viridis, ticklabelsize=32,
width=30)

# Adjust layout with more space for larger fonts
colgap!(main_layout, 15)
rowgap!(main_layout, 15)
colgap!(right_panel, 15)
rowgap!(right_panel, 15)
colgap!(time_panel, 15)

# Balance row sizes
rowsize!(main_layout, 1, Relative(0.3))  # Time series row
rowsize!(main_layout, 2, Relative(0.35)) # First row of original plots
rowsize!(main_layout, 3, Relative(0.35)) # Second row of original plots

# Add a title at the top
Label(fig[0, 1:3], text=" ", fontsize=34, font=:bold)

return fig
end

# Example of how to load and use the data to create the plot
function create_plot_from_file(filename="data/GMM_data/lorenz96_2.h5")
    data = read_lorenz96_plot_data(filename)
    
    return create_publication_plots(
        data["kde_true_x"], 
        data["kde_true_y"], 
        data["kde_clustered_x"], 
        data["kde_clustered_y"],
        data["kde_consecutive_x"], 
        data["kde_consecutive_y"], 
        data["kde_true_consecutive_density"],
        data["kde_clustered_consecutive_density"], 
        data["kde_skip_x"], 
        data["kde_skip_y"],
        data["kde_true_skip_density"], 
        data["kde_clustered_skip_density"],
        data["obs_trj"],
        data["score_trj"],
        data["dt"],
    )
end

# Create the plot from saved data
fig = create_plot_from_file()
save("figures/GMM_figures/lorenz96.png", fig)