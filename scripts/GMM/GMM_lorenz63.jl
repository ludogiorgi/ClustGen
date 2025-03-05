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

function F(x, t; σ=10.0, ρ=28.0, β=8/3)
    F1 = σ * (x[2] - x[1])
    F2 = x[1] * (ρ - x[3]) - x[2]
    F3 = x[1] * x[2] - β * x[3]
    return [F1, F2, F3]
end

function sigma(x, t; noise = 5.0)
    sigma1 = noise
    sigma2 = noise
    sigma3 = noise
    return [sigma1, sigma2, sigma3]
end

function normalize_f(f, x, t, M, S)
    return f(x .* S .+ M, t) .* S
end

dim = 3
dt = 0.01
Nsteps = 10000000
obs_nn = evolve([1.0, 1.5, 1.8], dt, Nsteps, F, sigma; resolution = 10)

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

plotly()
Plots.scatter(obs_uncorr[1,1:10000], obs_uncorr[2,1:10000], obs_uncorr[3,1:10000], markersize=1, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

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

trj_clustered = evolve([0.0, 0.0, 0.0], 0.5*dt, 10000000, score_clustered_xt, sigma_I; timestepper=:rk4, resolution=2, boundary=[-100,100])
# trj_score = evolve([0.0, 0.0], dt, 1000000, score_true, sigma_I; timestepper=:rk4, resolution=10, boundary=[-100,100])

kde_clustered_x = kde(trj_clustered[1,:])
kde_true_x = kde(obs[1,:])

kde_clustered_y = kde(trj_clustered[2,:])
kde_true_y = kde(obs[2,:])

kde_clustered_z = kde(trj_clustered[3,:])
kde_true_z = kde(obs[3,:])

gr()
plt1 = Plots.plot(kde_clustered_x.x, kde_clustered_x.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
plt1 = Plots.plot!(kde_true_x.x, kde_true_x.density, label="True", xlabel="X", ylabel="Density", title="True PDF")

plt2 = Plots.plot(kde_clustered_y.x, kde_clustered_y.density, label="Observed", xlabel="Y", ylabel="Density", title="Observed PDF")
plt2 = Plots.plot!(kde_true_y.x, kde_true_y.density, label="True", xlabel="Y", ylabel="Density", title="True PDF")

plt3 = Plots.plot(kde_clustered_z.x, kde_clustered_z.density, label="Observed", xlabel="Z", ylabel="Density", title="Observed PDF")
plt3 = Plots.plot!(kde_true_z.x, kde_true_z.density, label="True", xlabel="Z", ylabel="Density", title="True PDF")

Plots.plot(plt1, plt2, plt3, layout=(3, 1), size=(600, 600))

##
kde_true_xy = kde((obs[1,:], obs[2,:]))
kde_clustered_xy = kde((trj_clustered[1,:], trj_clustered[2,:]))

kde_true_xz = kde((obs[1,:], obs[3,:]))
kde_clustered_xz = kde((trj_clustered[1,:], trj_clustered[3,:]))

kde_true_yx = kde((obs[2,:], obs[1,:]))  # Same as xy but with axes swapped
kde_clustered_yx = kde((trj_clustered[2,:], trj_clustered[1,:]))

kde_true_yz = kde((obs[2,:], obs[3,:]))
kde_clustered_yz = kde((trj_clustered[2,:], trj_clustered[3,:]))

kde_true_zx = kde((obs[3,:], obs[1,:]))
kde_clustered_zx = kde((trj_clustered[3,:], trj_clustered[1,:]))

kde_true_zy = kde((obs[3,:], obs[2,:]))
kde_clustered_zy = kde((trj_clustered[3,:], trj_clustered[2,:]))

plt1 = Plots.heatmap(kde_true_xy.x, kde_true_xy.y, kde_true_xy.density, xlabel="X", ylabel="Y", title="True PDF")
plt2 = Plots.heatmap(kde_clustered_xy.x, kde_clustered_xy.y, kde_clustered_xy.density, xlabel="X", ylabel="Y", title="Sampled PDF XY", xrange=(kde_true_xy.x[1], kde_true_xy.x[end]), yrange=(kde_true_xy.y[1], kde_true_xy.y[end]), color=:viridis, clims=(minimum(kde_true_xy.density), maximum(kde_true_xy.density)))
plt3 = Plots.heatmap(kde_true_xz.x, kde_true_xz.y, kde_true_xz.density, xlabel="X", ylabel="Z", title="True PDF")
plt4 = Plots.heatmap(kde_clustered_xz.x, kde_clustered_xz.y, kde_clustered_xz.density, xlabel="X", ylabel="Z", title="Sampled PDF XZ", xrange=(kde_true_xz.x[1], kde_true_xz.x[end]), yrange=(kde_true_xz.y[1], kde_true_xz.y[end]), color=:viridis, clims=(minimum(kde_true_xz.density), maximum(kde_true_xz.density)))
plt5 = Plots.heatmap(kde_true_yz.x, kde_true_yz.y, kde_true_yz.density, xlabel="Y", ylabel="Z", title="True PDF")
plt6 = Plots.heatmap(kde_clustered_yz.x, kde_clustered_yz.y, kde_clustered_yz.density, xlabel="Y", ylabel="Z", title="Sampled PDF YZ", xrange=(kde_true_yz.x[1], kde_true_yz.x[end]), yrange=(kde_true_yz.y[1], kde_true_yz.y[end]), color=:viridis, clims=(minimum(kde_true_yz.density), maximum(kde_true_yz.density)))

Plots.plot(plt1, plt2, plt3, plt4, plt5, plt6, layout=(3, 2), size=(600, 600))

##

# Function to save all the data needed for the Lorenz63 plot
function save_lorenz63_data(filename="data/GMM_data/lorenz63.h5")
    # Create directory structure if it doesn't exist
    mkpath(dirname(filename))
    
    h5open(filename, "w") do file
        # Write scalar values and parameters
        write(file, "dt", dt)
        write(file, "dim", dim)
        write(file, "Nsteps", Nsteps)
        write(file, "Nc", Nc)
        write(file, "pdf_min", pdf_min)
        write(file, "pdf_max", pdf_max)
        
        # Write model parameters
        write(file, "M", M)
        write(file, "S", S)
        
        # Write averages and centers
        write(file, "averages", averages)
        write(file, "centers", centers)
        write(file, "averages_true", averages_true)
        write(file, "averages_gen", averages_gen)
        
        # Write univariate KDE data
        write(file, "kde_true_x_x", collect(kde_true_x.x))
        write(file, "kde_true_x_density", collect(kde_true_x.density))
        write(file, "kde_clustered_x_x", collect(kde_clustered_x.x))
        write(file, "kde_clustered_x_density", collect(kde_clustered_x.density))
        
        write(file, "kde_true_y_x", collect(kde_true_y.x))
        write(file, "kde_true_y_density", collect(kde_true_y.density))
        write(file, "kde_clustered_y_x", collect(kde_clustered_y.x))
        write(file, "kde_clustered_y_density", collect(kde_clustered_y.density))
        
        write(file, "kde_true_z_x", collect(kde_true_z.x))
        write(file, "kde_true_z_density", collect(kde_true_z.density))
        write(file, "kde_clustered_z_x", collect(kde_clustered_z.x))
        write(file, "kde_clustered_z_density", collect(kde_clustered_z.density))
        
        # Write bivariate KDE data
        write(file, "kde_true_xy_x", collect(kde_true_xy.x))
        write(file, "kde_true_xy_y", collect(kde_true_xy.y))
        write(file, "kde_true_xy_density", kde_true_xy.density)
        
        write(file, "kde_clustered_xy_x", collect(kde_clustered_xy.x))
        write(file, "kde_clustered_xy_y", collect(kde_clustered_xy.y))
        write(file, "kde_clustered_xy_density", kde_clustered_xy.density)
        
        write(file, "kde_true_xz_x", collect(kde_true_xz.x))
        write(file, "kde_true_xz_y", collect(kde_true_xz.y))
        write(file, "kde_true_xz_density", kde_true_xz.density)
        
        write(file, "kde_clustered_xz_x", collect(kde_clustered_xz.x))
        write(file, "kde_clustered_xz_y", collect(kde_clustered_xz.y))
        write(file, "kde_clustered_xz_density", kde_clustered_xz.density)
        
        write(file, "kde_true_yz_x", collect(kde_true_yz.x))
        write(file, "kde_true_yz_y", collect(kde_true_yz.y))
        write(file, "kde_true_yz_density", kde_true_yz.density)
        
        write(file, "kde_clustered_yz_x", collect(kde_clustered_yz.x))
        write(file, "kde_clustered_yz_y", collect(kde_clustered_yz.y))
        write(file, "kde_clustered_yz_density", kde_clustered_yz.density)
        
        # Write plot limits
        write(file, "x_limits", collect(x_limits))
        write(file, "y_limits", collect(y_limits))
        write(file, "z_limits", collect(z_limits))
        
        # Write small samples of trajectory data
        write(file, "obs_sample", obs[:, 1:min(10000, size(obs, 2))])
        write(file, "trj_clustered_sample", trj_clustered[:, 1:min(10000, size(trj_clustered, 2))])
    end
    
    println("Data saved to $filename")
end

# Function to read the Lorenz63 data
function read_lorenz63_data(filename="data/GMM_data/lorenz63.h5")
    data = Dict()
    
    h5open(filename, "r") do file
        # Read scalar values
        data["dt"] = read(file, "dt")
        data["dim"] = read(file, "dim")
        data["Nsteps"] = read(file, "Nsteps")
        data["Nc"] = read(file, "Nc")
        data["pdf_min"] = read(file, "pdf_min")
        data["pdf_max"] = read(file, "pdf_max")
        
        # Read model parameters
        data["M"] = read(file, "M")
        data["S"] = read(file, "S")
        
        # Read averages and centers
        data["averages"] = read(file, "averages")
        data["centers"] = read(file, "centers")
        data["averages_true"] = read(file, "averages_true")
        data["averages_gen"] = read(file, "averages_gen")
        
        # Read univariate KDE data
        data["kde_true_x_x"] = read(file, "kde_true_x_x")
        data["kde_true_x_density"] = read(file, "kde_true_x_density")
        data["kde_clustered_x_x"] = read(file, "kde_clustered_x_x")
        data["kde_clustered_x_density"] = read(file, "kde_clustered_x_density")
        
        data["kde_true_y_x"] = read(file, "kde_true_y_x")
        data["kde_true_y_density"] = read(file, "kde_true_y_density")
        data["kde_clustered_y_x"] = read(file, "kde_clustered_y_x")
        data["kde_clustered_y_density"] = read(file, "kde_clustered_y_density")
        
        data["kde_true_z_x"] = read(file, "kde_true_z_x")
        data["kde_true_z_density"] = read(file, "kde_true_z_density")
        data["kde_clustered_z_x"] = read(file, "kde_clustered_z_x")
        data["kde_clustered_z_density"] = read(file, "kde_clustered_z_density")
        
        # Read bivariate KDE data
        data["kde_true_xy_x"] = read(file, "kde_true_xy_x")
        data["kde_true_xy_y"] = read(file, "kde_true_xy_y")
        data["kde_true_xy_density"] = read(file, "kde_true_xy_density")
        
        data["kde_clustered_xy_x"] = read(file, "kde_clustered_xy_x")
        data["kde_clustered_xy_y"] = read(file, "kde_clustered_xy_y")
        data["kde_clustered_xy_density"] = read(file, "kde_clustered_xy_density")
        
        data["kde_true_xz_x"] = read(file, "kde_true_xz_x")
        data["kde_true_xz_y"] = read(file, "kde_true_xz_y")
        data["kde_true_xz_density"] = read(file, "kde_true_xz_density")
        
        data["kde_clustered_xz_x"] = read(file, "kde_clustered_xz_x")
        data["kde_clustered_xz_y"] = read(file, "kde_clustered_xz_y")
        data["kde_clustered_xz_density"] = read(file, "kde_clustered_xz_density")
        
        data["kde_true_yz_x"] = read(file, "kde_true_yz_x")
        data["kde_true_yz_y"] = read(file, "kde_true_yz_y")
        data["kde_true_yz_density"] = read(file, "kde_true_yz_density")
        
        data["kde_clustered_yz_x"] = read(file, "kde_clustered_yz_x")
        data["kde_clustered_yz_y"] = read(file, "kde_clustered_yz_y")
        data["kde_clustered_yz_density"] = read(file, "kde_clustered_yz_density")
        
        # Read plot limits
        data["x_limits"] = Tuple(read(file, "x_limits"))
        data["y_limits"] = Tuple(read(file, "y_limits"))
        data["z_limits"] = Tuple(read(file, "z_limits"))
        
        # Read sample data
        data["obs_sample"] = read(file, "obs_sample")
        data["trj_clustered_sample"] = read(file, "trj_clustered_sample")
    end
    
    println("Data loaded from $filename")
    return data
end

# Run the save function
save_lorenz63_data()

##
# Example of loading and using the data
data = read_lorenz63_data()

# Extract the necessary variables for plotting
dt = data["dt"]
dim = data["dim"]
pdf_min = data["pdf_min"] 
pdf_max = data["pdf_max"]

# Extract KDE data
kde_true_x_x = data["kde_true_x_x"]
kde_true_x_density = data["kde_true_x_density"]
kde_clustered_x_x = data["kde_clustered_x_x"]
kde_clustered_x_density = data["kde_clustered_x_density"]

kde_true_y_x = data["kde_true_y_x"]
kde_true_y_density = data["kde_true_y_density"]
kde_clustered_y_x = data["kde_clustered_y_x"]
kde_clustered_y_density = data["kde_clustered_y_density"]

kde_true_z_x = data["kde_true_z_x"]
kde_true_z_density = data["kde_true_z_density"]
kde_clustered_z_x = data["kde_clustered_z_x"]
kde_clustered_z_density = data["kde_clustered_z_density"]

kde_true_xy_x = data["kde_true_xy_x"]
kde_true_xy_y = data["kde_true_xy_y"]
kde_true_xy_density = data["kde_true_xy_density"]
kde_clustered_xy_x = data["kde_clustered_xy_x"]
kde_clustered_xy_y = data["kde_clustered_xy_y"]
kde_clustered_xy_density = data["kde_clustered_xy_density"]

kde_true_xz_x = data["kde_true_xz_x"]
kde_true_xz_y = data["kde_true_xz_y"]
kde_true_xz_density = data["kde_true_xz_density"]
kde_clustered_xz_x = data["kde_clustered_xz_x"]
kde_clustered_xz_y = data["kde_clustered_xz_y"]
kde_clustered_xz_density = data["kde_clustered_xz_density"]

kde_true_yz_x = data["kde_true_yz_x"]
kde_true_yz_y = data["kde_true_yz_y"]
kde_true_yz_density = data["kde_true_yz_density"]
kde_clustered_yz_x = data["kde_clustered_yz_x"]
kde_clustered_yz_y = data["kde_clustered_yz_y"]
kde_clustered_yz_density = data["kde_clustered_yz_density"]

# Extract plot limits
x_limits = data["x_limits"]
y_limits = data["y_limits"]
z_limits = data["z_limits"]

println("All data loaded and ready for plotting")

##

using GLMakie
using ColorSchemes

# Create main figure with 3x3 layout
fig = GLMakie.Figure(size=(2000, 1500), fontsize=22)

# Create layout for 3x3 grid
grid = fig[1:3, 1:3] = GLMakie.GridLayout(3, 3)

# First column: Univariate PDFs
ax_pdf_x = GLMakie.Axis(grid[1, 1],
    xlabel="x", ylabel="PDF",
    title="Univariate x PDF",
    titlesize=32, xlabelsize=28, ylabelsize=28)

ax_pdf_y = GLMakie.Axis(grid[2, 1],
    xlabel="y", ylabel="PDF", 
    title="Univariate y PDF",
    titlesize=32, xlabelsize=28, ylabelsize=28)

ax_pdf_z = GLMakie.Axis(grid[3, 1],
    xlabel="z", ylabel="PDF", 
    title="Univariate z PDF",
    titlesize=32, xlabelsize=28, ylabelsize=28)

# Second column: True Bivariate PDFs
ax_true_xy = GLMakie.Axis(grid[1, 2],
    xlabel="x", ylabel="y",
    title="True (x,y) PDF",
    aspect=GLMakie.DataAspect(),
    titlesize=32, xlabelsize=28, ylabelsize=28)

ax_true_xz = GLMakie.Axis(grid[2, 2],
    xlabel="x", ylabel="z",
    title="True (x,z) PDF",
    aspect=GLMakie.DataAspect(),
    titlesize=32, xlabelsize=28, ylabelsize=28)

ax_true_yz = GLMakie.Axis(grid[3, 2],
    xlabel="y", ylabel="z",
    title="True (y,z) PDF",
    aspect=GLMakie.DataAspect(),
    titlesize=32, xlabelsize=28, ylabelsize=28)

# Third column: GMM Bivariate PDFs
ax_gmm_xy = GLMakie.Axis(grid[1, 3],
    xlabel="x", ylabel="y",
    title="GMM (x,y) PDF",
    aspect=GLMakie.DataAspect(),
    titlesize=32, xlabelsize=28, ylabelsize=28)

ax_gmm_xz = GLMakie.Axis(grid[2, 3],
    xlabel="x", ylabel="z",
    title="GMM (x,z) PDF",
    aspect=GLMakie.DataAspect(),
    titlesize=32, xlabelsize=28, ylabelsize=28)

ax_gmm_yz = GLMakie.Axis(grid[3, 3],
    xlabel="y", ylabel="z",
    title="GMM (y,z) PDF",
    aspect=GLMakie.DataAspect(),
    titlesize=32, xlabelsize=28, ylabelsize=28)

# Univariate PDFs - First column
# Plot both true and GMM in the same axes with legends using read data
GLMakie.lines!(ax_pdf_x, kde_true_x_x, kde_true_x_density, 
       color=:red, linewidth=2, label="True")
GLMakie.lines!(ax_pdf_x, kde_clustered_x_x, kde_clustered_x_density, 
       color=:blue, linewidth=2, label="GMM")
GLMakie.axislegend(ax_pdf_x, position=:lt, labelsize=20)

GLMakie.lines!(ax_pdf_y, kde_true_y_x, kde_true_y_density, 
       color=:red, linewidth=2, label="True")
GLMakie.lines!(ax_pdf_y, kde_clustered_y_x, kde_clustered_y_density, 
       color=:blue, linewidth=2, label="GMM")

GLMakie.lines!(ax_pdf_z, kde_true_z_x, kde_true_z_density, 
       color=:red, linewidth=2, label="True")
GLMakie.lines!(ax_pdf_z, kde_clustered_z_x, kde_clustered_z_density, 
       color=:blue, linewidth=2, label="GMM")

# Use pre-computed min/max values for color scaling
# If not available in data, compute them from the density arrays
if !haskey(data, "pdf_min") || !haskey(data, "pdf_max")
    pdf_min = min(
        minimum(kde_true_xy_density),
        minimum(kde_clustered_xy_density),
        minimum(kde_true_xz_density),
        minimum(kde_clustered_xz_density),
        minimum(kde_true_yz_density),
        minimum(kde_clustered_yz_density)
    )
    pdf_max = max(
        maximum(kde_true_xy_density),
        maximum(kde_clustered_xy_density),
        maximum(kde_true_xz_density),
        maximum(kde_clustered_xz_density),
        maximum(kde_true_yz_density),
        maximum(kde_clustered_yz_density)
    )
else
    pdf_min = data["pdf_min"] 
    pdf_max = data["pdf_max"]
end

# True bivariate PDFs (second column) using heatmaps with read data
hm1 = GLMakie.heatmap!(ax_true_xy, 
        kde_true_xy_x, kde_true_xy_y, kde_true_xy_density,
        colormap=:viridis, 
        colorrange=(pdf_min, pdf_max))

hm2 = GLMakie.heatmap!(ax_true_xz, 
        kde_true_xz_x, kde_true_xz_y, kde_true_xz_density,
        colormap=:viridis, 
        colorrange=(pdf_min, pdf_max))

hm3 = GLMakie.heatmap!(ax_true_yz, 
        kde_true_yz_x, kde_true_yz_y, kde_true_yz_density,
        colormap=:viridis, 
        colorrange=(pdf_min, pdf_max))

# GMM bivariate PDFs (third column) using heatmaps with read data
hm4 = GLMakie.heatmap!(ax_gmm_xy, 
        kde_clustered_xy_x, kde_clustered_xy_y, kde_clustered_xy_density,
        colormap=:viridis, 
        colorrange=(pdf_min, pdf_max))

hm5 = GLMakie.heatmap!(ax_gmm_xz, 
        kde_clustered_xz_x, kde_clustered_xz_y, kde_clustered_xz_density,
        colormap=:viridis, 
        colorrange=(pdf_min, pdf_max))

hm6 = GLMakie.heatmap!(ax_gmm_yz, 
        kde_clustered_yz_x, kde_clustered_yz_y, kde_clustered_yz_density,
        colormap=:viridis, 
        colorrange=(pdf_min, pdf_max))

# Use plot limits from data file if available, otherwise compute them
if haskey(data, "x_limits") && haskey(data, "y_limits") && haskey(data, "z_limits")
    x_limits = data["x_limits"]
    y_limits = data["y_limits"]
    z_limits = data["z_limits"]
else
    # Compute them from the data
    x_limits = (min(minimum(kde_true_xy_x), minimum(kde_clustered_xy_x)), 
              max(maximum(kde_true_xy_x), maximum(kde_clustered_xy_x)))
    y_limits = (min(minimum(kde_true_xy_y), minimum(kde_clustered_xy_y)),
              max(maximum(kde_true_xy_y), maximum(kde_clustered_xy_y)))
    z_limits = (min(minimum(kde_true_xz_y), minimum(kde_clustered_xz_y)),
              max(maximum(kde_true_xz_y), maximum(kde_clustered_xz_y)))
end

# Apply limits to true and GMM plots
GLMakie.xlims!(ax_true_xy, x_limits)
GLMakie.ylims!(ax_true_xy, y_limits)
GLMakie.xlims!(ax_gmm_xy, x_limits)
GLMakie.ylims!(ax_gmm_xy, y_limits)

GLMakie.xlims!(ax_true_xz, x_limits)
GLMakie.ylims!(ax_true_xz, z_limits)
GLMakie.xlims!(ax_gmm_xz, x_limits)
GLMakie.ylims!(ax_gmm_xz, z_limits)

GLMakie.xlims!(ax_true_yz, y_limits)
GLMakie.ylims!(ax_true_yz, z_limits)
GLMakie.xlims!(ax_gmm_yz, y_limits)
GLMakie.ylims!(ax_gmm_yz, z_limits)

# Add a shared colorbar for all bivariate plots
pdf_bar = GLMakie.Colorbar(fig[1:3, 4], 
              colormap=:viridis, 
              limits=(pdf_min, pdf_max),
              label="Probability Density",
              labelsize=28,
              vertical=true,
              width=30)

# Adjust spacing
GLMakie.colgap!(grid, 5)
GLMakie.rowgap!(grid, 5)

# Add a title
GLMakie.Label(fig[0, 1:3], text="Lorenz63 Density Comparison", fontsize=36, font=:bold)

# Make sure the figures directory exists
mkpath("figures/GMM_figures")

# Save the figure
GLMakie.save("figures/GMM_figures/lorenz63.png", fig)

fig