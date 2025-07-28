using Pkg
Pkg.activate(".")
Pkg.instantiate()

##
using Statistics
using LinearAlgebra
using HDF5
using Plots
using KernelDensity
using LaTeXStrings
using StatsPlots
using ClustGen
using Flux
using BSON
using ProgressBars
using StatsBase
using Revise
using MarkovChainHammer
using Measures


# === Load all PCA reconstruction variables from file ===
data_dir = joinpath(@__DIR__, "PlaSim_data")
pc_scores_file = joinpath(data_dir, "pc_scores.h5")
println("Loading PCA variables from $pc_scores_file ...")
h5f = h5open(pc_scores_file, "r")
pc_scores = read(h5f["pc_scores"])
year_mean = read(h5f["year_mean"])
data_mean = read(h5f["data_mean"])
eigenvecs = read(h5f["eigenvecs"])
times = read(h5f["times"])
close(h5f)
println("Loaded pc_scores with shape ", size(pc_scores))
println("Loaded year_mean with shape ", size(year_mean))
println("Loaded data_mean with shape ", size(data_mean))
println("Loaded eigenvecs with shape ", size(eigenvecs))
println("Loaded times with shape ", size(times))
##

times_sin = sin.(2π * times / 360)
times_cos = cos.(2π * times / 360)
times_mod = vcat(reshape(times_sin, 1, :), reshape(times_cos, 1, :))
obs_nn = vcat(times_mod, pc_scores[1:20,:])

M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S

# M = mean(obs_nn, dims=2)
# S = std(obs_nn, dims=2)
# obs = zeros(size(obs_nn))
# obs[1,:] = (obs_nn[1,:] .- M[1]) ./ S[1]
# obs[2:end,:] = (obs_nn[2:end,:] .- M[2]) ./ S[2]

dim = size(obs, 1)

plotly()
obs_uncorr = obs[:, 1:1:end]

Plots.scatter(obs_uncorr[1,1:10000], obs_uncorr[3,1:10000], markersize=1, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
#################### TRAINING WITH VANILLA LOSS ####################
σ_value = 0.05
time_dim = 2
@time nn_vanilla, loss_vanilla = train(obs_uncorr, 40, 16, [dim, 128, 64, dim-time_dim], σ_value, time_dim; use_gpu=true, opt=Adam(0.0002))
nn_vanilla_cpu = nn_vanilla |> cpu
score_vanilla(x) = .- nn_vanilla_cpu(Float32.([x...])) ./ σ_value
Plots.plot(loss_vanilla)

##
###############################. CONSTRUCTION OF Φ AND Σ MATRICES ###############################
# Get dimensions from the data
obs_x = obs[3:end, :]
dim_x, num_timesteps_x = size(obs_x)


# Calculate state differences between consecutive time steps
dx = obs_x[:, 2:end] - obs_x[:, 1:end-1]

# Use states corresponding to the start of the time step
x_t = obs_x[:, 1:end-1]

# Estimate M using the expectation over the time series
# We average over (num_timesteps - 1) intervals
M_mat_x = (dx * x_t') / (num_timesteps_x - 1)

# --- 2. Calculate the V^T matrix ---
# V captures the covariance between the state and the score function.
# The paper's final equation uses V^T, which is E[s(x) * x'][cite: 170, 172].

# Assume obs is your (dim x num_timesteps) data matrix
# Assume score_vanilla(x) is your trained score function

score_arr = zeros(dim_x, num_timesteps_x) # Pre-allocate the matrix

for t in 1:num_timesteps_x
    # Get the state at time t
    current_state = obs[:, t]
    
    # Calculate the score for that state and store it in the corresponding column
    score_arr[:, t] = score_vanilla(current_state)
end

# Now 'score' is ready to be used in the V_T calculation
V_T_x = (score_arr * obs_x') / num_timesteps_x

# --- 3. Calculate 360 periodic Drift and Diffusion Matrices ---
# For each time point in the 360-unit period, calculate Φ and Σ using only nearby data points

# Initialize arrays to store 360 matrices
Φ_periodic = zeros(360, dim_x, dim_x)
Σ_periodic = zeros(360, dim_x, dim_x)

# Time window for selecting nearby points (±window around each target time)
time_window = 1  # Use points within ±1 time unit

println("Calculating periodic matrices for 360 time points...")

# Pre-compute periodic times to avoid repeated calculations
periodic_times = mod.(times, 360)  # Map all times to [0, 360) range

# Optimized function to find close indices for a given target time
function find_close_indices(target_time, periodic_times, time_window)
    close_indices = Int[]
    for i in eachindex(periodic_times)
        time_diff = min(abs(periodic_times[i] - target_time), 
                       360 - abs(periodic_times[i] - target_time))
        if time_diff <= time_window
            push!(close_indices, i)
        end
    end
    
    # If not enough points, expand the window
    if length(close_indices) < 10
        close_indices = Int[]
        for i in eachindex(periodic_times)
            time_diff = min(abs(periodic_times[i] - target_time), 
                           360 - abs(periodic_times[i] - target_time))
            if time_diff <= 2 * time_window
                push!(close_indices, i)
            end
        end
    end
    
    return close_indices
end

# Function to calculate matrices for a single target time
function calculate_matrices_for_time(target_time)
    # Find close indices
    close_indices = find_close_indices(target_time, periodic_times, time_window)
    
    # Extract subset of data for this time
    obs_x_subset = obs_x[:, close_indices]
    obs_subset = obs[:, close_indices]
    
    # Calculate state differences for this subset
    dx_subset = obs_x_subset[:, 2:end] - obs_x_subset[:, 1:end-1]
    x_t_subset = obs_x_subset[:, 1:end-1]
    
    # Calculate M matrix for this time
    M_mat_subset = (dx_subset * x_t_subset') / (size(x_t_subset, 2))
    
    # Calculate score array for this subset
    score_arr_subset = zeros(dim_x, size(obs_subset, 2))
    for j in 1:size(obs_subset, 2)
        current_state = obs_subset[:, j]
        score_arr_subset[:, j] = score_vanilla(current_state)
    end
    
    # Calculate V_T matrix for this subset
    V_T_subset = (score_arr_subset * obs_x_subset') / size(obs_x_subset, 2)
    
    # Calculate Φ for this time
    Φ_t = M_mat_subset * pinv(V_T_subset)
    
    # Initialize Σ_t to ensure it's always defined
    Σ_t = sqrt(1e-3) * I(dim_x)  # Default fallback value
    
    # Ensure the symmetric part of Φ_t is positive definite by removing negative eigenvalues
    try
        Φ_symmetric = 0.5 * (Φ_t .+ Φ_t')  # Extract symmetric part
        
        # Force exact symmetry to avoid numerical issues
        Φ_symmetric = Hermitian(Φ_symmetric)
        
        eigenvals, eigenvecs = eigen(Φ_symmetric)
        
        # Keep only positive eigenvalues and their corresponding eigenvectors
        positive_indices = eigenvals .> 1e-5  # Use a reasonable threshold
        if sum(positive_indices) > 0
            positive_eigenvals = eigenvals[positive_indices]
            positive_eigenvecs = eigenvecs[:, positive_indices]
            
            # Reconstruct the symmetric part using only positive eigenvalues
            Φ_symmetric_reconstructed = positive_eigenvecs * Diagonal(positive_eigenvals) * positive_eigenvecs'
            
            # Force exact Hermitian symmetry
            Φ_symmetric_reconstructed = Hermitian(0.5 * (Φ_symmetric_reconstructed .+ Φ_symmetric_reconstructed'))
        else
            # If no positive eigenvalues, use a small identity matrix
            Φ_symmetric_reconstructed = Hermitian(1e-3 * I(dim_x))
        end
        
        # Add regularization for numerical stability
        Φ_symmetric_final = Φ_symmetric_reconstructed + 1e-5 * I(dim_x)
        
        # Calculate Σ for this time using the reconstructed positive-definite symmetric part
        Σ_t = cholesky(Φ_symmetric_final).L
        
    catch e
        println("Warning: Matrix calculation failed for target_time $target_time, using fallback identity")
        Σ_t = sqrt(1e-3) * I(dim_x)
    end
    
    return Φ_t, Σ_t
end

# Parallelize the computation using Threads
using Base.Threads

# Pre-allocate thread-safe arrays
Φ_results = Vector{Matrix{Float64}}(undef, 360)
Σ_results = Vector{Matrix{Float64}}(undef, 360)

# Parallel computation
@threads for target_time in 1:360
    Φ_t, Σ_t = calculate_matrices_for_time(target_time)
    Φ_results[target_time] = Φ_t
    Σ_results[target_time] = Σ_t
    
    # Thread-safe progress reporting
    if target_time % 60 == 0
        println("Completed time $target_time/360 on thread $(threadid())")
    end
end

# Copy results to the 3D arrays
for target_time in 1:360
    Φ_periodic[target_time, :, :] = Φ_results[target_time]
    Σ_periodic[target_time, :, :] = Σ_results[target_time]
end

# Create functions to access the periodic matrices
function Φ_at_time(t)
    time_index = Int(mod(round(t), 360)) + 1  # Convert to 1-based indexing
    return Φ_periodic[time_index, :, :]
end

function Σ_at_time(t)
    time_index = Int(mod(round(t), 360)) + 1  # Convert to 1-based indexing
    return Σ_periodic[time_index, :, :]
end
##
plotly()
min_eigs = zeros(360)
for i in 1:360
    min_eigs[i] = minimum(eigvals(Φ_periodic[i,:,:] + Φ_periodic[i,:,:]'))
end
plot(1:360, min_eigs, label="Minimum Eigenvalue", xlabel="Time (mod 360)", ylabel="Minimum Eigenvalue", title="Minimum Eigenvalues of Φ Periodic Matrices", legend=:bottomright)
##
idx = 140
plot(eigvals(Φ_periodic[idx,:,:] + Φ_periodic[idx,:,:]'))

##
#################### SAMPLES GENERATION ####################

# The dimension of the physical (observed) system
dim_obs = dim - 2  # Now we have 2 time dimensions (sin and cos)

# --- Corrected Drift and Noise Functions with Time-dependent Matrices ---

# 2. Define a helper function to compute the full score vector.
#    This is for clarity and avoids repetition.
#    (Assumes M and S are pre-calculated mean/std for normalization)
function full_score(x_obs, t)
    # Augment the state with the normalized time variables (sin and cos)
    time_sin = (sin(2π * t / 360) - M[1]) / S[1]
    time_cos = (cos(2π * t / 360) - M[2]) / S[2]
    augmented_state = vcat(time_sin, time_cos, x_obs)
    
    # Return the full score vector (size: dim x 1)
    return score_vanilla(augmented_state)
end

# 3. Define the corrected drift function using time-dependent Φ matrices
drift_x_corrected(x_obs, t) = Φ_at_time(t) * full_score(x_obs, t)

# 4. Define the corrected noise matrix using time-dependent Σ matrices
sigma_x_corrected(x_obs, t) = Σ_at_time(t)


# --- Integration Step ---

Nsteps = 10000
dt = 0.01

# The initial condition vector should have size `dim_obs`
initial_condition = zeros(dim_obs)

# Evolve the system with the corrected functions.
# Note: Your SDE solver (`evolve`) must now use a noise vector `dW` that has `dim`
# components, not `dim_obs`, because sigma_x_corrected is a (dim_obs x dim) matrix.
trj_clustered = evolve(
    initial_condition,
    dt,
    100 * Nsteps,
    drift_x_corrected,
    sigma_x_corrected;
    timestepper=:euler,
    resolution=100,
    boundary=[-5, 5]
)

res = 1
tsteps = 26
auto_clustered = zeros(dim_obs, tsteps)
auto_obs = zeros(dim_obs, tsteps)

for i in 1:dim_obs
    auto_clustered[i,:] = autocovariance(trj_clustered[i,:]; timesteps=tsteps) 
    auto_obs[i,:] = autocovariance(obs[i+2,:]; timesteps=tsteps)  # Skip first 2 time dimensions
end

## Plotting
gr()

t_mod_sin = obs[1,1:Nsteps+1]
t_mod_cos = obs[2,1:Nsteps+1]
obs_clustered = vcat(reshape(t_mod_sin, 1, :), reshape(t_mod_cos, 1, :), trj_clustered)

# --- Univariate PDFs (Figure 1) ---
dim_obs = size(obs, 1) - 2  # Number of physical variables
colors = [:blue, :red]
labels = ["Generated", "Observed"]

#(Removed intermediate PDF plots, only final figure will be displayed)


# === Two separate figures, each with 10 rows, legend at the bottom (not repeated in each panel) ===

default(fontfamily="Computer Modern", lw=2, framestyle=:box, legendfontsize=10, guidefontsize=14, tickfontsize=12, titlefontsize=16)

# Helper to create a single row of 5 plots for variable i, with legend only in the bottom row



# Helper to create a single row of 5 plots for variable i, with no legend
function make_panel_row(i; add_legend_handles=false)
    # Trajectory
    p1 = plot(1:1000, trj_clustered[i,1:1000], label="Generated", color=:blue, lw=2,
        xlabel=i==dim_obs ? "Time step" : "", ylabel="x_$i", legend=false, grid=false, framestyle=:box, margin=5Plots.mm,
        xguidefont=font(13), yguidefont=font(13), xtickfont=font(11), ytickfont=font(11), title=i==1 ? "Trajectory" : "")
    if size(obs, 1) >= i+2
        plot!(p1, 1:1000, obs[i+2,1:1000], label="Observed", color=:red, lw=2)
    end
    # Autocovariance
    p2 = plot(0:tsteps-1, auto_clustered[i, :], label="Generated", color=:blue, lw=2,
        xlabel=i==dim_obs ? "Lag" : "", ylabel="Autocov(x_$i)", legend=false, grid=false, framestyle=:box, margin=5Plots.mm,
        xguidefont=font(13), yguidefont=font(13), xtickfont=font(11), ytickfont=font(11), title=i==1 ? "Autocovariance" : "")
    plot!(p2, 0:tsteps-1, auto_obs[i, :], label="Observed", color=:red, lw=2)
    # Univariate PDF
    kde_gen = kde(obs_clustered[i+2, :])
    kde_obs = kde(obs[i+2, :])
    p3 = plot(kde_gen.x, kde_gen.density, color=:blue, label="Generated", lw=2, grid=false, legend=false)
    plot!(p3, kde_obs.x, kde_obs.density, color=:red, label="Observed", lw=2, grid=false)
    ylabel!(p3, "pdf")
    xlabel!(p3, "x_$i")
    if i == 1
        title!(p3, "Univariate PDF")
    end
    # Bivariate Gen
    kde_gen2 = kde((obs_clustered[1, :], obs_clustered[i+2, :]))
    p4 = heatmap(kde_gen2.x, kde_gen2.y, kde_gen2.density', color=:blues, xlabel="t_sin", ylabel="x_$i", colorbar=false, legend=false)
    if i == 1
        title!(p4, "Bivariate Gen")
    end
    # Bivariate Obs
    kde_obs2 = kde((obs[1, :], obs[i+2, :]))
    p5 = heatmap(kde_obs2.x, kde_obs2.y, kde_obs2.density', color=:reds, xlabel="t_sin", ylabel="x_$i", colorbar=false, legend=false)
    if i == 1
        title!(p5, "Bivariate Obs")
    end
    return [p1, p2, p3, p4, p5]
end



# Split into two figures, 10 rows each, legend outside below the panels
rows_per_fig = 10
num_figs = ceil(Int, dim_obs / rows_per_fig)
for fig_idx in 1:num_figs
    start_row = (fig_idx-1)*rows_per_fig + 1
    end_row = min(fig_idx*rows_per_fig, dim_obs)
    panel_rows = []
    for (j, i) in enumerate(start_row:end_row)
        # Only the very first subplot gets the invisible legend handles
        add_legend_handles = (j == 1)
        append!(panel_rows, make_panel_row(i; add_legend_handles=add_legend_handles))
    end
    nrows = end_row - start_row + 1

    # Set titles for the first row only
    plot_titles = ["Trajectory", "Autocovariance", "Univariate PDF", "Bivariate Gen", "Bivariate Obs"]
    for k in 1:5
        plot!(panel_rows[k], title=plot_titles[k])
    end

    fig = plot(
        panel_rows...,
        layout = grid(nrows, 5),
        legend=false,
        size = (2400, 300*nrows+80),
        left_margin=8Plots.mm, bottom_margin=8Plots.mm, top_margin=8Plots.mm, right_margin=8Plots.mm,
        titlefont=font(20, "Computer Modern"),
        dpi=300,
        plot_title=""
    )
    display(fig)
    # Optionally save each figure
    savefig(fig, "combined_figure_vanilla_part$(fig_idx).pdf")
end

## === Two separate figures, each with 10 rows, legend at the bottom (not repeated in each panel) ===

# === Additional: Bivariate PDFs vs first three variables for all 20 variables ===

# Helper to create a row of 6 bivariate PDF plots for variable i vs variables 1,2,3 (generated and observed)
function make_bivar_panel_row(i)
    panels = []
    for j in 1:3
        # Generated
        kde_gen = kde((obs_clustered[j+2, :], obs_clustered[i+2, :]))
        density_rot_gen = kde_gen.density
        # Observed
        kde_obs = kde((obs[j+2, :], obs[i+2, :]))
        density_rot_obs = kde_obs.density

        # Axis labels: only leftmost and bottom panels get labels
        # These will be set later in the plotting loop
        p_gen = heatmap(
            kde_gen.y, kde_gen.x, density_rot_gen',
            color=:blues, xlabel="", ylabel="",
            colorbar=false, legend=false, grid=false, framestyle=:box, margin=3Plots.mm,
            title="Gen: x_$i vs x_$j"
        )
        p_obs = heatmap(
            kde_obs.y, kde_obs.x, density_rot_obs',
            color=:reds, xlabel="", ylabel="",
            colorbar=false, legend=false, grid=false, framestyle=:box, margin=3Plots.mm,
            title="Obs: x_$i vs x_$j"
        )
        push!(panels, p_gen)
        push!(panels, p_obs)
    end
    return panels
end

# Split into two figures, 10 rows each, 6 columns (x_i vs x_1, x_2, x_3 for i=1:10 and i=11:20)
rows_per_fig_bivar = 10
num_figs_bivar = ceil(Int, dim_obs / rows_per_fig_bivar)
for fig_idx in 1:num_figs_bivar
    start_row = (fig_idx-1)*rows_per_fig_bivar + 1
    end_row = min(fig_idx*rows_per_fig_bivar, dim_obs)
    panel_rows = []
    for i in start_row:end_row
        append!(panel_rows, make_bivar_panel_row(i))
    end
    nrows = end_row - start_row + 1
    ncols = 6
    # Remove all panel titles and the global title, and set axis labels only on leftmost and bottom panels
    for row in 1:nrows
        for col in 1:ncols
            idx = (row-1)*ncols + col
            if idx > length(panel_rows)
                continue
            end
            # Remove title
            plot!(panel_rows[idx], title="")
            # Set y label and yticks only for leftmost panels (col==1)
            if col == 1
                var_idx = row + start_row - 1
                ylabel!(panel_rows[idx], "x_$(var_idx)")
                yformatter = :auto
            else
                ylabel!(panel_rows[idx], "")
                yformatter = (y -> "")
            end
            # Set x label and xticks only for bottom panels (row==nrows)
            if row == nrows
                j_idx = ceil(Int, col/2)
                xlabel!(panel_rows[idx], "x_$(j_idx)")
                xformatter = :auto
            else
                xlabel!(panel_rows[idx], "")
                xformatter = (x -> "")
            end
            # Apply formatters
            plot!(panel_rows[idx], xformatter=xformatter, yformatter=yformatter)
        end
    end
    fig = plot(
        panel_rows...,
        layout = grid(nrows, ncols, hgap=1mm, vgap=1mm),
        legend=false,
        size = (300*ncols, 250*nrows),
        left_margin=15Plots.mm, bottom_margin=4Plots.mm, top_margin=4Plots.mm, right_margin=6Plots.mm,
        titlefont=font(18, "Computer Modern"),
        dpi=300,
        plot_title=""
    )
    display(fig)
    # Optionally save each figure
    savefig(fig, "bivar_vs_x123_part$(fig_idx).pdf")
end

##
## Create GIF comparing original and predicted reconstructed time series

# Number of frames for the GIF
n_frames = 360

# Reconstruct original data (first 1000 snapshots)
# Take first 20 PC scores from original data
original_pc_scores = pc_scores[1:20, 1:n_frames]

# Reconstruct using PCA: data_reconstructed = eigenvecs * pc_scores + data_mean
original_reconstructed = eigenvecs[:, 1:20] * original_pc_scores .+ data_mean

# Handle year_mean properly - it's a 3D array (64×32×360)
# We need to get the corresponding time indices for the seasonal cycle
time_indices_original = clamp.(Int.(round.((sin.(2π * (1:n_frames) / 360) .+ 1) * 180)), 1, 360)  # Clamp to 1-360
year_mean_original = zeros(size(original_reconstructed))
for i in 1:n_frames
    year_mean_original[:, i] = vec(year_mean[:, :, time_indices_original[i]])
end

# Add back the year mean (seasonal cycle)
original_full = original_reconstructed .+ year_mean_original

# Reconstruct predicted data 
# Take first 1000 points from predicted trajectory
predicted_pc_scores = trj_clustered[:, 1:n_frames]

# Rescale predicted PC scores back to original scale
# Remember: obs[3:end,:] = (pc_scores[1:20,:] .- M[3:end]) ./ S[3:end]  # Skip first 2 time dimensions
# So: pc_scores_rescaled = obs[3:end,:] .* S[3:end] .+ M[3:end]
predicted_pc_rescaled = predicted_pc_scores .* S[3:end] .+ M[3:end]

# Reconstruct using PCA
predicted_reconstructed = eigenvecs[:, 1:20] * predicted_pc_rescaled .+ data_mean

# For predicted data, we need to use the time from the simulation
# Get corresponding time indices for predicted data
time_indices_predicted = clamp.(Int.(round.((sin.(2π * (1:n_frames) / 360) .+ 1) * 180)), 1, 360)  # Clamp to 1-360
year_mean_predicted = zeros(size(predicted_reconstructed))
for i in 1:n_frames
    year_mean_predicted[:, i] = vec(year_mean[:, :, time_indices_predicted[i]])
end

# Add back the year mean (seasonal cycle)
predicted_full = predicted_reconstructed .+ year_mean_predicted

##
# Reshape to 64x32 spatial grids and rotate by 90 degrees LEFT (counterclockwise)
function reshape_and_rotate(data_vector)
    grid = reshape(data_vector, 64, 32)  # 64x32
    rotated_180 = grid[end:-1:1, end:-1:1]  # 180 degree rotation
    return permutedims(rotated_180)[:, end:-1:1]  # Additional 90 degrees LEFT
end

# Create the GIF
println("Creating GIF comparison...")
anim = @animate for i in 1:n_frames
    # Original data (left panel)
    p1 = heatmap(
        reshape_and_rotate(original_full[:, i]),
        title="Original (t=$i)",
        c=:viridis,
        aspect_ratio=:equal,
        showaxis=false,
        grid=false,
        colorbar=true
    )
    
    # Predicted data (right panel)
    p2 = heatmap(
        reshape_and_rotate(predicted_full[:, i]),
        title="Generated (t=$i)",
        c=:viridis,
        aspect_ratio=:equal,
        showaxis=false,
        grid=false,
        colorbar=true
    )
    
    # Combine plots
    plot(p1, p2, layout=(1, 2), size=(1500, 800))
end

# Save the GIF
gif(anim, "reconstruction_comparison2.gif", fps=30)
println("GIF saved as 'reconstruction_comparison.gif'")
##
# === Create a figure with 6 snapshots from the gif at times 60,120,180,240,300,360 ===
snapshot_times = [60, 120, 180, 240, 300, 360]
plots_snapshots = []
for t in snapshot_times
    # Original data (left panel)
    p1 = heatmap(
        reshape_and_rotate(original_full[:, t]),
        title="Original (t=$(t))",
        c=:viridis,
        aspect_ratio=:equal,
        showaxis=false,
        grid=false,
        colorbar=false
    )
    # Predicted data (right panel)
    p2 = heatmap(
        reshape_and_rotate(predicted_full[:, t]),
        title="Generated (t=$(t))",
        c=:viridis,
        aspect_ratio=:equal,
        showaxis=false,
        grid=false,
        colorbar=false
    )
    push!(plots_snapshots, plot(p1, p2, layout=(1,2), size=(800,400)))
end
# Combine all 6 pairs into a single figure (3 rows x 2 columns of pairs)
fig_snapshots = plot(plots_snapshots..., layout=(3,2), size=(1600, 1200), dpi=300)
display(fig_snapshots)
# Optionally save the figure
savefig(fig_snapshots, "reconstruction_snapshots.png")




##
#################### NEURAL NETWORK FOR Φ(t) CONSTRUCTION ####################

# Define the neural network architecture for Φ(t)
function create_phi_network(input_dim, output_dim)
    return Chain(
        Dense(input_dim, 128, relu),
        Dense(128, 64, relu),
        Dense(64, 32, relu),
        Dense(32, output_dim)  # Output flattened Φ matrix
    )
end

# Network parameters
periodic_input_dim = 2  # sin and cos of annual cycle
phi_output_dim = dim_x * dim_x  # Flattened Φ matrix

# Initialize the network
phi_network = create_phi_network(periodic_input_dim, phi_output_dim)

# Helper function to reshape network output to matrix form
function reshape_phi_output(phi_flat)
    return reshape(phi_flat, dim_x, dim_x)
end

# Helper function to create periodic input features
function create_periodic_features(t)
    return Float32[sin(2π * t / 360), cos(2π * t / 360)]
end

# Function to get Φ(t) from the network
function Φ_network(t)
    periodic_features = create_periodic_features(t)
    phi_flat = phi_network(periodic_features)
    return reshape_phi_output(phi_flat)
end

# Prepare training data for the neural network
function prepare_training_data(obs, obs_x, times, score_vanilla, window_size=5.0)
    println("Preparing training data for Φ network...")
    
    # Sample time points for training (use subset for efficiency)
    n_samples = min(5000, length(times))
    sample_indices = 1:div(length(times), n_samples):length(times)
    sample_times = times[sample_indices]
    
    # Pre-compute periodic times
    periodic_times = mod.(times, 360)
    
    training_data = []
    
    for (idx, target_time) in enumerate(sample_times)
        if idx % 500 == 0
            println("Processing sample $idx/$(length(sample_times))")
        end
        
        # Find indices within time window
        target_periodic = mod(target_time, 360)
        time_diffs = [min(abs(pt - target_periodic), 360 - abs(pt - target_periodic)) 
                     for pt in periodic_times]
        close_indices = findall(td -> td <= window_size, time_diffs)
        
        if length(close_indices) < 10
            continue  # Skip if not enough data points
        end
        
        # Extract subset of data
        obs_x_subset = obs_x[:, close_indices]
        obs_subset = obs[:, close_indices]
        
        # Calculate M matrix (local expectation)
        if size(obs_x_subset, 2) > 1
            dx_subset = obs_x_subset[:, 2:end] - obs_x_subset[:, 1:end-1]
            x_t_subset = obs_x_subset[:, 1:end-1]
            M_local = (dx_subset * x_t_subset') / size(x_t_subset, 2)
        else
            continue
        end
        
        # Calculate V matrix (score-state correlation)
        score_arr_subset = zeros(dim_x, size(obs_subset, 2))
        for j in 1:size(obs_subset, 2)
            score_arr_subset[:, j] = score_vanilla(obs_subset[:, j])
        end
        V_local = (score_arr_subset * obs_x_subset') / size(obs_x_subset, 2)
        
        # Store training sample
        push!(training_data, (
            time = target_time,
            periodic_features = create_periodic_features(target_time),
            M_target = M_local,
            V_matrix = V_local
        ))
    end
    
    return training_data
end

# Prepare the training data
training_data = prepare_training_data(obs, obs_x, times, score_vanilla)
println("Prepared $(length(training_data)) training samples")

# Training parameters
learning_rate = 0.001
n_epochs = 200
batch_size = 32

# Optimizer
optimizer = Adam(learning_rate)

# Training function
function train_phi_network!(network, training_data, optimizer, n_epochs)
    println("Training Φ network...")
    
    losses = Float64[]
    
    for epoch in 1:n_epochs
        epoch_loss = 0.0
        n_batches = 0
        
        # Shuffle training data
        shuffled_data = shuffle(training_data)
        
        # Process in batches
        for batch_start in 1:batch_size:length(shuffled_data)
            batch_end = min(batch_start + batch_size - 1, length(shuffled_data))
            batch_data = shuffled_data[batch_start:batch_end]
            
            # Compute loss and gradients
            loss, grads = Flux.withgradient(network) do model
                batch_loss = 0.0
                for sample in batch_data
                    # Get network prediction
                    phi_pred = reshape_phi_output(model(sample.periodic_features))
                    
                    # Compute target using moment matching: M = Φ * V^T
                    target = sample.M_target * pinv(sample.V_matrix)
                    
                    # L2 loss with regularization
                    reconstruction_loss = sum((phi_pred - target).^2)
                    regularization_loss = 0.001 * sum(phi_pred.^2)  # L2 regularization
                    
                    batch_loss += reconstruction_loss + regularization_loss
                end
                batch_loss / length(batch_data)
            end
            
            # Update parameters
            Flux.update!(optimizer, network, grads[1])
            
            epoch_loss += loss
            n_batches += 1
        end
        
        avg_loss = epoch_loss / n_batches
        push!(losses, avg_loss)
        
        if epoch % 20 == 0
            println("Epoch $epoch/$n_epochs, Loss: $(round(avg_loss, digits=6))")
        end
    end
    
    return losses
end

# Train the network
@time training_losses = train_phi_network!(phi_network, training_data, optimizer, n_epochs)

# Plot training loss
plot(training_losses, title="Φ Network Training Loss", xlabel="Epoch", ylabel="Loss", lw=2)

# Function to compute Σ(t) from the symmetric part of Φ(t)
function Σ_from_Φ(phi_matrix, regularization=1e-5)
    try
        # Extract symmetric part
        phi_symmetric = 0.5 * (phi_matrix + phi_matrix')
        
        # Ensure positive definiteness
        eigenvals, eigenvecs = eigen(Hermitian(phi_symmetric))
        
        # Keep only positive eigenvalues
        positive_mask = eigenvals .> 1e-6
        if sum(positive_mask) > 0
            pos_eigenvals = eigenvals[positive_mask]
            pos_eigenvecs = eigenvecs[:, positive_mask]
            
            # Reconstruct positive definite matrix
            phi_reconstructed = pos_eigenvecs * Diagonal(pos_eigenvals) * pos_eigenvecs'
        else
            # Fallback to small identity
            phi_reconstructed = regularization * I(size(phi_matrix, 1))
        end
        
        # Add regularization for numerical stability
        phi_final = phi_reconstructed + regularization * I(size(phi_matrix, 1))
        
        # Compute Σ via Cholesky decomposition
        return cholesky(Hermitian(phi_final)).L
        
    catch e
        println("Warning: Σ computation failed, using identity")
        return sqrt(regularization) * I(size(phi_matrix, 1))
    end
end

# Function to get Σ(t) from network
function Σ_network(t)
    phi_t = Φ_network(t)
    return Σ_from_Φ(phi_t)
end


# Updated functions to access time-dependent matrices
function Φ_at_time(t)
    return Φ_network(t)
end

function Σ_at_time(t)
    return Σ_network(t)
end

# Function to evaluate network performance
function evaluate_phi_network(test_times=1:10:360)
    println("Evaluating Φ network performance...")
    
    # Check periodicity
    phi_start = Φ_network(0.0)
    phi_end = Φ_network(360.0)
    periodicity_error = norm(phi_start - phi_end)
    println("Periodicity error: $periodicity_error")
    
    # Check positive definiteness of symmetric parts
    pos_def_count = 0
    for t in test_times
        phi_t = Φ_network(t)
        phi_sym = 0.5 * (phi_t + phi_t')
        if all(eigvals(phi_sym) .> -1e-6)
            pos_def_count += 1
        end
    end
    println("Positive definite symmetric parts: $pos_def_count/$(length(test_times))")
    
    # Plot matrix norms over time
    phi_norms = [norm(Φ_network(t)) for t in test_times]
    sigma_norms = [norm(Σ_network(t)) for t in test_times]
    
    p1 = plot(test_times, phi_norms, title="NN Φ(t) Matrix Norm", 
              xlabel="Time", ylabel="||Φ(t)||", lw=2, label="Φ NN")
    p2 = plot(test_times, sigma_norms, title="NN Σ(t) Matrix Norm", 
              xlabel="Time", ylabel="||Σ(t)||", lw=2, label="Σ NN", color=:red)
    
    return plot(p1, p2, layout=(1,2), size=(800, 300))
end

# Evaluate the trained network
evaluation_plot = evaluate_phi_network()
display(evaluation_plot)
