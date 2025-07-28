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
@time nn_vanilla, loss_vanilla = train(obs_uncorr, 40, 32, [dim, 128, 64, dim], σ_value; use_gpu=true, opt=Adam(0.0002))
nn_vanilla_cpu = nn_vanilla |> cpu
score_vanilla(x) = .- nn_vanilla_cpu(Float32.([x...])) ./ σ_value
Plots.plot(loss_vanilla)

##
###############################. CONSTRUCTION OF Φ AND Σ MATRICES (AUTONOMOUS) ###############################
# Get dimensions from the data
dim, num_timesteps = size(obs)

# Calculate state differences between consecutive time steps
dt = 1  # time step between consecutive points (assumed 1, adjust if needed)
dx = similar(obs[:, 2:end])
# Analytical derivatives for first two dimensions (sine and cosine)
for t in 1:size(dx, 2)
    # times[t] corresponds to the time at index t (for the left point of the diff)
    # If times is not integer, adjust accordingly
    time_val = times[t]
    # d/dt sin(2π t / 360) = (2π/360) * cos(2π t / 360)
    # d/dt cos(2π t / 360) = - (2π/360) * sin(2π t / 360)
    dx[1, t] = (2π/360) * cos(2π * time_val / 360)
    dx[2, t] = - (2π/360) * sin(2π * time_val / 360)
end
# For the remaining dimensions, use finite differences as before
dx[3:end, :] .= obs[3:end, 2:end] - obs[3:end, 1:end-1]

# Use states corresponding to the start of the time step
x_t = obs[:, 1:end-1]

# Estimate M using the expectation over the time series
# We average over (num_timesteps - 1) intervals
M_mat = (dx * x_t') / (num_timesteps - 1)

# --- 2. Calculate the V^T matrix ---
# V captures the covariance between the state and the score function.
score_arr = zeros(dim, num_timesteps) # Pre-allocate the matrix

for t in 1:num_timesteps
    # Get the state at time t
    current_state = obs[:, t]
    
    # Calculate the score for that state and store it in the corresponding column
    score_arr[:, t] = score_vanilla(current_state)
end

# Calculate V_T matrix using all data
V_T_mat = (score_arr * obs') / num_timesteps

# --- 3. Calculate Single Autonomous Φ and Σ Matrices ---
println("Calculating autonomous Φ and Σ matrices...")

# Calculate Φ using all data
Φ_autonomous = M_mat * pinv(V_T_mat)

# Initialize Σ to ensure it's always defined
Σ_autonomous = sqrt(1e-3) * I(dim)  # Default fallback value

# Ensure the symmetric part of Φ is positive definite
try
    Φ_symmetric = 0.5 * (Φ_autonomous .+ Φ_autonomous')  # Extract symmetric part
    
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
        Φ_symmetric_reconstructed = Hermitian(1e-3 * I(dim))
    end
    
    # Add regularization for numerical stability
    Φ_symmetric_final = Φ_symmetric_reconstructed + 1e-5 * I(dim)
    
    # Calculate Σ using the reconstructed positive-definite symmetric part
    Σ_autonomous = cholesky(Φ_symmetric_final).L
    
catch e
    println("Warning: Matrix calculation failed, using fallback identity")
    Σ_autonomous = sqrt(1e-3) * I(dim)
end

# Check positive definiteness of symmetric part
Φ_symmetric_check = 0.5 * (Φ_autonomous + Φ_autonomous')
min_eigenval = minimum(eigvals(Φ_symmetric_check))
println("Minimum eigenvalue of Φ_symmetric: ", min_eigenval)

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
drift_x_corrected(x_obs, t) = (Φ_autonomous * full_score(x_obs, t))[3:end]

# 4. Define the corrected noise matrix using time-dependent Σ matrices
sigma_x_corrected(x_obs, t) = Σ_autonomous[3:end, 3:end]


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

M_clustered = mean(trj_clustered, dims=2)
S_clustered = std(trj_clustered, dims=2)
trj_clustered = (trj_clustered .- M_clustered) ./ S_clustered

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
    # savefig(fig, "combined_figure_vanilla_aut_part$(fig_idx).pdf")
end

##

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
    savefig(fig, "bivar_vs_x123_aut_part$(fig_idx).pdf")
end