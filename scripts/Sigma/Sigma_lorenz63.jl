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
Nsteps = 1000000
obs_nn = evolve([1.0, 1.5, 1.8], dt, Nsteps, F, sigma; resolution = 1, n_ens=100)

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
obs_uncorr = obs[:, 1:10:end]

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

@time nn_clustered, loss_clustered = train(inputs_targets, 2000, 16, [dim, 100, 50, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value
Plots.plot(loss_clustered)


##

labels = [ssp.embedding(obs[:,i]) for i in 1:3:300000]
Q = generator(labels;dt=3dt)
P_steady = steady_state(Q)

tsteps = 51
res = 10

auto_obs = zeros(dim, tsteps)
auto_Q = zeros(dim, tsteps)

for i in 1:dim
    auto_obs[i,:] = autocovariance(obs[i,1:res:end]; timesteps=tsteps) 
    auto_Q[i,:] = autocovariance(centers[i,:], Q, [0:dt*res:Int(res * (tsteps-1) * dt)...])
end

plt = Plots.plot(auto_obs[1,:])
plt = Plots.plot!(auto_Q[3,:])
##

gradLogp = zeros(dim, Nc)
for i in 1:Nc
    gradLogp[:,i] = - averages[:,i] / σ_value
end

C0 = centers * (centers * Diagonal(P_steady))'
C1_Q = centers * Q * (centers * Diagonal(P_steady))'
C1_grad = gradLogp * (centers * Diagonal(P_steady))'
Σ_test2 = C1_Q * inv(C1_grad)
Σ_test = cholesky(0.5*(Σ_test2 .+ Σ_test2')).L
println("Σ_test = ", Σ_test)

##
#################### SAMPLES GENERATION ####################
score_clustered_xt(x,t) = Σ_test2 * score_clustered(x)
# score_true_xt(x,t) = Σ_test^2 * score_true(x, t)
sigma_Σ(x,t) = Matrix(Σ_test)

trj_clustered = evolve(zeros(dim), 0.2*dt, 100000, score_clustered_xt, sigma_Σ; timestepper=:euler, resolution=5, n_ens=100)
# trj_true = evolve([0.0], dt, Nsteps2, score_true_xt, sigma_Σ; timestepper=:euler, resolution=1)

# kde_true = kde(trj_true')

auto_clustered = zeros(dim, tsteps)
kde_clustered_x, kde_clustered_y = [], []
kde_obs_x, kde_obs_y = [], []
# auto_true = zeros(dim, tsteps)


for i in 1:dim
    kde_clustered_temp = kde(trj_clustered[i,:])
    push!(kde_clustered_x, [kde_clustered_temp.x...])
    push!(kde_clustered_y, kde_clustered_temp.density)
    kde_obs_temp = kde(obs[i,:])
    push!(kde_obs_x, [kde_obs_temp.x...])
    push!(kde_obs_y, kde_obs_temp.density)
    auto_clustered[i,:] = autocovariance(trj_clustered[i,1:res:end]; timesteps=tsteps) 
#    auto_true[i,:] = autocovariance(trj_true[i,1:res:end]; timesteps=tsteps)
end

Nsamples = 4000
samples_obs = (evolve(zeros(dim), dt, Nsamples, F, sigma; timestepper=:euler) .- M) ./ S
samples_clustered = evolve(zeros(dim), dt, Nsamples, score_clustered_xt, sigma_Σ; timestepper=:euler)

##
gr()
# Define a scientific color palette
science_colors = [:navy, :crimson, :darkgreen, :purple, :orange]
fontsize = 10
linewidth = 2

# Set global plotting theme for scientific publication
Plots.theme(:default)
default(fontfamily="Computer Modern", framestyle=:box, 
        tickfont=fontsize, guidefont=fontsize+2, legendfont=fontsize,
        titlefont=fontsize+4, margin=5Plots.mm, linewidth=linewidth,
        size=(1200, 1200), dpi=300)  # Increased height to accommodate three rows

# DIMENSION 1 PLOTS

# Plot 1: Time series comparison - Dimension 1
plt1 = Plots.plot(0:dt:Nsamples*dt, samples_obs[1,:], 
                 label="Observed", color=science_colors[1],
                 xlabel="t", ylabel="x1(t)", 
                 title="Time Series Comparison (Dim 1)",
                 legend=false)
Plots.plot!(plt1, 0:dt:Nsamples*dt, samples_clustered[1,:], 
           label="Model", color=science_colors[2], 
           linestyle=:solid)

# Plot 2: Probability density functions - Dimension 1
plt2 = Plots.plot(kde_obs_x[1], kde_obs_y[1], 
                 label="Observed", color=science_colors[1],
                 xlabel="x1", ylabel="Probability Density",
                 title="Probability Density Functions (Dim 1)",
                 xlims=(-4, 4), legend=false)
Plots.plot!(plt2, kde_clustered_x[1], kde_clustered_y[1], 
           label="Model", color=science_colors[2],
           linestyle=:solid)

# Plot 3: Autocorrelation functions - Dimension 1
plt3 = Plots.plot(auto_obs[1,:], 
                 label="Observed", color=science_colors[1],
                 xlabel="Lag × dt", ylabel="Autocorrelation",
                 title="Autocorrelation Functions (Dim 1)",
                 legend=false)
Plots.plot!(plt3, auto_clustered[1,:] ./ var(trj_clustered[1,:]), 
           label="Model", color=science_colors[2], 
           linestyle=:solid)

# DIMENSION 2 PLOTS

# Plot 4: Time series comparison - Dimension 2
plt4 = Plots.plot(0:dt:Nsamples*dt, samples_obs[2,:], 
                 label="Observed", color=science_colors[1],
                 xlabel="t", ylabel="x2(t)", 
                 title="Time Series Comparison (Dim 2)",
                 legend=false)
Plots.plot!(plt4, 0:dt:Nsamples*dt, samples_clustered[2,:], 
           label="Model", color=science_colors[2], 
           linestyle=:solid)

# Plot 5: Probability density functions - Dimension 2
plt5 = Plots.plot(kde_obs_x[2], kde_obs_y[2], 
                 label="Observed", color=science_colors[1],
                 xlabel="x2", ylabel="Probability Density",
                 title="Probability Density Functions (Dim 2)",
                 xlims=(-4, 4), legend=false)
Plots.plot!(plt5, kde_clustered_x[2], kde_clustered_y[2], 
           label="Model", color=science_colors[2],
           linestyle=:solid)

# Plot 6: Autocorrelation functions - Dimension 2
plt6 = Plots.plot(auto_obs[2,:], 
                 label="Observed", color=science_colors[1],
                 xlabel="Lag × dt", ylabel="Autocorrelation",
                 title="Autocorrelation Functions (Dim 2)",
                 legend=false)
Plots.plot!(plt6, auto_clustered[2,:] ./ var(trj_clustered[2,:]), 
           label="Model", color=science_colors[2], 
           linestyle=:solid)

# NEW PLOTS FOR DIMENSION 3

# Plot 7: Time series comparison - Dimension 3
plt7 = Plots.plot(0:dt:Nsamples*dt, samples_obs[3,:], 
                 label="Observed", color=science_colors[1],
                 xlabel="t", ylabel="x3(t)", 
                 title="Time Series Comparison (Dim 3)",
                 legend=false)
Plots.plot!(plt7, 0:dt:Nsamples*dt, samples_clustered[3,:], 
           label="Model", color=science_colors[2], 
           linestyle=:solid)

# Plot 8: Probability density functions - Dimension 3
plt8 = Plots.plot(kde_obs_x[3], kde_obs_y[3], 
                 label="Observed", color=science_colors[1],
                 xlabel="x3", ylabel="Probability Density",
                 title="Probability Density Functions (Dim 3)",
                 xlims=(-4, 4), legend=false)
Plots.plot!(plt8, kde_clustered_x[3], kde_clustered_y[3], 
           label="Model", color=science_colors[2],
           linestyle=:solid)

# Plot 9: Autocorrelation functions - Dimension 3 (with legend)
plt9 = Plots.plot(auto_obs[3,:], 
                 label="Observed", color=science_colors[1],
                 xlabel="Lag × dt", ylabel="Autocorrelation",
                 title="Autocorrelation Functions (Dim 3)",
                 legend=:topright)  # Only show legend on the last plot
Plots.plot!(plt9, auto_clustered[3,:] ./ var(trj_clustered[3,:]), 
           label="Model", color=science_colors[2], 
           linestyle=:solid)

# Combine plots in a 3×3 grid with increased left margin
figure = Plots.plot(
    plt1, plt4, plt7,  # Time series plots in first column
    plt2, plt5, plt8,  # PDF plots in second column
    plt3, plt6, plt9,  # Autocorrelation plots in third column
    layout=(3, 3),     # 3 rows, 3 columns
    size=(1800, 1200), # Wider to accommodate 3 columns
    grid=true,
    plot_title="Lorenz63 - Data from Files",
    left_margin=10Plots.mm  # Increase left margin to prevent y-label cutoff
)

# Save the figure in publication quality
savefig(figure, "figures/Sigma_figures/lorenz63.png")

display(figure)

##
#################### DATA SAVING AND LOADING ####################

"""
    save_data_to_files(; filename_prefix="sigma_lorenz63")

Save all relevant data to HDF5 and BSON files in the data/Sigma_data directory.
"""
function save_data_to_files(; filename_prefix="sigma_lorenz63")
    # Create directory if it doesn't exist
    data_dir = joinpath(@__DIR__, "..", "..", "data", "Sigma_data")
    mkpath(data_dir)
    
    # Define filenames
    hdf5_file = joinpath(data_dir, "$(filename_prefix)_data.h5")
    bson_file = joinpath(data_dir, "$(filename_prefix)_model.bson")
    
    # Save neural network to BSON
    BSON.@save bson_file nn_clustered_cpu
    println("Neural network saved to: $bson_file")
    
    # Save data to HDF5
    h5open(hdf5_file, "w") do file
        # Save clustering data
        file["averages"] = averages
        file["centers"] = centers
        file["Nc"] = Nc
        file["labels"] = labels
        
        # Save model parameters
        file["sigma_value"] = σ_value
        file["Sigma_test"] = Matrix(Σ_test)
        file["Sigma_test2"] = Σ_test2
        file["normalization"] = normalization
        file["M"] = M
        file["S"] = S
        
        # Save generator matrix and steady state
        file["Q"] = Array(Q)
        file["P_steady"] = P_steady
        
        # Save plot data
        file["samples_obs"] = samples_obs
        file["samples_clustered"] = samples_clustered
        file["trj_clustered"] = trj_clustered
        
        # Save KDE data - store each dimension separately to avoid HDF5 Vector{Any} issues
        for i in 1:dim
            file["kde_obs_x_$i"] = kde_obs_x[i]
            file["kde_obs_y_$i"] = kde_obs_y[i]
            file["kde_clustered_x_$i"] = kde_clustered_x[i]
            file["kde_clustered_y_$i"] = kde_clustered_y[i]
        end
        
        # Save autocorrelation data
        file["auto_obs"] = auto_obs
        file["auto_clustered"] = auto_clustered
        
        # Save parameters
        file["dt"] = dt
        file["Nsamples"] = Nsamples
        file["dim"] = dim
        file["resolution"] = res
        file["tsteps"] = tsteps
    end
    
    println("Data saved to: $hdf5_file")
    return hdf5_file, bson_file
end

"""
    load_data_from_files(; filename_prefix="sigma_lorenz63")

Load all data from HDF5 and BSON files in the data/Sigma_data directory.
Returns a NamedTuple containing all the loaded data.
"""
function load_data_from_files(; filename_prefix="sigma_lorenz63")
    # Define filenames
    data_dir = joinpath(@__DIR__, "..", "..", "data", "Sigma_data")
    hdf5_file = joinpath(data_dir, "$(filename_prefix)_data.h5")
    bson_file = joinpath(data_dir, "$(filename_prefix)_model.bson")
    
    # Check if files exist
    if !isfile(hdf5_file) || !isfile(bson_file)
        error("Data files not found. Run save_data_to_files() first.")
    end
    
    # Load neural network from BSON
    nn_model = BSON.load(bson_file)[:nn_clustered_cpu]
    
    # Load data from HDF5
    data = h5open(hdf5_file, "r") do file
        # Load clustering data
        averages = read(file["averages"])
        centers = read(file["centers"])
        Nc = read(file["Nc"])
        labels = read(file["labels"])
        
        # Load model parameters
        sigma_value = read(file["sigma_value"])
        Sigma_test = read(file["Sigma_test"])
        Sigma_test2 = read(file["Sigma_test2"])
        normalization = read(file["normalization"])
        M = read(file["M"])
        S = read(file["S"])
        
        # Load generator matrix and steady state
        Q = read(file["Q"])
        P_steady = read(file["P_steady"])
        
        # Load plot data
        samples_obs = read(file["samples_obs"])
        samples_clustered = read(file["samples_clustered"])
        trj_clustered = read(file["trj_clustered"])
        
        # Load KDE data - reconstruct arrays from individual dimension data
        dim = read(file["dim"])
        kde_obs_x = []
        kde_obs_y = []
        kde_clustered_x = []
        kde_clustered_y = []
        
        for i in 1:dim
            push!(kde_obs_x, read(file["kde_obs_x_$i"]))
            push!(kde_obs_y, read(file["kde_obs_y_$i"]))
            push!(kde_clustered_x, read(file["kde_clustered_x_$i"]))
            push!(kde_clustered_y, read(file["kde_clustered_y_$i"]))
        end
        
        # Load autocorrelation data
        auto_obs = read(file["auto_obs"])
        auto_clustered = read(file["auto_clustered"])
        
        # Load parameters
        dt = read(file["dt"])
        Nsamples = read(file["Nsamples"])
        resolution = read(file["resolution"])
        tsteps = read(file["tsteps"])
        
        # Return as NamedTuple
        (
            # Clustering data
            averages = averages,
            centers = centers,
            Nc = Nc,
            labels = labels,
            
            # Model parameters
            sigma_value = sigma_value,
            Sigma_test = Sigma_test,
            Sigma_test2 = Sigma_test2,
            normalization = normalization,
            M = M,
            S = S,
            
            # Generator matrix and steady state
            Q = Q,
            P_steady = P_steady,
            
            # Plot data
            samples_obs = samples_obs,
            samples_clustered = samples_clustered,
            trj_clustered = trj_clustered,
            
            # KDE data
            kde_obs_x = kde_obs_x,
            kde_obs_y = kde_obs_y, 
            kde_clustered_x = kde_clustered_x,
            kde_clustered_y = kde_clustered_y,
            
            # Autocorrelation data
            auto_obs = auto_obs,
            auto_clustered = auto_clustered,
            
            # Parameters
            dt = dt,
            Nsamples = Nsamples,
            resolution = resolution,
            tsteps = tsteps,
            dim = dim
        )
    end
    
    println("Data loaded from: $hdf5_file")
    println("Neural network loaded from: $bson_file")
    
    return merge(data, (nn_model = nn_model,))
end

"""
    plot_from_loaded_data(data)

Recreate the plots using the loaded data with LaTeX notation for labels.
"""
function plot_from_loaded_data(data)
    # Extract data
    samples_obs = data.samples_obs
    samples_clustered = data.samples_clustered
    trj_clustered = data.trj_clustered
    kde_obs_x = data.kde_obs_x
    kde_obs_y = data.kde_obs_y
    kde_clustered_x = data.kde_clustered_x
    kde_clustered_y = data.kde_clustered_y
    auto_obs = data.auto_obs
    auto_clustered = data.auto_clustered
    dt = data.dt
    Nsamples = data.Nsamples
    dim = data.dim
    
    # Define colors and style with increased font sizes
    science_colors = [:navy, :crimson, :darkgreen, :purple, :orange]
    fontsize = 16  # Increased from 10 to 16
    linewidth = 2.5  # Slightly thicker lines
    
    # Set plotting theme with larger fonts
    gr()
    Plots.theme(:default)
    default(fontfamily="Computer Modern", framestyle=:box, 
            tickfont=fontsize, guidefont=fontsize+4, legendfont=fontsize+2,
            titlefont=fontsize+6, margin=10Plots.mm, linewidth=linewidth,
            size=(1800, 1200), dpi=300)
    
    # Create all plots with the same content but enhanced readability
    # DIMENSION 1 PLOTS
    plt1 = Plots.plot(0:dt:Nsamples*dt, samples_obs[1,:], 
                     label="Observed", color=science_colors[1],
                     xlabel=L"t", ylabel=L"x_1(t)", 
                     title="Time Series Comparison (Dim 1)",
                     legend=false)
    Plots.plot!(plt1, 0:dt:Nsamples*dt, samples_clustered[1,:], 
               label="Model", color=science_colors[2], 
               linestyle=:solid)

    # Remaining plots follow the same pattern...
    # (code for plots 2-9 with same font settings)
    plt2 = Plots.plot(kde_obs_x[1], kde_obs_y[1], 
                     label="Observed", color=science_colors[1],
                     xlabel=L"x_1", ylabel="Probability Density",
                     title="Probability Density Functions (Dim 1)",
                     xlims=(-4, 4), legend=false)
    Plots.plot!(plt2, kde_clustered_x[1], kde_clustered_y[1], 
               label="Model", color=science_colors[2],
               linestyle=:solid)

    plt3 = Plots.plot(auto_obs[1,:], 
                     label="Observed", color=science_colors[1],
                     xlabel=L"Lag \times dt", ylabel="Autocorrelation",
                     title="Autocorrelation Functions (Dim 1)",
                     legend=false)
    Plots.plot!(plt3, auto_clustered[1,:] ./ var(trj_clustered[1,:]), 
               label="Model", color=science_colors[2], 
               linestyle=:solid)

    # DIMENSION 2 PLOTS
    plt4 = Plots.plot(0:dt:Nsamples*dt, samples_obs[2,:], 
                     label="Observed", color=science_colors[1],
                     xlabel=L"t", ylabel=L"x_2(t)", 
                     title="Time Series Comparison (Dim 2)",
                     legend=false)
    Plots.plot!(plt4, 0:dt:Nsamples*dt, samples_clustered[2,:], 
               label="Model", color=science_colors[2], 
               linestyle=:solid)

    plt5 = Plots.plot(kde_obs_x[2], kde_obs_y[2], 
                     label="Observed", color=science_colors[1],
                     xlabel=L"x_2", ylabel="Probability Density",
                     title="Probability Density Functions (Dim 2)",
                     xlims=(-4, 4), legend=false)
    Plots.plot!(plt5, kde_clustered_x[2], kde_clustered_y[2], 
               label="Model", color=science_colors[2],
               linestyle=:solid)

    plt6 = Plots.plot(auto_obs[2,:], 
                     label="Observed", color=science_colors[1],
                     xlabel=L"Lag \times dt", ylabel="Autocorrelation",
                     title="Autocorrelation Functions (Dim 2)",
                     legend=false)
    Plots.plot!(plt6, auto_clustered[2,:] ./ var(trj_clustered[2,:]), 
               label="Model", color=science_colors[2], 
               linestyle=:solid)

    # DIMENSION 3 PLOTS
    plt7 = Plots.plot(0:dt:Nsamples*dt, samples_obs[3,:], 
                     label="Observed", color=science_colors[1],
                     xlabel=L"t", ylabel=L"x_3(t)",
                     title="Time Series Comparison (Dim 3)",
                     legend=false)
    Plots.plot!(plt7, 0:dt:Nsamples*dt, samples_clustered[3,:], 
               label="Model", color=science_colors[2], 
               linestyle=:solid)

    plt8 = Plots.plot(kde_obs_x[3], kde_obs_y[3], 
                     label="Observed", color=science_colors[1],
                     xlabel=L"x_3", ylabel="Probability Density",
                     title="Probability Density Functions (Dim 3)",
                     xlims=(-4, 4), legend=false)
    Plots.plot!(plt8, kde_clustered_x[3], kde_clustered_y[3], 
               label="Model", color=science_colors[2],
               linestyle=:solid)

    plt9 = Plots.plot(auto_obs[3,:], 
                     label="Observed", color=science_colors[1],
                     xlabel=L"Lag \times dt", ylabel="Autocorrelation",
                     title="Autocorrelation Functions (Dim 3)",
                     legend=:topright)  # Only show legend on the last plot
    Plots.plot!(plt9, auto_clustered[3,:] ./ var(trj_clustered[3,:]), 
               label="Model", color=science_colors[2], 
               linestyle=:solid)

    # Combine plots in a 3×3 grid with increased margins
    figure = Plots.plot(
        plt1, plt4, plt7,  # Time series plots in first column
        plt2, plt5, plt8,  # PDF plots in second column
        plt3, plt6, plt9,  # Autocorrelation plots in third column
        layout=(3, 3),     # 3 rows, 3 columns
        size=(2000, 1400), # Increased size to accommodate larger fonts
        grid=true,
        plot_title="",
        left_margin=25Plots.mm,  # Increased left margin 
        bottom_margin=15Plots.mm, # Ensure x-labels are fully visible
        right_margin=30Plots.mm   # Ensure y-labels are fully visible
    )
    
    return figure
end

# Example usage
# save_data_to_files()  # Save data to files
data = load_data_from_files()  # Load data from files
figure = plot_from_loaded_data(data)  # Create plots from loaded data with LaTeX labels
savefig(figure, "figures/Sigma_figures/lorenz63.png")  # Save the figure
display(figure)  # Display the figure
