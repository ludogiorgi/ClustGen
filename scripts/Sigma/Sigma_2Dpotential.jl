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

function F(x, t; A1=1.0, A2=1.2, B1=0.6, B2=0.3, C=0.8)
    ∇U1 = 2 * (x[1] + A1) * (x[1] - A1)^2 + 2 * (x[1] - A1) * (x[1] + A1)^2 + B1
    ∇U2 = 2 * (x[2] + A2) * (x[2] - A2)^2 + 2 * (x[2] - A2) * (x[2] + A2)^2 + B2
    return [-∇U1 + C*∇U2, -∇U2 - C*∇U1]
end

# function F(x, t; A1=1.0, A2=1.0, B1=0.5, B2=0.5, C=1.0, D=1.0)
#     # Conservative gradient terms
#     ∇U1 = 2 * (x[1] + A1) * (x[1] - A1)^2 + 2 * (x[1] - A1) * (x[1] + A1)^2 + B1 + C * (x[1] * x[2])^2
#     ∇U2 = 2 * (x[2] + A2) * (x[2] - A2)^2 + 2 * (x[2] - A2) * (x[2] + A2)^2 + B2 + C * (x[1] * x[2])^2
    
#     # Non-conservative term (e.g., rotational flow)
#     F1 = -D * ∇U2
#     F2 = D * ∇U1
    
#     # Total force
#     return [-∇U1 + F1, -∇U2 + F2]
# end


dt = 0.025
Σ_true = [1.0 0.0; 0.0 1.0]
sigma(x,t) = Σ_true

function normalize_f(f, x, t, M, S)
    return f(x .* S .+ M, t) .* S
end

dt = 0.01
dim = 2

Nsteps = 1000000
obs_nn = evolve(zeros(dim), dt, Nsteps, F, sigma; resolution = 1, n_ens=100)

# plt1 = Plots.plot(obs_nn[1,1:1000], obs_nn[2,1:1000], label="Observed Trajectory", xlabel="X", ylabel="Y", title="Observed Trajectory")
# plt2 = Plots.plot(obs_nn[1,1:100:end], label="Observed Trajectory", xlabel="X", ylabel="Y", title="Observed Trajectory")
# Plots.plot(plt1, plt2, layout=(1,2), size=(800,400), title="Observed Trajectory", xlabel="X", ylabel="Y")
# ##
M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S

autocov_obs = zeros(dim, 300)
for i in 1:dim
    autocov_obs[i,:] = autocovariance(obs[i,1:100000]; timesteps=300)
end

autocov_obs_mean = mean(autocov_obs, dims=1)

Plots.plot(autocov_obs_mean[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")

##
obs_uncorr = obs[:, 1:10:end]

gr()
Plots.scatter(obs_uncorr[1,1:1:10000], obs_uncorr[2,1:1:10000], markersize=1, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

normalization = false
σ_value = 0.05

averages, _, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.0008, do_print=true, conv_param=0.005, normalization=normalization)

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

@time nn_clustered, loss_clustered = train(inputs_targets, 2000, 32, [dim, 100, 50, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value
Plots.plot(loss_clustered)

##

labels = [ssp.embedding(obs[:,i]) for i in 1:3:1000000]
Q = generator(labels;dt=3dt)
P_steady = steady_state(Q)

tsteps = 201
res = 10

auto_obs = zeros(dim, tsteps)
auto_Q = zeros(dim, tsteps)

for i in 1:dim
    auto_obs[i,:] = autocovariance(obs[i,1:res:1000000]; timesteps=tsteps) 
    auto_Q[i,:] = autocovariance(centers[i,:], Q, [0:dt*res:Int(res * (tsteps-1) * dt)...])
end

plt = Plots.plot(auto_obs[1,:])
plt = Plots.plot!(auto_Q[1,:])
plt = Plots.plot!(auto_Q[2,:])
plt = Plots.plot!(auto_Q[2,:])
display(plt)

gradLogp = zeros(dim, Nc)
for i in 1:Nc
    gradLogp[:,i] = - averages[:,i] / σ_value
end

C0 = centers * (centers * Diagonal(P_steady))'
C1_Q = centers * Q * (centers * Diagonal(P_steady))'
C1_grad = gradLogp * (centers * Diagonal(P_steady))'
Σ_test2 = C1_Q * inv(C1_grad)
Σ_test = cholesky(0.5*(Σ_test2 .+ Σ_test2')).L
println("Σ_test2 = ", Σ_test2)
##
#################### SAMPLES GENERATION ####################

Σ_test2 = [1.0 -0.8; 0.8 1.0]
Σ_test = [1.0 0.0; 0.0 1.0]
score_clustered_xt(x,t) = Σ_test2 * score_clustered(x)
score_true_xt(x,t) = Σ_test^2 * score_true(x, t)
sigma_Σ(x,t) = Matrix(Σ_test)

trj_clustered = evolve(zeros(dim), 0.2*dt, 100000, score_clustered_xt, sigma_Σ; timestepper=:euler, resolution=5, n_ens=100, boundary=[-10,10])
# trj_true = evolve([0.0], dt, Nsteps2, score_true_xt, sigma_Σ; timestepper=:euler, resolution=1)
trj_clustered_corr = evolve(zeros(dim), dt, 10000000, score_clustered_xt, sigma_Σ; timestepper=:rk4, resolution=1, n_ens=1, boundary=[-10,10])
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
    auto_clustered[i,:] = autocovariance(trj_clustered_corr[i,1:res:end]; timesteps=tsteps) 
#    auto_true[i,:] = autocovariance(trj_true[i,1:res:end]; timesteps=tsteps)
end

Nsamples = 4000
samples_obs = (evolve(zeros(dim), dt, Nsamples, F, sigma; timestepper=:euler) .- M) ./ S
samples_clustered = evolve(zeros(dim), dt, Nsamples, score_clustered_xt, sigma_Σ; timestepper=:euler)

##

# Define a scientific color palette
science_colors = [:navy, :crimson, :darkgreen, :purple, :orange]
fontsize = 10
linewidth = 2

# Set global plotting theme for scientific publication
Plots.theme(:default)
default(fontfamily="Computer Modern", framestyle=:box, 
        tickfont=fontsize, guidefont=fontsize+2, legendfont=fontsize,
        titlefont=fontsize+4, margin=5Plots.mm, linewidth=linewidth,
        size=(1200, 1000), dpi=300)  # Increased width to accommodate two columns

# Plot 1: Time series comparison - Dimension 1
plt1 = Plots.plot(0:dt:Nsamples*dt, samples_obs[1,:], 
                 label="Observed", color=science_colors[1],
                 xlabel=L"t", ylabel=L"x_1(t)", 
                 title="Time Series Comparison (Dim 1)",
                 legend=false)
Plots.plot!(plt1, 0:dt:Nsamples*dt, samples_clustered[1,:], 
           label="Model", color=science_colors[2], 
           linestyle=:solid)

# Plot 2: Probability density functions - Dimension 1
plt2 = Plots.plot(kde_obs_x[1], kde_obs_y[1], 
                 label="Observed", color=science_colors[1],
                 xlabel=L"x_1", ylabel="Probability Density",
                 title="Probability Density Functions (Dim 1)",
                 xlims=(-2.5, 4), legend=false)
Plots.plot!(plt2, kde_clustered_x[1], kde_clustered_y[1], 
           label="Model", color=science_colors[2],
           linestyle=:solid)

# Plot 3: Autocorrelation functions - Dimension 1
plt3 = Plots.plot(auto_obs[1,:], 
                 label="Observed", color=science_colors[1],
                 xlabel=L"Lag \times dt", ylabel="Autocorrelation",
                 title="Autocorrelation Functions (Dim 1)",
                 legend=false)
Plots.plot!(plt3, auto_clustered[1,:] ./ var(trj_clustered[1,:]), 
           label="Model", color=science_colors[2], 
           linestyle=:solid)

# NEW PLOTS FOR DIMENSION 2

# Plot 4: Time series comparison - Dimension 2
plt4 = Plots.plot(0:dt:Nsamples*dt, samples_obs[2,:], 
                 label="Observed", color=science_colors[1],
                 xlabel=L"t", ylabel=L"x_2(t)", 
                 title="Time Series Comparison (Dim 2)",
                 legend=false)
Plots.plot!(plt4, 0:dt:Nsamples*dt, samples_clustered[2,:], 
           label="Model", color=science_colors[2], 
           linestyle=:solid)

# Plot 5: Probability density functions - Dimension 2
plt5 = Plots.plot(kde_obs_x[2], kde_obs_y[2], 
                 label="Observed", color=science_colors[1],
                 xlabel=L"x_2", ylabel="Probability Density",
                 title="Probability Density Functions (Dim 2)",
                 xlims=(-2.5, 4), legend=false)
Plots.plot!(plt5, kde_clustered_x[2], kde_clustered_y[2], 
           label="Model", color=science_colors[2],
           linestyle=:solid)

# Plot 6: Autocorrelation functions - Dimension 2 (with legend)
plt6 = Plots.plot(auto_obs[2,:], 
                 label="Observed", color=science_colors[1],
                 xlabel=L"Lag \times dt", ylabel="Autocorrelation",
                 title="Autocorrelation Functions (Dim 2)",
                 legend=:topright)
Plots.plot!(plt6, auto_clustered[2,:] ./ var(trj_clustered[2,:]), 
           label="Model", color=science_colors[2], 
           linestyle=:solid)

# Combine plots with the same order as the original code
figure = Plots.plot(plt1, plt4, plt2, plt5, plt3, plt6,
                   layout=(3, 2), 
                   size=(1200, 1000),
                   grid=true,
                   plot_title="")

# Uncomment to save the figure in publication quality
# savefig(figure, "figures/Sigma_figures/2Dpotential.png")

display(figure)

##
#################### DATA SAVING AND LOADING ####################

"""
    save_data_to_files(; filename_prefix="sigma_2Dpotential")

Save all relevant data to HDF5 and BSON files in the data/Sigma_data directory.
"""
function save_data_to_files(; filename_prefix="sigma_2Dpotential2")
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
        
        # Save model parameters
        file["sigma_value"] = σ_value
        file["Sigma_test"] = Matrix(Σ_test)
        file["normalization"] = normalization
        file["M"] = M
        file["S"] = S
        
        # Save generator matrix and steady state
        file["Q"] = Array(Q)
        file["P_steady"] = P_steady
        
        # Save plot data
        file["samples_obs"] = samples_obs
        file["samples_clustered"] = samples_clustered
        
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
    end
    
    println("Data saved to: $hdf5_file")
    return hdf5_file, bson_file
end

"""
    load_data_from_files(; filename_prefix="sigma_2Dpotential")

Load all data from HDF5 and BSON files in the data/Sigma_data directory.
Returns a NamedTuple containing all the loaded data.
"""
function load_data_from_files(; filename_prefix="sigma_2Dpotential2")
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
        
        # Load model parameters
        sigma_value = read(file["sigma_value"])
        Sigma_test = read(file["Sigma_test"])
        normalization = read(file["normalization"])
        M = read(file["M"])
        S = read(file["S"])
        
        # Load generator matrix and steady state
        Q = read(file["Q"])
        P_steady = read(file["P_steady"])
        
        # Load plot data
        samples_obs = read(file["samples_obs"])
        samples_clustered = read(file["samples_clustered"])
        
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
        
        # Return as NamedTuple
        (
            # Clustering data
            averages = averages,
            centers = centers,
            Nc = Nc,
            
            # Model parameters
            sigma_value = sigma_value,
            Sigma_test = Sigma_test,
            normalization = normalization,
            M = M,
            S = S,
            
            # Generator matrix and steady state
            Q = Q,
            P_steady = P_steady,
            
            # Plot data
            samples_obs = samples_obs,
            samples_clustered = samples_clustered,
            
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
            dim = dim
        )
    end
    
    println("Data loaded from: $hdf5_file")
    println("Neural network loaded from: $bson_file")
    
    return merge(data, (nn_model = nn_model,))
end

"""
    plot_from_loaded_data(data)

Recreate the plots using the loaded data.
"""
function plot_from_loaded_data(data)
    # Extract data
    samples_obs = data.samples_obs
    samples_clustered = data.samples_clustered
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
    fontsize = 16  # Increased from 10 to 16 for better readability
    linewidth = 2.5  # Slightly thicker lines
    
    # Set plotting theme with larger fonts
    gr()
    Plots.theme(:default)
    default(fontfamily="Computer Modern", framestyle=:box, 
            tickfont=fontsize, guidefont=fontsize+4, legendfont=fontsize+2,
            titlefont=fontsize+6, margin=10Plots.mm, linewidth=linewidth,
            size=(1500, 1200), dpi=300)  # Adjusted size for 3x2 layout
    
    if dim == 2
        # Plot 1: Time series comparison - Dimension 1
        plt1 = Plots.plot(0:dt:Nsamples*dt, samples_obs[1,:], 
                     label="Observed", color=science_colors[1],
                     xlabel=L"t", ylabel=L"x_1(t)", 
                     title="Time Series Comparison (Dim 1)",
                     legend=false)
        Plots.plot!(plt1, 0:dt:Nsamples*dt, samples_clustered[1,:], 
               label="Model", color=science_colors[2], 
               linestyle=:solid)

        # Plot 4: Time series comparison - Dimension 2
        plt4 = Plots.plot(0:dt:Nsamples*dt, samples_obs[2,:], 
                     label="Observed", color=science_colors[1],
                     xlabel=L"t", ylabel=L"x_2(t)", 
                     title="Time Series Comparison (Dim 2)",
                     legend=false)
        Plots.plot!(plt4, 0:dt:Nsamples*dt, samples_clustered[2,:], 
               label="Model", color=science_colors[2], 
               linestyle=:solid)
        
        # Plot 2: Probability density functions - Dimension 1
        plt2 = Plots.plot(kde_obs_x[1], kde_obs_y[1], 
                     label="Observed", color=science_colors[1],
                     xlabel=L"x_1", ylabel="Probability Density",
                     title="Probability Density Functions (Dim 1)",
                     xlims=(-2.5, 3.5), legend=false)
        Plots.plot!(plt2, kde_clustered_x[1], kde_clustered_y[1], 
               label="Model", color=science_colors[2],
               linestyle=:solid)
        
        # Plot 5: Probability density functions - Dimension 2
        plt5 = Plots.plot(kde_obs_x[2], kde_obs_y[2], 
                     label="Observed", color=science_colors[1],
                     xlabel=L"x_2", ylabel="Probability Density",
                     title="Probability Density Functions (Dim 2)",
                     xlims=(-2.5, 3.5), legend=false)
        Plots.plot!(plt5, kde_clustered_x[2], kde_clustered_y[2], 
               label="Model", color=science_colors[2],
               linestyle=:solid)
        
        # Plot 3: Autocorrelation functions - Dimension 1
        plt3 = Plots.plot(auto_obs[1,:], 
                     label="Observed", color=science_colors[1],
                     xlabel=L"Lag \times dt", ylabel="Autocorrelation",
                     title="Autocorrelation Functions (Dim 1)",
                     legend=false)
        Plots.plot!(plt3, auto_clustered[1,:], 
               label="Model", color=science_colors[2], 
               linestyle=:solid)

        # Plot 6: Autocorrelation functions - Dimension 2 (with legend)
        plt6 = Plots.plot(auto_obs[2,:], 
                     label="Observed", color=science_colors[1],
                     xlabel=L"Lag \times dt", ylabel="Autocorrelation",
                     title="Autocorrelation Functions (Dim 2)",
                     legend=:topright)  # Only show legend on this plot
        Plots.plot!(plt6, auto_clustered[2,:], 
               label="Model", color=science_colors[2], 
               linestyle=:solid)
        
        # Combine plots into a 3x2 grid with adjusted margins
        figure = Plots.plot(
            plt1, plt4,  # Row 1: Time series plots
            plt2, plt5,  # Row 2: PDF plots
            plt3, plt6,  # Row 3: Autocorrelation plots
            layout=(3, 2),
            size=(1500, 1200),
            grid=true,
            left_margin=20Plots.mm,  # Increased left margin for y-labels
            bottom_margin=15Plots.mm  # Ensure x-labels are fully visible
        )
    else
        # For other dimensions, maintain the multi-dimension layout with increased font sizes
        ts_plots = []
        pdf_plots = []
        acf_plots = []
        
        for d in 1:dim
            # Time series comparison
            plt_ts = Plots.plot(0:dt:Nsamples*dt, samples_obs[d,:], 
                         label="Observed", color=science_colors[1],
                         xlabel=L"t", ylabel=L"x_%$d(t)", 
                         title="Time Series Comparison (Dim $d)",
                         legend=false)
            Plots.plot!(plt_ts, 0:dt:Nsamples*dt, samples_clustered[d,:], 
                   label="Model", color=science_colors[2], 
                   linestyle=:solid)
            push!(ts_plots, plt_ts)
            
            # Probability density functions
            plt_pdf = Plots.plot(kde_obs_x[d], kde_obs_y[d], 
                         label="Observed", color=science_colors[1],
                         xlabel=L"x_%$d", ylabel="Probability Density",
                         title="Probability Density Functions (Dim $d)",
                         xlims=(-2.5, 4), legend=false)
            Plots.plot!(plt_pdf, kde_clustered_x[d], kde_clustered_y[d], 
                   label="Model", color=science_colors[2],
                   linestyle=:solid)
            push!(pdf_plots, plt_pdf)
            
            # Autocorrelation functions
            legend_pos = (d == dim) ? :topright : false
            plt_acf = Plots.plot(auto_obs[d,:], 
                         label="Observed", color=science_colors[1],
                         xlabel=L"Lag \times dt", ylabel="Autocorrelation",
                         title="Autocorrelation Functions (Dim $d)",
                         legend=legend_pos)
            Plots.plot!(plt_acf, auto_clustered[d,:], 
                   label="Model", color=science_colors[2], 
                   linestyle=:solid)
            push!(acf_plots, plt_acf)
        end
        
        # Interleave the plots in the desired order
        all_plots = []
        for i in 1:dim
            push!(all_plots, ts_plots[i])
        end
        for i in 1:dim
            push!(all_plots, pdf_plots[i])
        end
        for i in 1:dim
            push!(all_plots, acf_plots[i])
        end
        
        figure = Plots.plot(all_plots..., 
                           layout=(3, dim), 
                           size=(600*dim, 1200),  # Adjusted height
                           grid=true,
                           left_margin=20Plots.mm,
                           bottom_margin=15Plots.mm)
    end
    
    return figure
end

# Example usage - uncomment these lines to use the functions
# save_data_to_files()  # Save data to files
data = load_data_from_files()  # Load data from files
figure = plot_from_loaded_data(data)  # Create plots from loaded data
savefig(figure, "figures/Sigma_figures/2Dpotential2.png")
display(figure)  # Display the figure