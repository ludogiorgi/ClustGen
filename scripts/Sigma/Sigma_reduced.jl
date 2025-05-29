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
obs_uncorr = obs[:, 1:1:end]

Plots.scatter(obs_uncorr[1,1:10000], markersize=2, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

function score_true_norm(x)
    return normalize_f(score_true, x, M, S)
end

normalization = false
σ_value = 0.05

averages, centers, Nc, labels = f_tilde_labels(σ_value, obs_uncorr; prob=0.01, do_print=true, conv_param=0.01, normalization=normalization)

if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
    # inputs_targets_residual, M_averages_values_residual, m_averages_values_residual = generate_inputs_targets(averages_residual, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
    # inputs_targets_residual = generate_inputs_targets(averages_residual, centers, Nc; normalization=false)
end

centers_sorted_indices = sortperm(centers[1,:])
centers_sorted = centers[:,centers_sorted_indices][:]
scores = .- averages[:,centers_sorted_indices][:] ./ σ_value
scores_true = [-score_true_norm(centers_sorted[i])[1] for i in eachindex(centers_sorted)]

Plots.scatter(centers_sorted, scores, color=:blue)
Plots.plot!(centers_sorted, .-scores_true, color=:red)

##
#################### TRAINING WITH CLUSTERING LOSS ####################

@time nn_clustered, loss_clustered = train(inputs_targets, 2000, 4, [dim, 50, 25, dim]; use_gpu=true, activation=swish, last_activation=identity)
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

Q = generator(labels;dt=dt)
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
plt = Plots.plot!(auto_Q[1,:])
##

gradLogp = zeros(dim, Nc)
for i in 1:Nc
    gradLogp[:,i] = - averages[:,i] / σ_value
end

C0 = centers * (centers * Diagonal(P_steady))'
C1_Q = centers * Q * (centers * Diagonal(P_steady))'
C1_grad = gradLogp * (centers * Diagonal(P_steady))'
Σ_test2 = C1_Q * inv(C1_grad)
Σ_test = cholesky(0.5*(Σ_test2 .+ Σ_test2')).L[1,1]
println("Σ_test = ", Σ_test)

Σ_test = computeSigma(centers, P_steady, Q, gradLogp)
##
#################### SAMPLES GENERATION ####################

score_clustered_xt(x,t) = Σ_test^2 * score_clustered(x)
score_true_xt(x,t) = Σ_test^2 * score_true_norm(x)
sigma_Σ(x,t) = Σ_test
Nsteps2 = Int(Nsteps)

trj_clustered = evolve([0.0], dt, Nsteps2, score_clustered_xt, sigma_Σ; timestepper=:euler, resolution=1)
# trj_true = evolve([0.0], dt, Nsteps2, score_true_xt, sigma_Σ; timestepper=:euler, resolution=1)

kde_clustered = kde(trj_clustered[:])
# kde_true = kde(trj_true[:])
kde_obs = kde(obs[:])

auto_clustered = zeros(dim, tsteps)
# auto_true = zeros(dim, tsteps)

for i in 1:dim
    auto_clustered[i,:] = autocovariance(trj_clustered[i,1:res:end]; timesteps=tsteps) 
#    auto_true[i,:] = autocovariance(trj_true[i,1:res:end]; timesteps=tsteps)
end

Nsamples = 4000
samples_obs = (evolve([0.0], dt, Nsamples, F, sigma1, sigma2; timestepper=:euler) .- M) ./ S
samples_clustered = evolve([0.0], dt, Nsamples, score_clustered_xt, sigma_Σ; timestepper=:euler)

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
        size=(800, 1000), dpi=300)

# Plot 1: Time series comparison
plt1 = Plots.plot(0:dt:Nsamples*dt, samples_obs[1,:], 
                 label="Observed", color=science_colors[1],
                 xlabel=L"t", ylabel=L"x(t)", 
                 title="Time Series Comparison",
                 legend=false)
Plots.plot!(plt1, 0:dt:Nsamples*dt, samples_clustered[1,:], 
           label="Model", color=science_colors[2], 
           linestyle=:solid)

# Plot 2: Probability density functions
plt2 = Plots.plot(kde_obs.x, kde_obs.density, 
                 label="Observed", color=science_colors[1],
                 xlabel=L"x", ylabel="Probability Density",
                 title="Probability Density Functions",
                 xlims=(-2.5, 5), legend=false)
Plots.plot!(plt2, kde_clustered.x, kde_clustered.density, 
           label="Model", color=science_colors[2],
           linestyle=:solid)

# Plot 3: Autocorrelation functions (keep the legend only here)
plt3 = Plots.plot(auto_obs[1,:], 
                 label="Observed", color=science_colors[1],
                 xlabel=L"Lag \times dt", ylabel="Autocorrelation",
                 title="Autocorrelation Functions",
                 legend=:topright)
Plots.plot!(plt3, auto_clustered[1,:] ./ var(trj_clustered[1,:]), 
           label="Model", color=science_colors[2], 
           linestyle=:solid)

# Combine plots with a main title
figure = Plots.plot(plt1, plt2, plt3, 
                   layout=(3, 1), 
                   size=(800, 1000),
                   grid=true,
                   plot_title="")

# Uncomment to save the figure in publication quality
# savefig(figure, "sigma_reduced_comparison.pdf")
#savefig(figure, "figures/Sigma_figures/reduced.png")

display(figure)

##
#################### DATA SAVING AND LOADING ####################

"""
    save_data_to_files(; filename_prefix="sigma_reduced")

Save all relevant data to HDF5 and BSON files in the data/Digma_data directory.
"""
function save_data_to_files(; filename_prefix="sigma_reduced")
    # Create directory if it doesn't exist
    data_dir = joinpath(@__DIR__, "..", "..", "data", "Digma_data")
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
        file["Sigma_test"] = Σ_test
        file["normalization"] = normalization
        file["M"] = M
        file["S"] = S
        
        # Save plot data
        file["samples_obs"] = samples_obs
        file["samples_clustered"] = samples_clustered
        file["kde_obs_x"] = [kde_obs.x...]
        file["kde_obs_density"] = kde_obs.density
        file["kde_clustered_x"] = [kde_clustered.x...]
        file["kde_clustered_density"] = kde_clustered.density
        file["auto_obs"] = auto_obs
        file["auto_clustered"] = auto_clustered
        file["dt"] = dt
        file["Nsamples"] = Nsamples
    end
    
    println("Data saved to: $hdf5_file")
    return hdf5_file, bson_file
end

"""
    load_data_from_files(; filename_prefix="sigma_reduced")

Load all data from HDF5 and BSON files in the data/Digma_data directory.
Returns a NamedTuple containing all the loaded data.
"""
function load_data_from_files(; filename_prefix="sigma_reduced")
    # Define filenames
    data_dir = joinpath(@__DIR__, "..", "..", "data", "Digma_data")
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
        normalization = read(file["normalization"])
        M = read(file["M"])
        S = read(file["S"])
        
        # Load plot data
        samples_obs = read(file["samples_obs"])
        samples_clustered = read(file["samples_clustered"])
        kde_obs_x = read(file["kde_obs_x"])
        kde_obs_density = read(file["kde_obs_density"])
        kde_clustered_x = read(file["kde_clustered_x"])
        kde_clustered_density = read(file["kde_clustered_density"])
        auto_obs = read(file["auto_obs"])
        auto_clustered = read(file["auto_clustered"])
        dt = read(file["dt"])
        Nsamples = read(file["Nsamples"])
        
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
            normalization = normalization,
            M = M,
            S = S,
            
            # Plot data
            samples_obs = samples_obs,
            samples_clustered = samples_clustered,
            kde_obs = (x = kde_obs_x, density = kde_obs_density),
            kde_clustered = (x = kde_clustered_x, density = kde_clustered_density),
            auto_obs = auto_obs,
            auto_clustered = auto_clustered,
            dt = dt,
            Nsamples = Nsamples
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
    kde_obs = data.kde_obs
    kde_clustered = data.kde_clustered
    auto_obs = data.auto_obs
    auto_clustered = data.auto_clustered
    dt = data.dt
    Nsamples = data.Nsamples
    
    # Define colors and style
    science_colors = [:navy, :crimson, :darkgreen, :purple, :orange]
    fontsize = 10
    linewidth = 2
    
    # Set plotting theme
    Plots.theme(:default)
    default(fontfamily="Computer Modern", framestyle=:box, 
            tickfont=fontsize, guidefont=fontsize+2, legendfont=fontsize,
            titlefont=fontsize+4, margin=5Plots.mm, linewidth=linewidth,
            size=(800, 1000), dpi=300)
    
    # Plot 1: Time series comparison
    plt1 = Plots.plot(0:dt:Nsamples*dt, samples_obs[1,:], 
                     label="Observed", color=science_colors[1],
                     xlabel=L"t", ylabel=L"x(t)", 
                     title="Time Series Comparison",
                     legend=false)
    Plots.plot!(plt1, 0:dt:Nsamples*dt, samples_clustered[1,:], 
               label="Model", color=science_colors[2], 
               linestyle=:solid)
    
    # Plot 2: Probability density functions
    plt2 = Plots.plot(kde_obs.x, kde_obs.density, 
                     label="Observed", color=science_colors[1],
                     xlabel=L"x", ylabel="Probability Density",
                     title="Probability Density Functions",
                     xlims=(-2.5, 5), legend=false)
    Plots.plot!(plt2, kde_clustered.x, kde_clustered.density, 
               label="Model", color=science_colors[2],
               linestyle=:solid)
    
    # Plot 3: Autocorrelation functions
    plt3 = Plots.plot(auto_obs[1,:], 
                     label="Observed", color=science_colors[1],
                     xlabel=L"Lag \times dt", ylabel="Autocorrelation",
                     title="Autocorrelation Functions",
                     legend=:topright)
    Plots.plot!(plt3, auto_clustered[1,:] ./ var(samples_clustered[1,:]), 
               label="Model", color=science_colors[2], 
               linestyle=:solid)
    
    # Combine plots with a main title
    figure = Plots.plot(plt1, plt2, plt3, 
                       layout=(3, 1), 
                       size=(800, 1000),
                       grid=true,
                       plot_title="Loaded from Saved Data")
    
    return figure
end
##
# Example usage:
# save_data_to_files()  # Save all data to files
data = load_data_from_files()  # Load data from files
plot_from_loaded_data(data)  # Create plots from loaded data
##
data.averages




