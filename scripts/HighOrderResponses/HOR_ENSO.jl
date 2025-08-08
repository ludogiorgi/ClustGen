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

function F(x, t; dᵤ=0.2, wᵤ=0.4, dₜ=2.0)
    F1 = - dᵤ*x[1] - wᵤ*x[2] + x[3]
    F2 = - dᵤ*x[2] + wᵤ*x[1]
    F3 = - dₜ*x[3]
    return [F1, F2, F3]
end

function sigma(x, t; σ₁=0.3, σ₂=0.3)
    sigma1 = σ₁
    sigma2 = σ₂
    sigma3 = 1.5*(tanh(x[1]) + 1)
    return [sigma1, sigma2, sigma3]
end

dim = 3
dt = 0.01
Nsteps = 100000000
obs_nn = evolve([0.0, 0.0, 0.0], dt, Nsteps, F, sigma; resolution = 10)
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

Plots.scatter(obs_uncorr[1,1:10000], obs_uncorr[2,1:10000], obs_uncorr[3,1:10000], markersize=1, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

normalization = false
σ_value = 0.05

averages, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.0001, do_print=true, conv_param=0.002, normalization=normalization)

if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
end

plotly()
targets_norm = [norm(averages[:,i]) for i in eachindex(centers[1,:])]
Plots.scatter(centers[1,:], centers[2,:], centers[3,:], marker_z=targets_norm, color=:viridis)
##
#################### TRAINING WITH CLUSTERING LOSS ####################

@time nn_clustered, loss_clustered = train(inputs_targets, 200, 32, [dim, 100, 50, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value
Plots.plot(loss_clustered)

##
#################### NN SAVINGS ####################

BSON.@save pwd() * "/data/HOR_data/nn_ENSO.bson" nn_clustered_cpu
##
#################### NN LOADINGS ####################

BSON.load(pwd() * "/data/HOR_data/nn_ENSO.bson")[:nn_clustered_cpu]

##


##
#################### SAMPLES GENERATION ####################

score_clustered_xt(x,t) = score_clustered(x)
score_qG_xt(x,t) = score_qG(x)
sigma_I(x,t) = 1.0

trj_clustered = evolve([0.0, 0.0, 0.0], 0.05*dt, Nsteps, score_clustered_xt, sigma_I; timestepper=:rk4, resolution=5, boundary=[-15,15])

kde_clustered_x = kde(trj_clustered[1,:])
kde_true_x = kde(obs[1,:])

kde_clustered_y = kde(trj_clustered[2,:])
kde_true_y = kde(obs[2,:])

kde_clustered_z = kde(trj_clustered[3,:])
kde_true_z = kde(obs[3,:])

plt1 = Plots.plot(kde_clustered_x.x, kde_clustered_x.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
plt1 = Plots.plot!(kde_true_x.x, kde_true_x.density, label="True", xlabel="X", ylabel="Density", title="True PDF")

plt2 = Plots.plot(kde_clustered_y.x, kde_clustered_y.density, label="Observed", xlabel="Y", ylabel="Density", title="Observed PDF")
plt2 = Plots.plot!(kde_true_y.x, kde_true_y.density, label="True", xlabel="Y", ylabel="Density", title="True PDF")

plt3 = Plots.plot(kde_clustered_z.x, kde_clustered_z.density, label="Observed", xlabel="Z", ylabel="Density", title="Observed PDF")
plt3 = Plots.plot!(kde_true_z.x, kde_true_z.density, label="True", xlabel="Z", ylabel="Density", title="True PDF")

Plots.plot(plt1, plt2, plt3, layout=(3, 1), size=(800, 800))
##
kde_obs_xy = kde(obs[[1,2],:]')
kde_obs_xz = kde(obs[[1,3],:]')
kde_obs_yz = kde(obs[[2,3],:]')

kde_clustered_xy = kde(trj_clustered[[1,2],:]')
kde_clustered_xz = kde(trj_clustered[[1,3],:]')
kde_clustered_yz = kde(trj_clustered[[2,3],:]')

gr()

plt1 = Plots.heatmap(kde_obs_xy.x, kde_obs_xy.y, kde_obs_xy.density, xlabel="X", ylabel="Y", title="Observed PDF XY")
plt2 = Plots.heatmap(kde_obs_xz.x, kde_obs_xz.y, kde_obs_xz.density, xlabel="X", ylabel="Z", title="Observed PDF XZ")
plt3 = Plots.heatmap(kde_obs_yz.x, kde_obs_yz.y, kde_obs_yz.density, xlabel="Y", ylabel="Z", title="Observed PDF YZ")
plt4 = Plots.heatmap(kde_clustered_xy.x, kde_clustered_xy.y, kde_clustered_xy.density, xlabel="X", ylabel="Y", title="Sampled PDF XY", xrange=(kde_obs_xy.x[1], kde_obs_xy.x[end]), yrange=(kde_obs_xy.y[1], kde_obs_xy.y[end]))
plt5 = Plots.heatmap(kde_clustered_xz.x, kde_clustered_xz.y, kde_clustered_xz.density, xlabel="X", ylabel="Z", title="Sampled PDF XZ", xrange=(kde_obs_xz.x[1], kde_obs_xz.x[end]), yrange=(kde_obs_xz.y[1], kde_obs_xz.y[end]))
plt6 = Plots.heatmap(kde_clustered_yz.x, kde_clustered_yz.y, kde_clustered_yz.density, xlabel="Y", ylabel="Z", title="Sampled PDF YZ", xrange=(kde_obs_yz.x[1], kde_obs_yz.x[end]), yrange=(kde_obs_yz.y[1], kde_obs_yz.y[end]))

Plots.plot(plt1, plt2, plt3, plt4, plt5, plt6)
##

using ClustGen
f(t) = 1.0

res_trj = 2
steps_trj = 1000000
trj = obs[:,1:res_trj:steps_trj*res_trj]

ϵ = 0.01
u(x) = [0.0, 0.0, -0.4*x[3]]
div_u(x) = -0.4

F_pert(x,t) = F(x,t) + u(x) * f(t)

invC0 = inv(cov(obs'))
score_qG(x) = - invC0*x

score_gen(x) = score_clustered(x)

dim_Obs = 3
n_tau = 50

# R_num, δObs_num = zeros(4, dim, n_tau+1), zeros(4, dim, n_tau+1)
# R_lin, δObs_lin = zeros(4, dim, n_tau+1), zeros(4, dim, n_tau+1)
R_gen, δObs_gen = zeros(4, dim, n_tau+1), zeros(4, dim, n_tau+1)

# R_num[1,:,:], R_num[2,:,:], R_num[3,:,:], R_num[4,:,:] = generate_numerical_response_HO(F, u, dim, dt, n_tau, 600, sigma, M; n_ens=100000, resolution=10*res_trj, timestepper=:rk4)

for i in 1:4
    Obs(x) = x.^i
    # R_lin[i,:,:], δObs_lin[i,:,:] = generate_score_response(trj, u, div_u, f, score_qG, res_trj*dt, n_tau, Obs, dim_Obs)
    R_gen[i,:,:], δObs_gen[i,:,:] = generate_score_response(trj, u, div_u, f, score_gen, res_trj*dt, n_tau, Obs, dim_Obs)
end


##
R_gen_hack = copy(R_gen)

gr()
plt1 = Plots.plot(R_num[1,1,:], legend=false, label="Numerical", xlabel="Time lag", ylabel="Response", title="Response 1st moment")
plt1 = Plots.plot!(R_lin[1,1,:], label="Linear")
plt1 = Plots.plot!(R_gen_hack[1,1,:], label="Generative")
plt2 = Plots.plot(R_num[2,1,:] ./ S[1], legend=false, xlabel="Time lag", ylabel="Response", title="Response 2nd moment")
plt2 = Plots.plot!(R_lin[2,1,:])
plt2 = Plots.plot!(R_gen_hack[2,1,:])
plt3 = Plots.plot(R_num[3,1,:] ./ S[1]^2, legend=false, xlabel="Time lag", ylabel="Response", title="Response 3rd moment")
plt3 = Plots.plot!(R_lin[3,1,:])
plt3 = Plots.plot!(R_gen_hack[3,1,:], label="Generative")
plt4 = Plots.plot(R_num[4,1,:] ./ S[1]^3, legend=false, xlabel="Time lag", ylabel="Response", title="Response 4th moment")
plt4 = Plots.plot!(R_lin[4,1,:])
plt4 = Plots.plot!(R_gen_hack[4,1,:])

plt5 = Plots.plot(R_num[1,2,:], legend=false, label="Numerical", xlabel="Time lag", ylabel="Response", title="Response 1st moment")
plt5 = Plots.plot!(R_lin[1,2,:], label="Linear")
plt5 = Plots.plot!(R_gen_hack[1,2,:], label="Generative")
plt6 = Plots.plot(R_num[2,2,:] ./ S[1], legend=false, xlabel="Time lag", ylabel="Response", title="Response 2nd moment")
plt6 = Plots.plot!(R_lin[2,2,:], label="Linear")
plt6 = Plots.plot!(R_gen_hack[2,2,:], label="Generative")
plt7 = Plots.plot(R_num[3,2,:] ./ S[1]^2, legend=false, xlabel="Time lag", ylabel="Response", title="Response 3rd moment")
plt7 = Plots.plot!(R_lin[3,2,:], label="Linear")
plt7 = Plots.plot!(R_gen_hack[3,2,:], label="Generative")
plt8 = Plots.plot(R_num[4,2,:] ./ S[1]^3, legend=false, xlabel="Time lag", ylabel="Response", title="Response 4th moment")
plt8 = Plots.plot!(R_lin[4,2,:], label="Linear")
plt8 = Plots.plot!(R_gen_hack[4,2,:], label="Generative")

plt9 = Plots.plot(R_num[1,3,:], legend=false, label="Numerical", xlabel="Time lag", ylabel="Response", title="Response 1st moment")
plt9 = Plots.plot!(R_lin[1,3,:], label="Linear")
plt9 = Plots.plot!(R_gen_hack[1,3,2:end], label="Generative")
plt10 = Plots.plot(R_num[2,3,:] ./ S[1], legend=false, xlabel="Time lag", ylabel="Response", title="Response 2nd moment")
plt10 = Plots.plot!(R_lin[2,3,:], label="Linear")
plt10 = Plots.plot!(R_gen_hack[2,3,1:end], label="Generative")
plt11 = Plots.plot(R_num[3,3,:] ./ S[1]^2, legend=false, xlabel="Time lag", ylabel="Response", title="Response 3rd moment")
plt11 = Plots.plot!(R_lin[3,3,:], label="Linear")
plt11 = Plots.plot!(R_gen_hack[3,3,2:end], label="Generative")
plt12 = Plots.plot(R_num[4,3,:] ./ S[1]^3, legend=false, xlabel="Time lag", ylabel="Response", title="Response 4th moment")
plt12 = Plots.plot!(R_lin[4,3,:], label="Linear")
plt12 = Plots.plot!(R_gen_hack[4,3,2:end], label="Generative")

Plots.plot(plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8, plt9, plt10, plt11, plt12,
    layout=(3, 4), size=(1200, 800))
##
R0_gen = zeros(dim,dim)
for j in 1:steps_trj
    R0_gen .+= trj[:,j] * score_gen(trj[:,j])'
end
R0_gen ./= (steps_trj)
invR0_gen = .-inv(R0_gen)
    
R_gen_hack = zeros(4, dim, n_tau+1)
for i in 1:4
    for j in 1:n_tau+1
        R_gen_hack[i,:,j] = R_gen[i,:,j]'*invR0_gen 
    end
end

##

using Distributions

kde_clustered_x_x = [kde_clustered_x.x...]
kde_clustered_y_x = [kde_clustered_y.x...]
kde_clustered_z_x = [kde_clustered_z.x...]
kde_clustered_x_y = kde_clustered_x.density
kde_clustered_y_y = kde_clustered_y.density
kde_clustered_z_y = kde_clustered_z.density

kde_true_x_x = [kde_true_x.x...]
kde_true_y_x = [kde_true_y.x...]
kde_true_z_x = [kde_true_z.x...]
kde_true_x_y = kde_true_x.density
kde_true_y_y = kde_true_y.density
kde_true_z_y = kde_true_z.density

kde_gauss_x = Distributions.pdf.(Normal(0, 1), kde_clustered_x_x)
kde_gauss_y = Distributions.pdf.(Normal(0, 1), kde_clustered_y_x)
kde_gauss_z = Distributions.pdf.(Normal(0, 1), kde_clustered_z_x)

##

using GLMakie
using Distributions

# Create figure with 3×5 layout (3 dimensions, PDF + 4 moments)
fig = Figure(resolution=(1800, 1000), font="CMU Serif", fontsize=24)

# Define common elements - update colors to match your previous plots
colors = [:blue, :black, :red]  # True/Numerical, Linear, KGMM/Generative
labels = ["Dynamical", "Gaussian", "KGMM"]
time_axis = 0:10*dt*res_trj:n_tau*10*dt*res_trj

# Create axes array - 5 columns (PDFs + 4 moments)
axes = Matrix{Axis}(undef, 3, 5)

# Column titles
titles = ["PDF", "1st moment", "2nd moment", 
          "3rd moment", "4th moment"]
# Row labels for 3D system
ylabels = ["u₁", "u₂", "τ"]

# Calculate y-axis limits for each column of response functions (columns 2-5)
y_limits = zeros(4, 2)  # [moment, min/max]
for j in 1:4  # For each moment
    all_values = []
    for i in 1:3  # For each dimension
        # Include all response functions in limit calculation
        if @isdefined(R_num)
            if j == 1
                push!(all_values, R_num[j,i,:])
            else
                push!(all_values, R_num[j,i,:] ./ S[i]^(j-1))  # Normalize higher moments
            end
        end
        if @isdefined(R_lin)
            push!(all_values, R_lin[j,i,:])
        end
        push!(all_values, R_gen_hack[j,i,:])  # Use R_gen_hack as you did in Plots
    end
    
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

# Create subplots
for i in 1:3
    # First column: PDF plots
    axes[i,1] = Axis(fig[i,1],
        xlabel = (i == 3) ? "Value" : "",
        ylabel = ylabels[i],
        title = (i == 1) ? titles[1] : "",
        titlesize = 36,
        xlabelsize = 28,
        ylabelsize = 28,
        xticklabelsize = 24,
        yticklabelsize = 24
    )
    
    # Get the appropriate KDE data based on dimension
    if i == 1      # x dimension
        true_x = kde_true_x_x
        true_density = kde_true_x_y
        clustered_x = kde_clustered_x_x
        clustered_density = kde_clustered_x_y
        gauss_density = kde_gauss_x
    elseif i == 2  # y dimension
        true_x = kde_true_y_x
        true_density = kde_true_y_y
        clustered_x = kde_clustered_y_x
        clustered_density = kde_clustered_y_y
        gauss_density = kde_gauss_y
    else           # z dimension
        true_x = kde_true_z_x
        true_density = kde_true_z_y
        clustered_x = kde_clustered_z_x
        clustered_density = kde_clustered_z_y
        gauss_density = kde_gauss_z
    end
    
    # Plot PDFs
    lines!(axes[i,1], true_x, true_density, color=colors[1], linewidth=3)
    lines!(axes[i,1], clustered_x, gauss_density, color=colors[2], linewidth=3) 
    lines!(axes[i,1], clustered_x, clustered_density, color=colors[3], linewidth=3)
    
    # Response function plots (columns 2-5)
    for j in 1:4
        response_col = j + 1  # Column index in the figure (PDF is col 1)
        axes[i,response_col] = Axis(fig[i,response_col],
            xlabel = (i == 3) ? "Time lag" : "",
            ylabel = "",  # Only first column gets y labels
            title = (i == 1) ? titles[response_col] : "",
            titlesize = 36,
            xlabelsize = 28,
            ylabelsize = 28,
            xticklabelsize = 24,
            yticklabelsize = 24
            #limits = (nothing, (y_limits[j, 1], y_limits[j, 2]))
        )

        # Plot numerical response data (normalize higher moments)
        if @isdefined(R_num) && size(R_num, 1) >= j && size(R_num, 2) >= i
            if j == 1
                lines!(axes[i,response_col], time_axis, R_num[j,i,:], color=colors[1], linewidth=3)
            else
                lines!(axes[i,response_col], time_axis, R_num[j,i,:] ./ S[i]^(j-1), color=colors[1], linewidth=3)
            end
        end
        
        # Plot linear response data
        if @isdefined(R_lin) && size(R_lin, 1) >= j && size(R_lin, 2) >= i
            lines!(axes[i,response_col], time_axis, R_lin[j,i,:], color=colors[2], linewidth=3)
        end

        # Plot generative response data using R_gen_hack as in your Plots.jl code
        if @isdefined(R_gen_hack) && size(R_gen_hack, 1) >= j && size(R_gen_hack, 2) >= i
            # For z dimension (i=3), skip the first time point as in your plots
            lines!(axes[i,response_col], time_axis, R_gen_hack[j,i,:], color=colors[3], linewidth=3)
        end
        
        # Add grid lines for readability
        axes[i,response_col].xgridvisible = true
        axes[i,response_col].ygridvisible = true
    end
end

# Add unified legend at the bottom
Legend(fig[4, :],
    [LineElement(color=c, linewidth=3, linestyle=:solid)
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
fig.layout[5, :] = GridLayout(height=20)

# Make sure the directory exists
mkpath("figures/HOR_figures")

# Save figure
save("figures/HOR_figures/ENSO_responses.png", fig, px_per_unit=2)

fig

##

# # Save all relevant variables for the ENSO responses figure
# # Make sure the directory exists
# mkpath("data/HOR_data")

# # Create dictionary with all the variables needed for the plot
# data_to_save = Dict(
#     # Response functions
#     "R_gen_hack" => R_gen_hack,
#     "R_lin" => R_lin,
#     "R_num" => R_num,
    
#     # PDF data for each dimension
#     # X dimension
#     "kde_true_x_x" => kde_true_x_x,
#     "kde_true_x_y" => kde_true_x_y,
#     "kde_clustered_x_x" => kde_clustered_x_x,
#     "kde_clustered_x_y" => kde_clustered_x_y,
#     "kde_gauss_x" => kde_gauss_x,
    
#     # Y dimension
#     "kde_true_y_x" => kde_true_y_x,
#     "kde_true_y_y" => kde_true_y_y,
#     "kde_clustered_y_x" => kde_clustered_y_x, 
#     "kde_clustered_y_y" => kde_clustered_y_y,
#     "kde_gauss_y" => kde_gauss_y,
    
#     # Z dimension
#     "kde_true_z_x" => kde_true_z_x,
#     "kde_true_z_y" => kde_true_z_y,
#     "kde_clustered_z_x" => kde_clustered_z_x,
#     "kde_clustered_z_y" => kde_clustered_z_y,
#     "kde_gauss_z" => kde_gauss_z,
    
#     # Model parameters
#     "S" => S,
#     "M" => M,
    
#     # Requested variables
#     "centers" => centers,
#     "averages" => averages,
#     "Nc" => Nc,
    
#     # Time parameters
#     "dt" => dt,
#     "res_trj" => res_trj,
#     "n_tau" => n_tau
# )

# # Save all data using the existing function
# save_variables_to_hdf5("data/HOR_data/ENSO_responses_data.h5", data_to_save)

##

# Load the ENSO response data using the existing function
results = read_variables_from_hdf5("data/HOR_data/ENSO_responses_data.h5")

# Extract all variables from the results dictionary
R_gen_hack = results["R_gen_hack"]
R_lin = results["R_lin"]
R_num = results["R_num"]

# Model parameters
S = results["S"]
M = results["M"]

# PDF data for each dimension
# X dimension
kde_true_x_x = results["kde_true_x_x"]
kde_true_x_y = results["kde_true_x_y"]
kde_clustered_x_x = results["kde_clustered_x_x"]
kde_clustered_x_y = results["kde_clustered_x_y"]
kde_gauss_x = results["kde_gauss_x"]

# Y dimension
kde_true_y_x = results["kde_true_y_x"]
kde_true_y_y = results["kde_true_y_y"]
kde_clustered_y_x = results["kde_clustered_y_x"]
kde_clustered_y_y = results["kde_clustered_y_y"]
kde_gauss_y = results["kde_gauss_y"]

# Z dimension
kde_true_z_x = results["kde_true_z_x"]
kde_true_z_y = results["kde_true_z_y"]
kde_clustered_z_x = results["kde_clustered_z_x"]
kde_clustered_z_y = results["kde_clustered_z_y"]
kde_gauss_z = results["kde_gauss_z"]

# Requested variables
centers = results["centers"]
averages = results["averages"]
Nc = results["Nc"]

# Time parameters
dt = results["dt"]
res_trj = results["res_trj"]
n_tau = results["n_tau"]