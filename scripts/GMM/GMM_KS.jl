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
using FFTW
##

N = 64
L = 34
symm = false
dt = 0.1
T = 10005000
tstore = 1.

x,_ = domain(L, N)
u = sin.(2*pi*x/L) + 0.1*sin.(10*pi*x/L)

u = dealias(fft(u))
u[1] = 0 # Ensure zero mean
u, t = KSE_integrate(L, dt, T, tstore, u, symm, true)

u = u[:,5001:end]
t = t[5001:end]

# Original code for getting physical space representation
uu = zeros(length(t), N)
E = zeros(length(t))

for i = 1:length(t)
    u_spectral = vector2field(u[:,i], N, symm)
    u_physical = real(ifft(u_spectral))
    uu[i,:] = u_physical
    E[i] = sqrt(sum(u_physical.^2))
end

# # Save data to HDF5 file
# h5open("data/GMM_data/KS_simulation_data.h5", "w") do file
#     file["u"] = u
#     file["t"] = t
#     file["dt"] = dt
#     file["T"] = T
#     file["tstore"] = tstore
#     file["N"] = N
#     file["L"] = L
#     file["symm"] = symm
# end
# println("Simulation data saved to KS_simulation_data.h5")
##

# Read simulation data from HDF5 file
u, t, dt, T, tstore, N, L, symm = h5open("data/GMM_data/KS_simulation_data.h5", "r") do file
    u = read(file["u"])
    t = read(file["t"])
    dt = read(file["dt"])
    T = read(file["T"])
    tstore = read(file["tstore"])
    N = read(file["N"])
    L = read(file["L"])
    symm = read(file["symm"])
    
    # Return all values
    return u, t, dt, T, tstore, N, L, symm
end

println("Simulation data loaded from KS_simulation_data.h5")

dim = 6
# Or reduce to a specific number of modes
u_reduced, kept_modes, relative_energy = reduce_fourier_energy(u, dim)
Plots.plot(relative_energy)
cumulative_energy = cumsum(relative_energy)
Plots.plot(cumulative_energy)
println("The top $(dim) modes contain $(round(cumulative_energy*100, digits=2))% of the total energy")

# Convert reduced Fourier modes back to physical space
uu_reduced = reconstruct_physical_from_reduced(u_reduced, kept_modes, N, symm)

uu = zeros(20000, N)
E = zeros(20000)

for i = 1:20000
    u_spectral = vector2field(u[:,i], N, symm)
    u_physical = real(ifft(u_spectral))
    uu[i,:] = u_physical
    E[i] = sqrt(sum(u_physical.^2))
end

# Visualize the original vs reduced representation
fig = Figure(resolution=(500, 2000))
ax1 = Axis(fig[1, 1], title="Original", xlabel="x", ylabel="u")
ax2 = Axis(fig[2, 1], title="Reduced", xlabel="x", ylabel="u")
ax3 = Axis(fig[3, 1], title="Energy", xlabel="t", ylabel="E")
ax4 = Axis(fig[4, 1], title="Energy", xlabel="t", ylabel="E")

# Display a specific time snapshot
time_idx1, time_idx2, time_idx3, time_idx4 = 1, 5000, 15000, 20000
GLMakie.lines!(ax1, uu[time_idx1,:], color=:blue)
GLMakie.lines!(ax1, uu_reduced[time_idx1,:], color=:red)
GLMakie.lines!(ax2, uu[time_idx2,:], color=:blue)
GLMakie.lines!(ax2, uu_reduced[time_idx2,:], color=:red)
GLMakie.lines!(ax3, uu_reduced[time_idx3,:], color=:blue)
GLMakie.lines!(ax3, uu[time_idx3,:], color=:red)
GLMakie.lines!(ax4, uu[time_idx4,:], color=:blue)
GLMakie.lines!(ax4, uu_reduced[time_idx4,:], color=:red)

display(fig)

##

u_reduced_scaled = u_reduced .* relative_energy

plt1 = Plots.plot(u_reduced_scaled[1,1:1000])
plt2 = Plots.plot(u_reduced_scaled[2,1:1000])
plt3 = Plots.plot(u_reduced_scaled[3,1:1000])
plt4 = Plots.plot(u_reduced_scaled[4,1:1000])
plt5 = Plots.plot(u_reduced_scaled[5,1:1000])
plt6 = Plots.plot(u_reduced_scaled[6,1:1000])

Plots.plot(plt1, plt2, plt3, plt4, plt5, plt6,
    layout=(3, 2), size=(800, 800),
    xlabel="t", ylabel="u", title="Reduced Fourier Modes")
##

M = mean(u_reduced, dims=2)[:]
S = std(u_reduced, dims=2)[:]
obs = (u_reduced .- M) ./ S

# M = mean(u_reduced_scaled[1,:])
# S = std(u_reduced_scaled[1,:])
# obs = (u_reduced .- M) ./ S

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
Plots.scatter(obs_uncorr[1,1:1:1000], obs_uncorr[2,1:1:1000], obs_uncorr[3,1:1:1000], markersize=1, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

normalization = false
σ_value = 0.05

averages, _, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.0001, do_print=true, conv_param=0.05, normalization=normalization)

if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
end

gr()
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

trj_clustered = evolve(zeros(dim), 0.1*dt, 1000000, score_clustered_xt, sigma_I; timestepper=:rk4, resolution=10, boundary=[-5,5])
# trj_score = evolve([0.0, 0.0], dt, 1000000, score_true, sigma_I; timestepper=:rk4, resolution=10, boundary=[-100,100])

kde_clustered_1 = kde(trj_clustered[1,:])
kde_true_1 = kde(obs[1,:])

kde_clustered_2 = kde(trj_clustered[2,:])
kde_true_2 = kde(obs[2,:])

kde_clustered_3 = kde(trj_clustered[3,:])
kde_true_3 = kde(obs[3,:])

kde_clustered_4 = kde(trj_clustered[4,:])
kde_true_4 = kde(obs[4,:])

gr()
plt1 = Plots.plot(kde_clustered_1.x, kde_clustered_1.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
plt1 = Plots.plot!(kde_true_1.x, kde_true_1.density, label="True", xlabel="X", ylabel="Density", title="True PDF")

plt2 = Plots.plot(kde_clustered_2.x, kde_clustered_2.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
plt2 = Plots.plot!(kde_true_2.x, kde_true_2.density, label="True", xlabel="X", ylabel="Density", title="True PDF")

plt3 = Plots.plot(kde_clustered_3.x, kde_clustered_3.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
plt3 = Plots.plot!(kde_true_3.x, kde_true_3.density, label="True", xlabel="X", ylabel="Density", title="True PDF")

plt4 = Plots.plot(kde_clustered_4.x, kde_clustered_4.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
plt4 = Plots.plot!(kde_true_4.x, kde_true_4.density, label="True", xlabel="X", ylabel="Density", title="True PDF")

Plots.plot(plt1, plt2, plt3, plt4, layout=(2, 2), size=(800, 800))

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

plt1 = Plots.heatmap(kde_true_12.x, kde_true_12.y, kde_true_12.density, label="True", xlabel="X", ylabel="Y", title="True PDF")
plt2 = Plots.heatmap(kde_clustered_12.x, kde_clustered_12.y, kde_clustered_12.density, label="Observed", xlabel="X", ylabel="Y", title="Observed PDF")
plt3 = Plots.heatmap(kde_true_23.x, kde_true_23.y, kde_true_23.density, label="True", xlabel="X", ylabel="Y", title="True PDF")
plt4 = Plots.heatmap(kde_clustered_23.x, kde_clustered_23.y, kde_clustered_23.density, label="Observed", xlabel="X", ylabel="Y", title="Observed PDF")
plt5 = Plots.heatmap(kde_true_34.x, kde_true_34.y, kde_true_34.density, label="True", xlabel="X", ylabel="Y", title="True PDF")
plt6 = Plots.heatmap(kde_clustered_34.x, kde_clustered_34.y, kde_clustered_34.density, label="Observed", xlabel="X", ylabel="Y", title="Observed PDF")
plt7 = Plots.heatmap(kde_true_41.x, kde_true_41.y, kde_true_41.density, label="True", xlabel="X", ylabel="Y", title="True PDF")
plt8 = Plots.heatmap(kde_clustered_41.x, kde_clustered_41.y, kde_clustered_41.density, label="Observed", xlabel="X", ylabel="Y", title="Observed PDF")
plt9 = Plots.heatmap(kde_true_13.x, kde_true_13.y, kde_true_13.density, label="True", xlabel="X", ylabel="Y", title="True PDF")
plt10 = Plots.heatmap(kde_clustered_13.x, kde_clustered_13.y, kde_clustered_13.density, label="Observed", xlabel="X", ylabel="Y", title="Observed PDF")
plt11 = Plots.heatmap(kde_true_24.x, kde_true_24.y, kde_true_24.density, label="True", xlabel="X", ylabel="Y", title="True PDF")
plt12 = Plots.heatmap(kde_clustered_24.x, kde_clustered_24.y, kde_clustered_24.density, label="Observed", xlabel="X", ylabel="Y", title="Observed PDF")
Plots.plot(plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8, plt9, plt10, plt11, plt12,
    layout=(3, 4), size=(1200, 800))
