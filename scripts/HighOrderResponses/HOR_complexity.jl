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


# x = [u, hW, TC, TE, τ, I] u, hW, TC, TE, tau, I = x
# Parameter definitions (from Table 1 in Chen et al., 2022)
r    = 0.25
α1   = 0.0625
α2   = 0.125
b0   = 2.5
μ    = 0.5
γ    = 0.75
# Coupling with τ (wind bursts); set to zero here for simplicity.
β_u  = 0.0
β_h  = 0.0
β_C  = 0.0
β_E  = 0.0

# Precompute common factor: b0 * μ^2 = 2.5 * (0.5)^2 = 0.625
b0μ2 = b0 * μ^2

# Nonlinear damping for TC (Eq. (15)):
c1 = 25.0 * TC + 42.1875 * TC^2 + 0.9

# Decadal dynamics; m is the mean of I (here taken as 2), λ = 2/60
m = 2.0
λ = 2.0 / 60.0

function F(x, t) 
    # Drift components (deterministic part)
    du_dt    = - r * u - α1 * b0μ2 * (TC + TE) + β_u * tau
    dhW_dt   = - r * hW - α2 * b0μ2 * (TC + TE) + β_h * tau
    dTC_dt   = γ * b0μ2 * ( - c1 + TE ) + γ * hW + β_C * tau
    # For TE, using c2 = 1/4
    c2       = 1/4
    dTE_dt   = γ * hW - 3 * γ * b0μ2 * c2 * TE + β_E * tau
    dtau_dt  = - 2.0 * tau
    dI_dt    = - λ * (I - m)
    return [du_dt, dhW_dt, dTC_dt, dTE_dt, dtau_dt, dI_dt]
end

function sigma(x, t) # x = [u, hW, TC, TE, τ, I] u, hW, TC, TE, tau, I = x
    # Noise amplitude parameters (from Table 1)
    σu = 0.04
    σh = 0.02
    σC = 0.04
    σE = 0.0
    # Intraseasonal noise coefficient (state-dependent)
    στ = 0.9 * tanh(7.5 * TC) + 1.0
    # Decadal noise amplitude (assumed constant here)
    σI = 0.1
    
    return [σu, σh, σC, σE, στ, σI]
end


dim = 3
dt = 0.01
Nsteps = 100000000
obs_nn = evolve([0.0, 0.0, 0.0], dt, Nsteps, F, sigma; resolution = 10)
obs = obs_nn #(obs_nn .- mean(obs_nn, dims=2)) ./ std(obs_nn, dims=2)

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

averages, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.001, do_print=true, conv_param=0.002, normalization=normalization)

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

invC0 = inv(cov(obs'))
score_qG(x) = - invC0*x

score_clustered_xt(x,t) = score_clustered(x)
score_qG_xt(x,t) = score_qG(x)
sigma_I(x,t) = 1.0

trj_clustered = evolve([0.0, 0.0, 0.0], 0.2*dt, 2000000, score_clustered_xt, sigma_I; timestepper=:rk4, resolution=5, boundary=[-15,15])
trj_qG = evolve([0.0, 0.0, 0.0], 0.5*dt, 500000, score_qG_xt, sigma_I; timestepper=:rk4, resolution=2, boundary=[-15,15])

kde_clustered_x = kde(trj_clustered[1,:])
kde_true_x = kde(obs[1,:])
kde_qG_x = kde(trj_qG[1,:])

kde_clustered_y = kde(trj_clustered[2,:])
kde_true_y = kde(obs[2,:])
kde_qG_y = kde(trj_qG[2,:])

kde_clustered_z = kde(trj_clustered[3,:])
kde_true_z = kde(obs[3,:])
kde_qG_z = kde(trj_qG[3,:])

plt1 = Plots.plot(kde_clustered_x.x, kde_clustered_x.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
plt1 = Plots.plot!(kde_true_x.x, kde_true_x.density, label="True", xlabel="X", ylabel="Density", title="True PDF")
plt1 = Plots.plot!(kde_qG_x.x, kde_qG_x.density, label="qG", xlabel="X", ylabel="Density", title="qG PDF")

plt2 = Plots.plot(kde_clustered_y.x, kde_clustered_y.density, label="Observed", xlabel="Y", ylabel="Density", title="Observed PDF")
plt2 = Plots.plot!(kde_true_y.x, kde_true_y.density, label="True", xlabel="Y", ylabel="Density", title="True PDF")
plt2 = Plots.plot!(kde_qG_y.x, kde_qG_y.density, label="qG", xlabel="Y", ylabel="Density", title="qG PDF")

plt3 = Plots.plot(kde_clustered_z.x, kde_clustered_z.density, label="Observed", xlabel="Z", ylabel="Density", title="Observed PDF")
plt3 = Plots.plot!(kde_true_z.x, kde_true_z.density, label="True", xlabel="Z", ylabel="Density", title="True PDF")
plt3 = Plots.plot!(kde_qG_z.x, kde_qG_z.density, label="qG", xlabel="Z", ylabel="Density", title="qG PDF")

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

res_trj = 1
steps_trj = 100000
trj = obs[:,1:res_trj:steps_trj*res_trj]

ϵ = 1.0
u(x) = [0.0, 0.0, -0.4*x[3]]
div_u(x) = -0.4

# ϵ = 0.1
# u(x) = [ϵ, 0.0, 0.0]
# div_u(x) = -0.0
m = mean(obs, dims=2)

F_pert(x,t) = F(x,t) + u(x) * f(t)

score_gen(x) = score_clustered(x)
score_test(x,t) = score_clustered(x) # (F(x,t) .- [0.0, 0.0, 4.5*(tanh(x[3]) + 1)*sech(x[3])^2])./ sigma(x,t).^2 

dim_Obs = 3
n_tau = 100

R_num, δObs_num = zeros(4, dim, n_tau+1), zeros(4, dim, n_tau+1)
R_lin, δObs_lin = zeros(4, dim, n_tau+1), zeros(4, dim, n_tau+1)
R_gen, δObs_gen = zeros(4, dim, n_tau+1), zeros(4, dim, n_tau+1)
#R_test_score, δObs_test_score = zeros(4, dim, n_tau+1), zeros(4, dim, n_tau+1)

for i in 1:4
    Obs(x) = (x .- m).^i
    R_num[i,:,:] = generate_numerical_response(F, u, dim, dt, n_tau, 1000, sigma, Obs, dim_Obs; n_ens=10000, resolution=10*res_trj)
    R_lin[i,:,:], δObs_lin[i,:,:] = generate_score_response(trj, u, div_u, f, score_qG, res_trj*dt, n_tau, Obs, dim_Obs)
    R_gen[i,:,:], δObs_gen[i,:,:] = generate_score_response(trj, u, div_u, f, score_gen, res_trj*dt, n_tau, Obs, dim_Obs)
    # R_test_score[i,:,:] = generate_numerical_response3(score_test, u, dim, 0.2*dt, n_tau, 1000, sigma_I, Obs, dim_Obs; n_ens=1000, resolution=5*10*res_trj)
end

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
        R_gen_hack[i,:,j] = 0.5 .* (R_gen[i,:,j]'*invR0_gen .+ (invR0_gen*R_gen[i,:,j])')
    end
end

##

using GLMakie
r1_min, r1_max = -5, 7.5
r2_min, r2_max = -2.5, 7.5
r3_min, r3_max = -5, 5

# Create figure with 3x3 layout for PDFs and heatmaps
fig = Figure(resolution=(1600, 1200), font="CMU Serif")

# Define common elements
colors = [:black, :red, :blue]
labels = ["Numerical", "Linear", "Generative"]

# Create the 1D PDF plots (first row)
ax1 = Axis(fig[1,1], 
    xlabel="u₁", ylabel="Probability density",
    title="PDF u₁",
    titlesize=32, xlabelsize=28, ylabelsize=28)

ax2 = Axis(fig[1,2], 
    xlabel="u₂", ylabel="Probability density",
    title="PDF u₂",
    titlesize=32, xlabelsize=28, ylabelsize=28)

ax3 = Axis(fig[1,3], 
    xlabel="τ", ylabel="Probability density",
    title="PDF τ",
    titlesize=32, xlabelsize=28, ylabelsize=28)

# Plot 1D PDFs
lines!(ax1, kde_clustered_x.x, kde_clustered_x.density, color=colors[3], linewidth=2)
lines!(ax1, kde_true_x.x, kde_true_x.density, color=colors[1], linewidth=2)
lines!(ax1, kde_qG_x.x, kde_qG_x.density, color=colors[2], linewidth=2)

lines!(ax2, kde_clustered_y.x, kde_clustered_y.density, color=colors[3], linewidth=2)
lines!(ax2, kde_true_y.x, kde_true_y.density, color=colors[1], linewidth=2)
lines!(ax2, kde_qG_y.x, kde_qG_y.density, color=colors[2], linewidth=2)

lines!(ax3, kde_clustered_z.x, kde_clustered_z.density, color=colors[3], linewidth=2)
lines!(ax3, kde_true_z.x, kde_true_z.density, color=colors[1], linewidth=2)
lines!(ax3, kde_qG_z.x, kde_qG_z.density, color=colors[2], linewidth=2)

# Create heatmap plots (second and third rows)
# Observed PDFs
ax4 = Axis(fig[2,1], xlabel="u₁", ylabel="u₂", title="Observed PDF u₁-u₂",
    titlesize=32, xlabelsize=28, ylabelsize=28,
    limits=(r1_min, r1_max, r2_min, r2_max))

ax5 = Axis(fig[2,2], xlabel="u₁", ylabel="τ", title="Observed PDF u₁-τ",
    titlesize=32, xlabelsize=28, ylabelsize=28,
    limits=(r1_min, r1_max, r3_min, r3_max))

ax6 = Axis(fig[2,3], xlabel="u₂", ylabel="τ", title="Observed PDF u₂-τ",
    titlesize=32, xlabelsize=28, ylabelsize=28,
    limits=(r2_min, r2_max, r3_min, r3_max))

# Sampled PDFs
ax7 = Axis(fig[3,1], xlabel="u₁", ylabel="u₂", title="Sampled PDF u₁-u₂",
    titlesize=32, xlabelsize=28, ylabelsize=28,
    limits=(r1_min, r1_max, r2_min, r2_max))

ax8 = Axis(fig[3,2], xlabel="u₁", ylabel="τ", title="Sampled PDF u₁-τ",
    titlesize=32, xlabelsize=28, ylabelsize=28,
    limits=(r1_min, r1_max, r3_min, r3_max))

ax9 = Axis(fig[3,3], xlabel="u₂", ylabel="τ", title="Sampled PDF u₂-τ",
    titlesize=32, xlabelsize=28, ylabelsize=28,
    limits=(r2_min, r2_max, r3_min, r3_max))

# Custom colormap using the same colors
cmap = cgrad([:blue, :black, :red])

# Plot heatmaps
GLMakie.heatmap!(ax4, kde_obs_xy.x, kde_obs_xy.y, kde_obs_xy.density, colormap=cmap)
GLMakie.heatmap!(ax5, kde_obs_xz.x, kde_obs_xz.y, kde_obs_xz.density, colormap=cmap)
GLMakie.heatmap!(ax6, kde_obs_yz.x, kde_obs_yz.y, kde_obs_yz.density, colormap=cmap)
GLMakie.heatmap!(ax7, kde_clustered_xy.x, kde_clustered_xy.y, kde_clustered_xy.density, colormap=cmap)
GLMakie.heatmap!(ax8, kde_clustered_xz.x, kde_clustered_xz.y, kde_clustered_xz.density, colormap=cmap)
GLMakie.heatmap!(ax9, kde_clustered_yz.x, kde_clustered_yz.y, kde_clustered_yz.density, colormap=cmap)

# Add legend
Legend(fig[4, :],
    [LineElement(color=c, linewidth=2) for c in colors],
    labels,
    "Methods",
    orientation = :horizontal,
    titlesize = 28,
    labelsize = 24
)

# Adjust spacing
colgap!(fig.layout, 20)
rowgap!(fig.layout, 20)

# Save figure
# save("figures/pdfs_ENSO.png", fig)

fig

##


# Create figure with 3x4 layout
fig = Figure(resolution=(1600, 1200), font="CMU Serif")

# Define common elements
colors = [:black, :red, :blue]
labels = ["Numerical", "Linear", "Generative"]
time_axis = 0:dt:n_tau*dt

# Create axes array
axes = Matrix{Axis}(undef, 3, 4)

# Column titles
titles = ["Response 1st moment", "Response 2nd moment", 
          "Response 3rd moment", "Response 4th moment"]
# Row labels
ylabels = ["Response u₁", "Response u₂", "Response τ"]

# Create subplots
for i in 1:3, j in 1:4
    axes[i,j] = Axis(fig[i,j],
        xlabel = (i == 3) ? "Time lag" : "",
        ylabel = (j == 1) ? ylabels[i] : "",
        title = (i == 1) ? titles[j] : "",
        titlesize = 32,
        xlabelsize = 28,
        ylabelsize = 28
    )

    # Plot data
    lines!(axes[i,j], time_axis, R_num[j,i,:]./ϵ, color=colors[1], linewidth=2)
    lines!(axes[i,j], time_axis, R_lin[j,i,:]./ϵ, color=colors[2], linewidth=2)
    lines!(axes[i,j], time_axis, R_gen_hack[j,i,:]./ϵ, color=colors[3], linewidth=2)
end

# Add legend
Legend(fig[4, :],
    [LineElement(color=c, linewidth=2) for c in colors],
    labels,
    "Methods",
    orientation = :horizontal,
    titlesize = 28,
    labelsize = 24
)

# Adjust spacing
colgap!(fig.layout, 20)
rowgap!(fig.layout, 20)

# # Save figure
# save("figures/responses2_ENSO_hack4.png", fig)

fig

##

