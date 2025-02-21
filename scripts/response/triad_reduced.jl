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
using Random
using QuadGK
using GLMakie

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
obs = obs_nn #(obs_nn .- mean(obs_nn, dims=2)) ./ std(obs_nn, dims=2)

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
obs_uncorr = obs[:, 10:10:end]

Plots.scatter(obs_uncorr[1,1:10000], markersize=2, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

normalization = false
σ_value = 0.01

averages, averages_residual, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.002, do_print=true, conv_param=0.001, normalization=normalization)

if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
    inputs_targets_residual, M_averages_values_residual, m_averages_values_residual = generate_inputs_targets(averages_residual, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
    inputs_targets_residual = generate_inputs_targets(averages_residual, centers, Nc; normalization=false)
end

centers_sorted_indices = sortperm(centers[1,:])
centers_sorted = centers[:,centers_sorted_indices][:]
scores = .- averages[:,centers_sorted_indices][:] ./ σ_value
scores_residual = (averages_residual[:,centers_sorted_indices][:] .- centers_sorted)./ σ_value^2
scores_true = [-score_true(centers_sorted[i])[1] for i in eachindex(centers_sorted)]

Plots.plot(centers_sorted, scores, color=:blue)
Plots.plot!(centers_sorted, scores_residual, color=:green)
Plots.plot!(centers_sorted, scores_true, color=:red)
##
#################### TRAINING WITH CLUSTERING LOSS ####################

@time nn_clustered, loss_clustered = train(inputs_targets, 1000, 32, [dim, 128, 64, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value

# @time nn_clustered_residual, loss_clustered_residual = train(inputs_targets_residual, 1000, 32, [dim, 128, 64, dim]; use_gpu=true, activation=swish, last_activation=identity)
# if normalization == true
#     nn_clustered_residual_cpu  = Chain(nn_clustered_residual, x -> x .* (M_averages_values_residual .- m_averages_values_residual) .+ m_averages_values_residual) |> cpu
# else
#     nn_clustered_residual_cpu = nn_clustered_residual |> cpu
# end
# score_clustered_residual(x) = (nn_clustered_residual_cpu(Float32.([x...]))[:] .- x) ./ σ_value^2

cluster_loss = check_loss(obs_uncorr, nn_clustered_cpu, σ_value)
# cluster_loss_residual = check_loss(obs_uncorr, nn_clustered_residual_cpu, σ_value)

println(cluster_loss)
# println(cluster_loss_residual)

Plots.plot(loss_clustered)
# Plots.plot!(loss_clustered_residual)

##
########### Additional training on the full data (not used in the draft) ############

@time nn_clustered, loss_clustered = train(obs_uncorr, 20, 128, nn_clustered, σ_value; use_gpu=true)

##
#################### TRAINING WITH VANILLA LOSS ####################

@time nn_vanilla, loss_vanilla = train(obs_uncorr, 200, 128, [dim, 128, 64, dim], σ_value; use_gpu=true, opt=Adam(0.001))
nn_vanilla_cpu = nn_vanilla |> cpu
score_vanilla(x) = .- nn_vanilla_cpu(Float32.([x...])) ./ σ_value
Plots.plot(loss_vanilla)
hline!([cluster_loss])

##
#################### VECTOR FIELDS ####################

invC0 = inv(cov(obs'))
score_qG(x) = - invC0*x

xax = [centers_sorted[1]:0.005:centers_sorted[end]...]

s_true = [-score_true(xax[i])[1] for i in eachindex(xax)]
s_gen = [score_clustered(xax[i])[1] for i in eachindex(xax)]
s_qG = [score_qG(xax[i])[1] for i in eachindex(xax)]
# s_gen_residual = [score_clustered(xax[i])[1] for i in eachindex(xax)]

Plots.plot(xax, s_true, label="True", xlabel="X", ylabel="Force", title="Forces")
Plots.plot!(xax, s_gen, label="Learned")
Plots.plot!(xax, s_qG, label="qG")
# Plots.plot!(xax, s_gen_residual, label="Learned Residual")

##
#################### SAMPLES GENERATION ####################

score_clustered_xt(x,t) = score_clustered(x)
score_true_xt(x,t) = - score_true(x)
score_qG_xt(x,t) = score_qG(x)
sigma_I(x,t) = 1.0

trj_clustered = evolve([0.0], 0.1*dt, 500000, score_clustered_xt, sigma_I; timestepper=:rk4, resolution=10)
trj_true = evolve([0.0], 0.1*dt, 500000, score_true_xt, sigma_I; timestepper=:rk4, resolution=10)
trj_qG = evolve([0.0], 0.1*dt, 500000, score_qG_xt, sigma_I; timestepper=:rk4, resolution=10)

kde_clustered = kde(trj_clustered[:])
kde_true = kde(trj_true[:])
kde_qG = kde(trj_qG[:])

Plots.plot(kde_clustered.x, kde_clustered.density, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
Plots.plot!(kde_true.x, kde_true.density, label="True", xlabel="X", ylabel="Density", title="True PDF")
Plots.plot!(kde_qG.x, kde_qG.density, label="qG", xlabel="X", ylabel="Density", title="qG PDF")


##
#################### NN SAVINGS ####################

BSON.@save pwd() * "/NNs/nn_clustered_D$(dim)_$(normalization)_$(σ_value).bson" nn_clustered_cpu
BSON.@save pwd() * "/NNs/nn_vanilla_D$(dim)_$(normalization)_$(σ_value).bson" nn_vanilla_cpu

##
#################### NN LOADINGS ####################

BSON.load(pwd() * "/NNs/nn_clustered_D$(dim)_$(normalization)_$(σ_value).bson")[:nn_clustered_cpu]
BSON.load(pwd() * "/NNs/nn_vanilla_D$(dim)_$(normalization)_$(σ_value).bson")[:nn_vanilla_cpu]

##



#f(t; a=0.1) = (0 ≤ t ≤ a) ? 10 : 0

# function f(t; a=1/3)
#     t_max = 1 / a  # Time at which f(t) reaches 1
#     t_end = 2 * t_max  # Time at which f(t) returns to 0

#     if 0 ≤ t ≤ t_max
#         return a * t
#     elseif t_max < t ≤ t_end
#         return 1 - a * (t - t_max)
#     else
#         return 0  # f(t) is defined only in [0, 2/a], extend if needed
#     end
# end


using ClustGen
f(t) = 1.0

res_trj = 10
steps_trj = 10000
trj = obs[:,1:res_trj:steps_trj*res_trj]

ϵ = 0.05
u(x) = [ϵ]
div_u(x) = 0.0
m = mean(obs, dims=2)

F_pert(x,t) = F(x,t) + u(x) * f(t)

score_gen(x) = score_clustered(x)

n_tau = 100

R_true, δObs_true = zeros(4, n_tau+1), zeros(4, n_tau+1)
R_lin, δObs_lin = zeros(4, n_tau+1), zeros(4, n_tau+1)
R_gen, δObs_gen = zeros(4, n_tau+1), zeros(4, n_tau+1)

for i in 1:4
    Obs(x) = (x[1:1,:] .- m).^i
    R_true[i,:], δObs_true[i,:] = generate_score_response(trj, u, div_u, f, score_true, res_trj*dt, n_tau, Obs, 1)
    R_lin[i,:], δObs_lin[i,:] = generate_score_response(trj, u, div_u, f, score_qG, res_trj*dt, n_tau, Obs, 1)
    R_gen[i,:], δObs_gen[i,:] = generate_score_response(trj, u, div_u, f, score_gen, res_trj*dt, n_tau, Obs, 1)
end

##
fig = Figure(resolution=(1200, 1000), font="CMU Serif")

# Define common elements
colors = [:black, :red, :blue]
labels = ["True", "Linear", "Generative"]
time_axis = 0.0:dt*res_trj:n_tau*dt*res_trj

# Create subplots
ax1 = Axis(fig[1,1], 
    xlabel="State", ylabel="Score function",
    title="Scores",
    titlesize=20,
    xlabelsize=16, ylabelsize=16)

ax2 = Axis(fig[1,2], 
    xlabel="State", ylabel="Probability density",
    title="PDFs",
    titlesize=20,
    xlabelsize=16, ylabelsize=16)

ax3 = Axis(fig[2,1], 
    xlabel="Time lag", ylabel="Response",
    title="Response 1st moment",
    titlesize=20,
    xlabelsize=16, ylabelsize=16)

ax4 = Axis(fig[2,2], 
    xlabel="Time lag", ylabel="Response",
    title="Response 2nd moment",
    titlesize=20,
    xlabelsize=16, ylabelsize=16)

ax5 = Axis(fig[3,1], 
    xlabel="Time lag", ylabel="Response",
    title="Response 3rd moment",
    titlesize=20,
    xlabelsize=16, ylabelsize=16)

ax6 = Axis(fig[3,2], 
    xlabel="Time lag", ylabel="Response",
    title="Response 4th moment",
    titlesize=20,
    xlabelsize=16, ylabelsize=16)

# Plot data
lines!(ax1, xax, s_true, color=colors[1], linewidth=2)
lines!(ax1, xax, s_qG, color=colors[2], linewidth=2)
lines!(ax1, xax, s_gen, color=colors[3], linewidth=2)

lines!(ax2, kde_clustered.x, kde_clustered.density, color=colors[1], linewidth=2)
lines!(ax2, kde_qG.x, kde_qG.density, color=colors[2], linewidth=2)
lines!(ax2, kde_true.x, kde_true.density, color=colors[3], linewidth=2)

lines!(ax3, time_axis, R_true[1,:]./ϵ, color=colors[1], linewidth=2)
lines!(ax3, time_axis, R_lin[1,:]./ϵ, color=colors[2], linewidth=2)
lines!(ax3, time_axis, R_gen[1,:]./ϵ, color=colors[3], linewidth=2)

lines!(ax4, time_axis, R_true[2,:]./ϵ, color=colors[1], linewidth=2)
lines!(ax4, time_axis, R_lin[2,:]./ϵ, color=colors[2], linewidth=2)
lines!(ax4, time_axis, R_gen[2,:]./ϵ, color=colors[3], linewidth=2)

lines!(ax5, time_axis, R_true[3,:]./ϵ, color=colors[1], linewidth=2)
lines!(ax5, time_axis, R_lin[3,:]./ϵ, color=colors[2], linewidth=2)
lines!(ax5, time_axis, R_gen[3,:]./ϵ, color=colors[3], linewidth=2)

lines!(ax6, time_axis, R_true[4,:]./ϵ, color=colors[1], linewidth=2)
lines!(ax6, time_axis, R_lin[4,:]./ϵ, color=colors[2], linewidth=2)
lines!(ax6, time_axis, R_gen[4,:]./ϵ, color=colors[3], linewidth=2)

# Add legend
Legend(fig[4, :], 
    [LineElement(color=c, linewidth=2) for c in colors],
    labels,
    "Methods",
    orientation=:horizontal,
    titlesize=16,
    labelsize=14)

# Adjust spacing
fig[1:3,1:2] = GridLayout() 
colgap!(fig.layout, 20)
rowgap!(fig.layout, 20)

fig

save("figures/response_reduced.png", fig)