include(pwd() * "/src/ClustGen.jl")

using Main.ClustGen: f_tilde, train_clustered, train_vanilla, check_loss, sample
using KernelDensity
using HDF5
using Flux, Random, KernelDensity
using Plots, LinearAlgebra
using Statistics
using ProgressBars, QuadGK, ParallelKMeans,Distances, Interpolations

##################### DATA GENERATION #####################

function evolve(x0, system, timesteps, Δt, res, σ)
    x_f = Array{Float64,2}(undef, 1, timesteps*res)
    x_f[:, 1] = x0
    for i = ProgressBar(2:timesteps*res)
        x_old = x_f[:, i-1]
        k = system(x_old)
        x_f[:, i] = x_old + Δt*k + σ * randn(1) * sqrt(Δt)
    end
    return x_f[:, 1:res:end]
end

forcing(x) = @. x * (1 - x^2)

evolve(x0, forcing, timesteps, σ; Δt=0.01, res=100) = evolve(x0, forcing, timesteps, Δt, res, σ)

b = 0.5
b2half = b^2/2
x0 = [1.0]
sol = evolve(x0, forcing, 1000000, b) # It works well also with 100000 timesteps
U(x) = - x^2/2 + x^4/4
nrm, _ = quadgk(x -> exp(-U(x)/b2half), -10, 10)
P(x) = exp(-U.(x)/b2half) / nrm
kde_sol = kde(sol[1,:])
plt = plot(kde_sol.x, kde_sol.density, color=:blue, label="Observed")
plt = plot!(kde_sol.x, P.(kde_sol.x), color=:red, label="True")
display(plt)
##
######################### DATA NORMALIZATION #########################
function normalize_to_01(series)
    min_val = minimum(series, dims=2)
    max_val = maximum(series, dims=2)
    return (series .- min_val) ./ (max_val - min_val)
end

obs = normalize_to_01(sol)

kde_obs = kde(obs[1,:])
interp_P = interpolate((kde_obs.x,), kde_obs.density, Gridded(Linear()))
extrp_P = extrapolate(interp_P, Flat())
plt = plot(-0.2:0.01:1.2, extrp_P(-0.2:0.01:1.2), color=:red, label="True")
##
##################### CLUSTERING #####################

G(x,mu,sigma) = 1/sqrt(2*π*sigma^2) * exp(-(x-mu)^2/2/sigma^2)

function f_true(z,sigma; tol=1e-8)
    ext = sigma*6
    num, _ = quadgk(x -> x * G(x,0.0,sigma) * extrp_P(z - x), - ext, ext, atol=tol, rtol=tol)
    den, _ = quadgk(x -> G(x,0.0,sigma) * extrp_P(z - x), - ext, ext, atol=tol, rtol=tol)
    return - num / den / sigma
end

variance_exploding(t; σ_min = 0.01, σ_max = 5.0) = @. σ_min * (σ_max/σ_min)^t
g(t; σ_min=0.01, σ_max=5.0) = σ_min * (σ_max/σ_min)^t * sqrt(2*log(σ_max/σ_min))

n_diffs = 20
diff_times = [i/n_diffs for i in 1:n_diffs]
σ_values = variance_exploding.(diff_times)
μ = repeat(obs, 1, 1)

averages_values, centers_values, Nc_values = f_tilde(σ_values, μ; prob=0.05, do_plot=true, conv_param=0.00001)

##
####################### CLUSTERING CHECK #######################
index = 2
Ndim, Nz = size(μ)
z = randn!(similar(μ))
x = @. μ + σ_values[index] * z

xax = [0:0.05:1...]
yax =  f_true.(xax, σ_values[index])

averages_true = [f_true.(centers_values[index][i], σ_values[index]) for i in 1:Nc_values[index]]

rmse = 0.0
for i in 1:n_diffs
    rmse += mean(abs2, f_true.(centers_values[i], σ_values[i]) .+ averages_values[i])
end
rmse /= n_diffs
println("RMSE: ", rmse)

scatter(centers_values[index][:], averages_true, color=:red, label="True")
scatter!(centers_values[index][:], .- averages_values[index][:], color=:blue, label="Observed")
##

scatter(x[1,1:10000], .- z[1,1:10000], color=:blue, legend=false, markerstrokewidth=0,markersize=0.5)
plot!(xax, yax, color=:black, linewidth=3)
scatter!(centers_values[index,:], .- averages_values[index,:], color=:red, label="Observed", markerstrokewidth=0,markersize=5)
##
####################### TRAINING WITH CLUSTERING LOSS #######################

Dim = size(μ)[1]
M_averages_values = maximum(hcat(averages_values...))
m_averages_values = minimum(hcat(averages_values...))
averages_values_norm = []
for t in 1:n_diffs
    push!(averages_values_norm, (averages_values[t] .- m_averages_values) ./ (M_averages_values - m_averages_values))
end
data_clustered = []
for t in 1:n_diffs
    data_t = Flux.Float32.(hcat([[centers_values[t][:,i]..., diff_times[t], averages_values_norm[t][:,i]...] for i in 1:Nc_values[t]]...))
    push!(data_clustered, [(data_t[1:Dim+1, i], data_t[Dim+2:2*Dim+1, i]) for i in 1:Nc_values[t]])
end
data_clustered = vcat(data_clustered...)

nn_clustered, loss_clustered = train_clustered(data_clustered, 2000, 16, [Dim+1, 64, 32, Dim]; activation=tanh, opt=ADAM(0.001))
nnc(x, t) = .- nn_clustered(Flux.Float32.([x..., t]))[:] .* (M_averages_values - m_averages_values) .- m_averages_values
Plots.plot(loss_clustered)
Plots.hline!([rmse])
##
####################### TRAINING WITH VANILLA LOSS #######################

nn_vanilla, loss_vanilla = train_vanilla(obs[:,1:10:end], 1, 1000, [Dim+1, 64, 32, Dim], variance_exploding; activation=tanh)
cluster_loss = check_loss(obs, nnc, variance_exploding)
nnv(x, t) = nn_vanilla(Flux.Float32.([x..., t]))[:]
Plots.plot(loss_vanilla)
Plots.hline!([cluster_loss])
##
####################### SCORES COMPARISON #######################

index = 13
tt = diff_times[index]

xax = [minimum(centers_values[index]):0.01:maximum(centers_values[index])...]

yax =  f_true.(xax, variance_exploding(tt))
y_nn1 = [nnc(x, tt)[1] for x in xax]
y_nn2 = [nn_vanilla([x, diff_times[index]])[1] for x in xax]

println("Error clustering: ", mean(abs2,yax .- y_nn1))
println("Error vanilla: ", mean(abs2,yax .- y_nn2))

plot(xax, yax, color=:blue)
plot!(xax, y_nn1, color=:red)
scatter!(centers_values[index], .- averages_values[index], markersize=3, legend=false, xlabel="x", ylabel="f(x)", markerstrokewidth=0, color=:red)
plot!(xax, y_nn2, color=:green)
##
######################## SAMPLES GENERATION #########################

n_diffs_sampling = 100
n_ens = 10000

ens_clustered = sample(Dim, nnc, n_ens, n_diffs_sampling, variance_exploding, g)
ens_vanilla = sample(Dim, nnv, n_ens, n_diffs_sampling, variance_exploding, g)
##

kde_clustered = kde(ens_clustered[1,:])
kde_vanilla = kde(ens_vanilla[1,:])

plot(kde_clustered.x, kde_clustered.density, color=:red, label="Clustered", xlims=(-0.5,1.5))
plot!(kde_vanilla.x, kde_vanilla.density, color=:green, label="Vanilla", xlims=(-0.5,1.5))
plot!(kde_obs.x, kde_obs.density, color=:blue, label="Observed", xlims=(-0.5,1.5))
##
# ######################### LARGE DEVIATIONS #########################

# function evolve_f(x0, n_diffs, g, diff_stop)
#     dt = 1.0 / n_diffs
#     x_f = Array{Float64,2}(undef, 1, diff_stop)
#     x_f[:, 1] = x0
#     for i = 2:diff_stop
#         t_diff = i * dt
#         x_old = x_f[:, i-1]
#         x_f[:, i] = x_old + g(t_diff) * randn(1) * sqrt(dt)
#     end
#     return x_f
# end

# function evolve_r(x1, n_diffs, g, σ, nn, diff_stop)
#     dt = 1.0 / n_diffs
#     x_r = Array{Float64,2}(undef, 1, diff_stop)
#     x_r[:, 1] = x1
#     for i = 2:diff_stop
#         t_diff = (diff_stop - i + 1) * dt
#         x_old = x_r[:, i-1]
#         s = σ(t_diff)
#         score = nn(x_old, t_diff) ./ s
#         x_r[:, i] = x_old + score * g(t_diff)^2 * dt + g(t_diff) * randn(1) * sqrt(dt)
#     end
#     return x_r[:, 1:end]
# end

# n_diffs = 1000
# n_ens = 10000
# diff_stop = 1000
# ens_r = zeros(n_ens, diff_stop)
# #ens_r_vanilla = zeros(n_ens, n_diffs)
# for i in ProgressBar(1:n_ens)
#     start_r = [variance_exploding(diff_stop/n_diffs)*randn()]
#     ens_r[i,:] = evolve_r(start_r, n_diffs, g, variance_exploding, nnc, diff_stop)
#     #ens_r_vanilla[i,:] = evolve_r([variance_exploding(1)*randn()], n_diffs, g, variance_exploding, nnv, n_diffs)
# end
# ##

# function get_closest_trajectories_indices_and_mean(ens_r::Array{Float64,2}, n::Int, target_value::Float64)
#     differences = abs.(ens_r[:, end] .- target_value)
#     closest_positions_indices = sortperm(differences)[1:n]
#     closest_trajectories = ens_r[closest_positions_indices, :]
#     mean_trajectory = mean(closest_trajectories, dims=1)
#     return closest_positions_indices, mean_trajectory
# end

# target_value = 0.
# n = 100
# forward = evolve_f([target_value], n_diffs, g, diff_stop)
# cpi, mt = get_closest_trajectories_indices_and_mean(ens_r, n, target_value)
# #cpi_vanilla, mt_vanilla = get_closest_trajectories_indices_and_mean(ens_r_vanilla, n, target_value)

# t_f = [i/n_diffs for i in 1:diff_stop]
# t_r = [(diff_stop +1 - i)/n_diffs for i in 1:diff_stop]

# plt = plot(xlims=(0,0.8), ylims=(-1.5,1.5))
# for i in cpi
#     plot!(plt, t_r, ens_r[i,:], color=:red, alpha=0.1, legend=false)
# #    plot!(plt, t_r, ens_r_vanilla[i,:], color=:green, alpha=0.1, legend=false)
# end
# plot!(plt, t_r, mt[1,:], color=:red, alpha=1.0, legend=false)
# #plot!(plt, t_r, mt_vanilla[1,:], color=:green, alpha=1.0, legend=false)
# plot!(plt, t_f, forward[1,:], color=:blue, legend=false)
# display(plt)
# ##
# target_value = 0.6
# diff_stop = 200
# reverse = evolve_r([target_value], n_diffs, g, variance_exploding, nnc, diff_stop)
# reverse_vanilla = evolve_r([target_value], n_diffs, g, variance_exploding, nnv, diff_stop)
# forward = evolve_f([target_value], n_diffs, g, diff_stop)

# plot(t_r[n_diffs-diff_stop+1:end], reverse[1,:], color=:red, label="Reverse",ylims=(-0.5,1.5))
# plot!(t_r[n_diffs-diff_stop+1:end], reverse_vanilla[1,:], color=:green, label="Reverse Vanilla")
# plot!(t_f[1:diff_stop], forward[1,:], color=:blue, label="Forward")

