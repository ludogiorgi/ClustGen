using Revise
using ClustGen
using Flux, BSON, HDF5, ProgressBars, Plots, Random, LinearAlgebra, Statistics, KernelDensity
plotly()

T = 10000000.0
dt = 0.01
dim = 8
F = 8
u0 = randn(dim)
noise = 0.2

u_lorenz96 = simulate_lorenz96(T, dt, F, u0, noise; res=100)
obs = (u_lorenz96 .- mean(u_lorenz96)) ./ std(u_lorenz96)

scatter(obs[1,1:10000], obs[2,1:10000], obs[3,1:10000], markersize=1)
##
############################ CLUSTERING ####################

normalization = false
σ_value = 0.2

μ = repeat(obs, 1, 1)

@time inputs_targets = f_tilde(σ_value, μ; prob=0.0005, do_print=true, conv_param=0.002, normalization=normalization)
if normalization == true
    inputs, targets, M_averages_values, m_averages_values = inputs_targets
else
    inputs, targets = inputs_targets
end
plotly()
targets_norm = [norm(targets[:,i]) for i in eachindex(inputs[1,:])]
Plots.scatter(inputs[1,:], inputs[2,:], inputs[3,:],  marker_z=targets_norm, color=:viridis)
##
#################### TRAINING WITH CLUSTERING LOSS ####################

@time nn_clustered, loss_clustered = train(inputs_targets, 1000, 64, [dim, 128, 64, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value
cluster_loss = check_loss(obs, nn_clustered_cpu, σ_value)
println(cluster_loss)
Plots.plot(loss_clustered)

##
########### Additional training on the full data (not used in the draft) ############

@time nn_clustered, loss_clustered = train(obs, 20, 128, nn_clustered, σ_value; use_gpu=true)

##
#################### TRAINING WITH VANILLA LOSS ####################

@time nn_vanilla, loss_vanilla = train(obs, 200, 128, [dim, 128, 64, dim], σ_value; use_gpu=true, opt=Adam(0.001))
nn_vanilla_cpu = nn_vanilla |> cpu
score_vanilla(x) = .- nn_vanilla_cpu(Float32.([x...])) ./ σ_value
Plots.plot(loss_vanilla)
hline!([cluster_loss])

##
#################### SAMPLES GENERATION ####################

trj_clustered = sample_langevin(10000, 0.01, score_clustered, obs; seed=123, res = 100, boundary=[-50,50])
#trj_vanilla = sample_langevin(10000, 0.01, score_vanilla, randn(dim); seed=123, res = 100, boundary=[-50,50])

gr()
kde_clustered_xy = kde(trj_clustered[1:2,:]')
kde_clustered_xw = kde(trj_clustered[[1,3],:]')
kde_clustered_xz = kde(trj_clustered[[1,4],:]')

# kde_vanilla_xy = kde(trj_vanilla[1:2,:]')
# kde_vanilla_xw = kde(trj_vanilla[[1,3],:]')
# kde_vanilla_xz = kde(trj_vanilla[[1,4],:]')

kde_obs_xy = kde(obs[1:2,:]')
kde_obs_xw = kde(obs[[1,3],:]')
kde_obs_xz = kde(obs[[1,4],:]')

plt1 = heatmap(kde_clustered_xy.x, kde_clustered_xy.y, kde_clustered_xy.density, xlabel="X", ylabel="Y", title="Sampled PDF XY")
plt2 = heatmap(kde_clustered_xw.x, kde_clustered_xw.y, kde_clustered_xw.density, xlabel="X", ylabel="W", title="Sampled PDF XW")
plt3 = heatmap(kde_clustered_xz.x, kde_clustered_xz.y, kde_clustered_xz.density, xlabel="X", ylabel="Z", title="Sampled PDF XZ")

plt4 = heatmap(kde_obs_xy.x, kde_obs_xy.y, kde_obs_xy.density, xlabel="X", ylabel="Y", title="Observed PDF XY")
plt5 = heatmap(kde_obs_xw.x, kde_obs_xw.y, kde_obs_xw.density, xlabel="X", ylabel="W", title="Observed PDF XW")
plt6 = heatmap(kde_obs_xz.x, kde_obs_xz.y, kde_obs_xz.density, xlabel="X", ylabel="Z", title="Observed PDF XZ")

plot(plt1, plt2, plt3, plt4, plt5, plt6, layout=(2,3), size=(1200,600))

# plt7 = heatmap(kde_vanilla_xy.x, kde_vanilla_xy.y, kde_vanilla_xy.density, xlabel="X", ylabel="Y", title="Sampled PDF XY Vanilla")
# plt8 = heatmap(kde_vanilla_xw.x, kde_vanilla_xw.y, kde_vanilla_xw.density, xlabel="X", ylabel="W", title="Sampled PDF XW Vanilla")
# plt9 = heatmap(kde_vanilla_xz.x, kde_vanilla_xz.y, kde_vanilla_xz.density, xlabel="X", ylabel="Z", title="Sampled PDF XZ Vanilla")

# plot(plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8, plt9, layout=(3,3), size=(1200,600))

savefig("figures/lorenz96_D$(dim)_sigma$(noise)_F$(F).png")


##
plotly()
plt1 = scatter(trj_clustered[1,1:10000], trj_clustered[2,1:10000], trj_clustered[3,1:10000], markersize=0.2)
#plt1 = scatter(trj_vanilla[1,1:10000], trj_vanilla[2,1:10000], trj_vanilla[3,1:10000], markersize=0.2)
plt1 = scatter!(obs[1,1:10000], obs[2,1:10000], obs[3,1:10000], markersize=1)

##
plotly()
targets_pred = nn_clustered_cpu(inputs)

scatter(inputs[1,:], targets[1,:], markersize=3, color=:red)
scatter!(inputs[1,:], targets_pred[1,:], markersize=3, color=:blue)

