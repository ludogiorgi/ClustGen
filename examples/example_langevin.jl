using Pkg
Pkg.activate(".")
Pkg.instantiate()
##
using Revise
using ClustGen
using KernelDensity
using HDF5
using Flux
using BSON
using Plots
using LinearAlgebra

dim = 2

if isfile(pwd() * "/data/potential_data_D$(dim).hdf5")
    @info "potential well data already exists. skipping data generation"
else 
    potential_data(randn(dim), 20000000, 0.025, 200)
end

hfile = h5open(pwd() * "/data/potential_data_D$(dim).hdf5", "r")
obs = read(hfile["x"])
dt = read(hfile["dt"])
res = read(hfile["res"])
close(hfile)

##
############################ CLUSTERING ####################

normalization = false
σ_value = 0.05

μ = repeat(obs, 1, 1)

@time inputs_targets = f_tilde(σ_value, μ; prob=0.001, do_print=true, conv_param=0.001, normalization=normalization)
if normalization == true
    inputs, targets, M_averages_values, m_averages_values = inputs_targets
else
    inputs, targets = inputs_targets
end
plotly()
targets_norm = [norm(targets[:,i]) for i in eachindex(inputs[1,:])]
Plots.scatter(inputs[1,:], inputs[2,:], marker_z=targets_norm, color=:viridis)

##
#################### TRAINING WITH CLUSTERING LOSS ####################

@time nn_clustered, loss_clustered = train(inputs_targets, 1000, 128, [dim, 128, 64, dim]; use_gpu=true, activation=swish, last_activation=identity)
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

@time nn_vanilla, loss_vanilla = train(obs, 20, 128, [dim, 128, 64, dim], σ_value; use_gpu=true, opt=Adam(0.001))
nn_vanilla_cpu = nn_vanilla |> cpu
score_vanilla(x) = .- nn_vanilla_cpu(Float32.([x...])) ./ σ_value
Plots.plot(loss_vanilla)
hline!([cluster_loss])

##
#################### VECTOR FIELDS ####################

gr()
force(x) = -∇U_2D(x)
vf(x, y) = force([x,y])
vf_c(x, y) = score_clustered([x, y])
vf_v(x, y) = score_vanilla([x, y])

n_grid = 30
d_grid = 1/10
c_grid = [((n_grid+1)*d_grid)/2, ((n_grid+1)*d_grid)/2]
grid = meshgrid(n_grid) .* d_grid .- c_grid

plt1 = vectorfield2d(vf_c, grid, xlabel="X", ylabel="Y", title="Clustered Force Field")
plt2 = vectorfield2d(vf_v, grid, xlabel="X", ylabel="Y", title="Vanilla Force Field")
plt3 = vectorfield2d(vf, grid, xlabel="X", ylabel="Y", title="Observed Force Field")
plot(plt1, plt2, plt3, layout=(3, 1), size=(800, 1200))

##
#################### SAMPLES GENERATION ####################

trj_clustered = sample_langevin(10000, 0.01, score_clustered, randn(2); seed=123, res = 1)
trj_vanilla = sample_langevin(10000, 0.01, score_vanilla, randn(2); seed=123, res = 1)

kde_clustered = kde(trj_clustered')
kde_vanilla = kde(trj_vanilla')
kde_obs = kde(obs')

plt1 = heatmap(kde_clustered.x, kde_clustered.y, kde_clustered.density, xlabel="X", ylabel="Y", title="Clustered PDF")
plt2 = heatmap(kde_vanilla.x, kde_vanilla.y, kde_vanilla.density, xlabel="X", ylabel="Y", title="Vanilla PDF")
plt3 = heatmap(kde_obs.x, kde_obs.y, kde_obs.density, xlabel="X", ylabel="Y", title="Observed PDF")
plot(plt1, plt2, plt3, layout=(3, 1), size=(800, 1200))

##
#################### NN SAVINGS ####################

BSON.@save pwd() * "/NNs/nn_clustered_D$(dim)_$(normalization)_$(σ_value).bson" nn_clustered_cpu
BSON.@save pwd() * "/NNs/nn_vanilla_D$(dim)_$(normalization)_$(σ_value).bson" nn_vanilla_cpu

##
#################### NN LOADINGS ####################

BSON.load(pwd() * "/NNs/nn_clustered_D$(dim)_$(normalization)_$(σ_value).bson")[:nn_clustered_cpu]
BSON.load(pwd() * "/NNs/nn_vanilla_D$(dim)_$(normalization)_$(σ_value).bson")[:nn_vanilla_cpu]

##
#################### DRAFT PLOTS ####################

plt1 = vectorfield2d(vf_c, grid, xlabel="X", ylabel="Y", title="Learned Force Field")
plt2 = heatmap(kde_clustered.x, kde_clustered.y, kde_clustered.density, xlabel="X", ylabel="Y", title="Learned PDF", clims=(0, 0.5), colormap=:viridis)
plt3 = vectorfield2d(vf, grid, xlabel="X", ylabel="Y", title="Observed Force Field")
plt4 = heatmap(ekde_obs.x, kde_obs.y, kde_obs.density, xlabel="X", ylabel="Y", title="Observed PDF", clims=(0, 0.5), colormap=:viridis)
plot(plt1, plt2, plt3, plt4, layout=(2, 2), size=(800, 800))
#savefig(pwd() * "/figures/2D_pot_fig2.png")

##
scatter(inputs[1,:], inputs[2,:], marker_z=targets_norm, label="", xlabel="X", ylabel="Y", title="Vector Field Norm", size=(600, 600), colormap=:viridis, markersize=7)
savefig(pwd() * "/figures/2D_pot_fig1.png")