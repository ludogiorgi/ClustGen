#################### MNIST DATASET ####################

include(pwd() * "/src/ClustGen.jl")

using Main.ClustGen
using MLDatasets, Flux, BSON, HDF5, ProgressBars, Plots, Random, LinearAlgebra, Statistics, KernelDensity

train_x, _ = MNIST(split=:train)[:]
test_x, _ = MNIST(split=:test)[:]

train_x = reshape(train_x, 28*28, :)
test_x = reshape(test_x, 28*28, :)

data = Flux.Float32.(hcat(train_x, test_x))

file_name = "MNIST"

if !isfile(pwd() * "/NNs/encoder_$(file_name).bson") || !isfile(pwd() * "/NNs/decoder_$(file_name).bson")
    encoder, decoder = apply_autoencoder(data, [28*28, 128, 32, 8], 30, 32, file_name=file_name)
else
    encoder, decoder = read_autoencoder(file_name)
end
##
#################### AUTOENCODER CHECK ####################

test_sample = data[:, rand(1:size(data, 2))]
autoencoder = Flux.Chain(encoder, decoder)
reconstructed = Flux.cpu(autoencoder(test_sample))

p = Plots.plot(
    Plots.heatmap(reshape(test_sample, 28, 28), title="Original"),
    Plots.heatmap(reshape(reconstructed, 28, 28), title="Reconstructed"),
    layout = (1, 2)
)
display(p)

##
#################### CLUSTERING ####################

variance_exploding(t; σ_min = 0.01, σ_max = 1.0) = @. σ_min * (σ_max/σ_min)^t
g(t; σ_min=0.01, σ_max=1.0) = σ_min * (σ_max/σ_min)^t * sqrt(2*log(σ_max/σ_min))

n_diffs = 20
diff_times = [i/n_diffs for i in 1:n_diffs]
σ_values = variance_exploding.(diff_times)
obs = (encoder(data) .+ 1) ./ 2

# kde_obs = kde(obs[3,:])
# plot(kde_obs.x, kde_obs.density, color=:blue, label="Observed")

μ = repeat(obs, 1, 1)

averages_values, centers_values, Nc_values = f_tilde(σ_values, μ; prob=0.001, do_plot=true, conv_param=0.001)
##
plotly()
ii = 1
k_norm = [norm(averages_values[ii][:,i]) for i in 1:Nc_values[ii]]
Plots.scatter(centers_values[ii][1,:], centers_values[ii][2,:], centers_values[ii][3,:], marker_z=k_norm, color=:viridis, markersize=1)
##
#################### TRAINING WITH CLUSTERING LOSS ####################

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

nn_clustered, loss_clustered = train_clustered(data_clustered, 250, 128, [Dim+1, 128, 64, Dim]; activation=tanh)
nnc(x, t) = .- nn_clustered(Flux.Float32.([x..., t]))[:] .* (M_averages_values .- m_averages_values) .- m_averages_values
Plots.plot(loss_clustered)
##
#################### TRAINING WITH VANILLA LOSS ####################

nn_vanilla, loss_vanilla = train_vanilla(obs, 200, 128, [Dim+1, 128, 64, Dim], variance_exploding; activation=tanh)
nnv(x, t) = nn_vanilla(Flux.Float32.([x..., t]))[:]
cluster_loss = check_loss(obs, nnc, variance_exploding)
Plots.plot(loss_vanilla)
Plots.hline!([cluster_loss])
savefig("loss_vanilla_mnist.png")
##
######################## SAMPLES GENERATION ########################

n_diffs_sampling = 100

ens_clustered = sample(Dim, nnc, 100, n_diffs_sampling, variance_exploding, g)
ens_vanilla = sample(Dim, nnv, 100, n_diffs_sampling, variance_exploding, g)

ens_clustered_decoded = decoder(ens_clustered .* 2 .- 1)
ens_vanilla_decoded = decoder(ens_vanilla .* 2 .- 1)
##
test_sample1 = ens_clustered_decoded[:, rand(1:100,16)]
test_sample2 = ens_vanilla_decoded[:, rand(1:100,16)]

plt1 = Plots.plot(layout = (4, 2), size=(1000,1000), title="Clustered")
plt2 = Plots.plot(layout = (4, 2), size=(1000,1000), title="Vanilla")

for i in 1:8
    sample_matrix1 = reshape(test_sample1[:,i], 28, 28)
    sample_matrix2 = reshape(test_sample2[:,i], 28, 28)

    rotated_matrix1 = rotl90(sample_matrix1)
    rotated_matrix2 = rotl90(sample_matrix2)

    Plots.heatmap!(plt1, rotated_matrix1, subplot = i, colorbar=false)
    Plots.heatmap!(plt2, rotated_matrix2, subplot = i, colorbar=false)
end
Plots.plot(plt1, plt2)
savefig("samples_mnist.png")