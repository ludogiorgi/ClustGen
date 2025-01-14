using Pkg
Pkg.activate(".")
Pkg.instantiate()
##
using Revise
using ClustGen
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

obs = (encoder(data) .+ 1) ./ 2
dim = 8

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

normalization = false
n_diffs = 20
diff_times = [i/n_diffs for i in 1:n_diffs]
σ_values = σ_variance_exploding.(diff_times)

μ = repeat(obs, 1, 1)

inputs_targets = f_tilde(σ_values, diff_times, μ; prob=0.001, do_print=true, conv_param=0.001, normalization=normalization)

if normalization == true
    inputs, targets, M_averages_values, m_averages_values = inputs_targets
else
    inputs, targets = inputs_targets
end

##
#################### TRAINING WITH CLUSTERING LOSS ####################

@time nn_clustered, loss_clustered = train(inputs_targets, 500, 128, [dim+1, 128, 64, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:]
cluster_loss = check_loss(obs, nn_clustered_cpu, σ_variance_exploding)
println(cluster_loss)
Plots.plot(loss_clustered)

##
########### Additional training on the full data (not used in the draft) ############

@time nn_clustered, loss_clustered = train(obs, 20, 128, nn_clustered, σ_variance_exploding; use_gpu=true)

##
#################### TRAINING WITH VANILLA LOSS ####################

@time nn_vanilla, loss_vanilla = train(obs, 100, 128, [dim+1, 128, 64, dim], σ_variance_exploding; use_gpu=true, opt=Adam(0.0001))
nn_vanilla_cpu = nn_vanilla |> cpu
score_vanilla(x) = .- nn_vanilla_cpu(Float32.([x...]))
Plots.plot(loss_vanilla)
hline!([cluster_loss])

##
#################### SAMPLES GENERATION ####################

ens_clustered = sample_reverse(dim, score_clustered, 100, 100, σ_variance_exploding, g_variance_exploding)
ens_vanilla = sample_reverse(dim, score_vanilla, 100, 100, σ_variance_exploding, g_variance_exploding)

ens_clustered_decoded = decoder(ens_clustered .* 2 .- 1)
ens_vanilla_decoded = decoder(ens_vanilla .* 2 .- 1)

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
##
#################### NNs SAVINGS ####################


BSON.@save pwd() * "/NNs/MNIST_nn_clustered_L$(length(σ_values)).bson" nn_clustered_cpu
BSON.@save pwd() * "/NNs/MNIST_nn_vanilla_L$(length(σ_values)).bson" nn_vanilla_cpu


##
#################### NNs LOADINGS ####################

BSON.@load (pwd() * "/NNs/MNIST_nn_clustered_L$(length(σ_values)).bson")[:nn_clustered_cpu]
BSON.@load (pwd() * "/NNs/MNIST_nn_vanilla_L$(length(σ_values)).bson")[:nn_vanilla_cpu]

##
#######################################################################################

############################### LANGEVIN SAMPLING ####################################



############################ CLUSTERING ####################

normalization = true
σ_value = 0.05

μ = repeat(obs, 1, 1)

inputs_targets = f_tilde(σ_value, μ; prob=0.001, do_print=true, conv_param=0.001, normalization=normalization)
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

@time nn_clustered, loss_clustered = train(inputs_targets, 2000, 128, [dim, 128, 64, dim]; use_gpu=true, activation=swish, last_activation=sigmoid, opt=Adam(0.001))
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

@time nn_vanilla, loss_vanilla = train(obs, 100, 128, [dim, 128, 64, dim], σ_value; use_gpu=true, opt=Adam(0.0001))
nn_vanilla_cpu = nn_vanilla |> cpu
score_vanilla(x) = .- nn_vanilla_cpu(Float32.([x...])) ./ σ_value
Plots.plot(loss_vanilla)
hline!([cluster_loss])

##
#################### SAMPLES GENERATION ####################

ens_clustered = sample_langevin(10000, 0.01, score_clustered, randn(dim); seed=123, res = 100, boundary=true)
ens_vanilla = sample_langevin(10000, 0.01, score_vanilla, randn(dim); seed=123, res = 100, boundary=true)

test_sample1 = zeros(28*28, 16)
test_sample2 = zeros(28*28, 16)
for i in 1:16
    test_sample1[:,i] = decoder(ens_clustered[:,rand(1:10000)] .* 2 .- 1)
    test_sample2[:,i] = decoder(ens_vanilla[:,rand(1:10000)] .* 2 .- 1)
end

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

##
#################### NNs SAVINGS ####################


BSON.@save pwd() * "/NNs/MNIST_nn_clustered_L$(length(σ_values)).bson" nn_clustered_cpu
BSON.@save pwd() * "/NNs/MNIST_nn_vanilla_L$(length(σ_values)).bson" nn_vanilla_cpu


##
#################### NNs LOADINGS ####################

BSON.@load (pwd() * "/NNs/MNIST_nn_clustered_L$(length(σ_values)).bson")[:nn_clustered_cpu]
BSON.@load (pwd() * "/NNs/MNIST_nn_vanilla_L$(length(σ_values)).bson")[:nn_vanilla_cpu]