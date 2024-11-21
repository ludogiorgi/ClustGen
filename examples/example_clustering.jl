using Revise
using ClustGen
using KernelDensity
using HDF5
using Flux
using BSON
using Plots
using LinearAlgebra
using Random

dim = 1

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

function generate_xz(y, σ)
    z = randn!(similar(y))
    x = @. y + σ * z
    return x, z
end

function generate_data(obs, σ)
    x, z = generate_xz(obs', σ)
    inputs = hcat([[x[i, :]...] for i in 1:size(obs, 2)]...)
    targets = hcat([z[i, :] for i in 1:size(obs, 2)]...)
    return inputs, targets
end

force(x) = -∇U_1D(x)[1]

normalization = false
σ_values = [0.02, 0.05, 0.1, 0.2]
μ = repeat(obs, 1, 1)
inputs = []
targets = []
inputs_noisy = []
targets_noisy = []
for σ_value in σ_values
    inputs_noisy_temp, targets_noisy_temp = generate_data(obs, σ_value)
    push!(inputs_noisy, inputs_noisy_temp)
    push!(targets_noisy, targets_noisy_temp)
    inputs_targets = f_tilde(σ_value, μ; prob=0.025, do_print=true, conv_param=0.0001, normalization=normalization)
    inputs_temp, targets_temp = inputs_targets
    push!(inputs, inputs_temp)
    push!(targets, targets_temp)
end
##
xax = -2:0.05:2

plt = plot(layout=(2,2), size=(800,800))
for i in 1:4
    scatter!(plt[i], inputs_noisy[i][1,1:10:end], .- targets_noisy[i][1,1:10:end] ./ σ_values[i], color=:orange, label="Noisy Data", markerstrokewidth=0, markersize=1, title="σ = $(σ_values[i])")
    plot!(plt[i], xax, force.(xax), color=:green, lw=2, label="True forcing")
    scatter!(plt[i], inputs[i][1,:], .- targets[i][1,:] ./ σ_values[i], color=:red, markerstrokewidth=0, markersize=4, label="Clustered Data")
end
plt

savefig(plt, pwd() * "/figures/1D_pot_fig.png")
