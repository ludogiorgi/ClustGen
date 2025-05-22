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
using PlotlyJS
using LinearAlgebra
using ProgressBars
using Distributions

isdefined(ClustGen, :train_giulio)  # deve restituire true
function create_nn_giulio(layers_size::Vector{Int}; activation_hidden=swish, activation_output=identity)
    #create an array to store the layers
    layers = [] 

    #create hidden layers
    for i in 1:length(layers_size)-2
        push!(layers, Flux.Dense(layers_size[i], layers_size[i+1], activation_hidden))
    end

    #create output layer
    push!(layers, Flux.Dense(layers_size[end-1], layers_size[end], activation_output))


    return Flux.Chain(layers...)
end

nn = create_nn_giulio([1, 50, 25, 1]; activation_hidden=swish, activation_output=identity)

#Define the rhs of the Lorenz system for later integration with evolve function, changes wrt GMM_Lorenz63: x is a 4 dimensional vector that contains x, y1,y2 and y3.
function F(x, t, σ, ε ; µ=10.0, ρ=28.0, β=8/3)
    dx = x[1] * (1 - x[1]^2) + (σ / ε) * x[3]
    dy1 = µ/ε^2 * (x[3] - x[2])
    dy2 = 1/ε^2 * (x[2] * (ρ - x[4]) - x[3])
    dy3 = 1/ε^2 * (x[2] * x[3] - β * x[4])
    return [dx, dy1, dy2, dy3]
end

function sigma(x, t; noise = 0.0)
    sigma1 = noise
    sigma2 = noise
    sigma3 = noise
    sigma4 = noise #Added: This is for the 4th variable
    return [sigma1, sigma2, sigma3, sigma4]
end

function normalize_f(f, x, t, M, S)
    return f(x .* S .+ M, t) .* S
end
σ=0.08 
ε=0.5
σ_value=0.05 
prob=0.007
conv_param=0.001
n_epochs=1000 
batch_size=32
########## 1. Simulate System ##########
ndim = 1
dimensions = 4 
dt = 0.01
Nsteps = 1000000000
resolution = 100
f = (x, t) -> F(x, t, σ, ε)
obs_nn = evolve(randn(4), dt, Nsteps, f, sigma; timestepper=:rk4, resolution=100)
ts = collect(0:(div(Nsteps, resolution) - 1)) .* (dt * resolution)
size(ts)
size(obs_nn)
########## 2. Normalize and compute autocovariance ##########
M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S

autocov_obs = zeros(dimensions, 50)
for i in 1:dimensions
    autocov_obs[i,:] = autocovariance(obs_nn[i,:]; timesteps=50)
end
plot(autocov_obs[3,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of y2")
D_eff = dt * (0.5*autocov_obs[3, 1] + sum(autocov_obs[3, 2:end-1]) + 0.5*autocov_obs[3, end])

########## 3. Clustering ##########
μ = reshape(obs[1,:], 1, :)
    
averages, _, centers, Nc, _ = f_tilde_ssp(σ_value, μ; prob=prob, do_print=false, conv_param=conv_param, normalization=false)
inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
typeof(inputs_targets)
size(inputs_targets[1])
@show size(nn(inputs_targets[1]))
@show size(inputs_targets[2])
# Plot dettagliato dell'istogramma dei centroidi non normalizzato
h = Plots.histogram(centers[1,:], 
    bins=30,
    title="Distribuzione dei Centroidi",
    xlabel="Valore del Centroide",
    ylabel="Numero di Centroidi",
    label="Conteggio Centroidi",
    fillalpha=0.6,
    color=:blue)
########## 4. Score functions ##########
f1(x,t) = x .- x.^3
score_true(x, t) = normalize_f(f1, x, t, M, S)
kde_x = kde(μ[1,:])

#x_vals = collect(range(minimum(centers[1, :]), maximum(centers[1, :]), length=200))
true_score = [2*score_true([x], 0.0)[1] / D_eff for x in kde_x.x]
plot(kde_x.x, true_score, label="True score", xlabel="x", ylabel="score(x)", title="True score function", markersize=2.5)
@show D_eff

########## 5. Train NN ##########
@time nn, losses = train(inputs_targets, n_epochs, batch_size, [ndim, 50, 25, ndim], opt=Flux.Adam(0.001), activation=swish, last_activation=identity, use_gpu=false)

nn_clustered_cpu = nn |> cpu
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value

########## 6. Compute PDF ##########

xax = [-1.6:0.02:1.6...]

pdf_interpolated_norm = compute_density_from_score(xax, score_clustered)
interpolated_score = [score_clustered([x])[1] for x in kde_x.x]

########## 7. Plotting ##########
#plot time series 
plotlyjs()
# p_time = plot(ts, obs_nn[1, 1:end], xlabel="t", ylabel="x(t)", label = "x(t)", title="Time series of x")
# p_autocovy = plot(autocov_obs[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")
p_autocovy = plot(autocov_obs[3,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")

p_score = Plots.scatter(inputs_targets[1][1,:], .-inputs_targets[2][1,:] ./ σ_value, label="Score from data", xlabel="x", ylabel="score(x)", title="Real score and interpolated score from NN", markersize=2.5)
Plots.plot!(p_score, kde_x.x, interpolated_score, label="NN interpolation", linewidth=2, color=:steelblue)
Plots.plot!(p_score, kde_x.x, true_score, label="True score", linewidth=2, color=:red)

p_pdf = Plots.plot(kde_x.x, kde_x.density, label="PDF observed", xlabel="x", ylabel="PDF", title="Estimated PDF vs True", linewidth=2, color=:blue)
Plots.plot!(p_pdf, xax, pdf_interpolated_norm, label="PDF learned", linewidth=2, color=:red)
# p_log = Plots.plot(kde_x.x, log_pdf_interpolated, label="log PDF from NN interpolation", xlabel="x", ylabel="log PDF", title="Estimated log PDF vs True", linewidth=2, color=:red)
# Plots.plot!(p_log, kde_x.x, log.(kde_x.density), label="log PDF of real data", linewidth=2, color=:blue)
p_loss = Plots.plot(1:n_epochs, losses, xlabel="Epoch", ylabel="Loss", title="NN training loss", linewidth=2, label="Loss")
########## 8. Save or Display ##########
# if save_figs
#     base_path = "/Users/giuliodelfelice/Desktop/MIT"
#     test_folder = joinpath(base_path, "TEST_$(test_number)")
#     mkpath(test_folder)
#     savefig(p_time, joinpath(test_folder, "time_series.pdf"))
#     savefig(p_autocovy, joinpath(test_folder, "fast_signal_autocov.pdf"))
#     savefig(p_score, joinpath(test_folder, "Interpolation.pdf"))
#     savefig(p_pdf, joinpath(test_folder, "PDFs.pdf"))
#     savefig(p_loss, joinpath(test_folder, "loss_plot.pdf"))
#     display(p_time)
#     display(p_autocovy)
#     display(p_score)
#     display(p_pdf)
#     display(p_loss)
# else
    #display(p_time)
    display(p_autocovy)
    display(p_score)
    display(h)
    display(p_pdf)
    #display(p_log)
    display(p_loss)
        
