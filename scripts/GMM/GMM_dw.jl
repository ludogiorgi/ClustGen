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

hfile = h5open("data/GMM_data/double_well.hdf5", "r")
networks = []
gm_pdfs = []
gm_scores = []
σs = hfile["sigmas"][:]
yCs = []
for i in 1:4
    push!(networks, read(hfile["network vals $i"]))
    push!(gm_pdfs, read(hfile["gm pdfs $i"]))
    push!(gm_scores, read(hfile["gm_scores $i"]))
    push!(yCs, read(hfile["clusters $i"]))
end
exact_score = read(hfile["exact score"])
network_xs =read( hfile["network xs"])
x₀ = read(hfile["x values"])
close(hfile)

##

fig = Figure(resolution = (750, 750))
σs = [0.01, 0.05, 0.1, 0.5]
op = 0.7
for (k, σ) in enumerate(σs)
    ii = (k-1)÷2 + 1
    jj = (k-1) % 2 + 1
    yC = yCs[k]
    gm_score = gm_scores[k]
    ax = Axis(fig[ii, jj]; xlabel = "x", ylabel = "Score", title = "σ = $σ")
    xs = range(-2, 2, length = 100)
    ys = range(-20, 20, length = 100)
    GLMakie.heatmap!(ax, xs, ys, gm_pdfs[k], colormap = :grays, alpha = 0.8)
    GLMakie.contour!(ax, xs, ys, gm_pdfs[k], levels = 20, linewidth = 3, color = (:yellow, 0.05))
    GLMakie.lines!(ax, sort(x₀), gm_score[sortperm(x₀)], color = (:orange, op), linewidth = 3, label = "GMM score")
    GLMakie.lines!(ax, sort(x₀), exact_score[sortperm(x₀)], color = (:red, op), linewidth = 3, label = "Exact score")
    GLMakie.lines!(ax, network_xs, networks[k], color = (:blue, op), linewidth = 3, label = "KGMM Score Network")
    GLMakie.scatter!(ax, yC[1, :], yC[2, :], color = :yellow, label = "KGMM Score Points")
    # scatter!(ax, x⃗[1, :], x⃗[2, :], color = (:yellow, 0.1), label = "Samples")
    if k == 4
        axislegend(ax, position = :rt, orientation = :vertical)
    end
    GLMakie.xlims!(-2, 2)
    GLMakie.ylims!(-20, 20)
end
display(fig)
save("figures/GMM_figures/score_estimates_with_network.png", fig)
##
networks[1] = score_GMM



##
dim = 1 
inputs_targets = (yCs[1][1:1, :], yCs[1][2:2, :])

nn_clustered, loss_clustered = train(inputs_targets, 200, 1, [dim, 40, 20, dim]; use_gpu=true, activation=swish, last_activation=identity)
nn_clustered_cpu = nn_clustered |> cpu
score_clustered(x) = nn_clustered_cpu(Float32.([x...]))[:]

score_GMM = [score_clustered(x)[1] for x in network_xs]

plt1 = Plots.scatter(yCs[1][1, :], yCs[1][2, :])
plt1 = Plots.plot!(network_xs, score_GMM, label = "GMM Score")

plt2 = Plots.plot(loss_clustered)

Plots.plot(plt1, plt2, layout = (2, 1), size = (800, 800), legend = false)
