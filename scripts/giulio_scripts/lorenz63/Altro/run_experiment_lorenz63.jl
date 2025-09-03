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
using Distributions
using QuadGK
using LaTeXStrings



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

function run_experiments(; fix_initial_state=true,
    σ=0.08, ε=0.5,
    σ_value=0.05, prob=0.007, conv_param=0.002,
    n_epochs=1000, batch_size=32,
    label="experiment", test_number=1, save_figs=true)
     ########## 1. Simulate System ##########
    ndim = 1
    dimensions = 4
    #only if use_fourier == true
    n_freq = 8


    dt = 0.01
    Nsteps = 1000000000
    f = (x, t) -> F(x, t, σ, ε)
    # if fix_initial_state == true
    #     initial_state = [-1.5, -2.0, -2.0, -2.0]
    # else
    #     initial_state = [-1.5, randn(3)]
    # end
    obs_nn = evolve(randn(4), dt, Nsteps, f, sigma; timestepper=:rk4, resolution=1000)
    println("size of time series:", size(obs_nn))
    #ts = collect(0:dt*resolution:dt*(Nsteps-resolution)) 
    # for subsequent plotting
    obs_uncorr = obs_nn[1:1, 1:1:end]
    ########## 2. Normalize and compute autocovariance ##########
    M = mean(obs_nn, dims=2)
    S = std(obs_nn, dims=2)
    obs = (obs_nn .- M) ./ S

    autocov_obs = zeros(dimensions, 100)
    for i in 1:dimensions
        autocov_obs[i,:] = autocovariance(obs_nn[i,:]; timesteps=100)
    end

    D_eff = dt * (0.5*autocov_obs[3, 1] + sum(autocov_obs[3, 2:end-1]) + 0.5*autocov_obs[3, end])
    D_eff =  0.3
    @show D_eff

    ########## 3. Clustering ##########
    #μ = reshape(obs[1,:], 1, :)
    
    averages, _, centers, Nc, _ = f_tilde_ssp(σ_value, obs_uncorr; prob=prob, do_print=false, conv_param=conv_param, normalization=false)

    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
    @show typeof(inputs_targets)

    #gr()  
    
    # Usiamo il backend GR che è più stabile

    # Plot dettagliato dell'istogramma dei centroidi non normalizzato
    # h = Plots.histogram(centers[1,:], 
    #     bins=30,
    #     title="Distribuzione dei Centroidi",
    #     xlabel="Valore del Centroide",
    #     ylabel="Numero di Centroidi",
    #     label="Conteggio Centroidi",
    #     fillalpha=0.6,
    #     color=:blue)

    ########## 4. Score functions ##########
    f1(x,t) = x .- x.^3
    score_true(x, t) = normalize_f(f1, x, t, M, S)

    kde_x = kde(obs_nn[1,200:end])

    #x_vals = collect(range(minimum(centers[1, :]), maximum(centers[1, :]), length=200))

    centers_sorted_indices = sortperm(centers[1,:])

    centers_sorted = centers[:,centers_sorted_indices][:]
    scores = .- averages[:,centers_sorted_indices][:] ./ σ_value

    ########## 5. Train NN ##########

    #create and train the model
    @time nn, losses = train_giulio(inputs_targets, n_epochs, batch_size, [ndim, 50, 25, ndim]; opt=Flux.Adam(0.001), activation_hidden=swish, activation_output=identity, use_gpu=false, use_fourier=false, input_dim=1)

    nn_clustered_cpu = nn |> cpu
    score_clustered(x) = .- nn_clustered_cpu(reshape(Float32[x...], :, 1))[:] ./ σ_value



    ########## 6. Compute PDF ##########
  
    function true_pdf_normalized(x)
        x_phys = x .* S[1] .+ M[1]  # torna nello spazio fisico
        U = .-0.5 .* x_phys.^2 .+ 0.25 .* x_phys.^4
        p = exp.(-2 .* U ./ D_eff)
        return p ./ S[1]  # cambio di variabile
    end
    

    
    xax = [-1.25:0.005:1.25...]


    interpolated_score = [score_clustered(xax[i])[1] for i in eachindex(xax)]

    true_score = [2*score_true(xax[i], 0.0)[1] / D_eff for i in eachindex(xax)]

    xax_2 = [-1.6:0.02:1.6...]
    pdf_interpolated_norm = compute_density_from_score(xax_2, score_clustered)
    pdf_true = true_pdf_normalized(xax_2)
    scale_factor = maximum(kde_x.density) / maximum(pdf_true)
    pdf_true .*= scale_factor
# --------- Normalized PDF of y₂(t) vs Gaussian fit ---------
y2_samples = obs_nn[3, 200:end]
kde_y2 = kde(y2_samples)

μ_y2 = mean(y2_samples)
σ_y2 = std(y2_samples)
gauss_y2(x) = pdf(Normal(μ_y2, σ_y2), x)

xax_y2 = kde_y2.x
pdf_kde = kde_y2.density
pdf_gaussian_y2 = [gauss_y2(x) for x in xax_y2]

# Calcola passo dx per integrazione numerica (rettangoli)
dx = xax_y2[2] - xax_y2[1]

# Normalizza entrambe le distribuzioni per area = 1
area_kde = sum(pdf_kde) * dx
area_gauss = sum(pdf_gaussian_y2) * dx

pdf_kde ./= area_kde
pdf_gaussian_y2 ./= area_gauss


    ########## 7. Plotting ##########
    # Imposta font coerente con LaTeX
Plots.default(fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10, legendfontsize=10)

# Backend
plotlyjs()

# Plot score
p_score = scatter(
    centers_sorted, scores;
    color=:blue, alpha=0.2, label="Cluster centers",
    xlims=(-1.3, 1.3), ylims=(-5, 5),
    xlabel="𝑥", ylabel="𝐒𝐜𝐨𝐫𝐞(𝑥)", title="Score Function Estimate"
)
plot!(p_score, xax, interpolated_score; label="NN interpolation", linewidth=2, color=:red)
plot!(p_score, xax, true_score; label="Score analytic", linewidth=2, color=:green)

# Plot PDF
p_pdf = plot(
    kde_x.x, kde_x.density;
    label="PDF observed", xlabel="𝑥", ylabel="PDF",
    title="Estimated PDF vs True", linewidth=2,color=:blue)
plot!(p_pdf, xax_2, pdf_interpolated_norm; label="PDF learned", linewidth=2,color=:red)
plot!(p_pdf, xax_2, pdf_true; label="PDF analytic", linewidth=2, linestyle=:dash, color=:green)

# Plot loss
p_loss = plot(losses; xlabel="Epoch", ylabel="Loss", title="NN training loss", linewidth=2, label="Loss")
p_y2 = plot(
    kde_y2.x, kde_y2.density;
    label="PDF of y2(t)", xlabel="y2", ylabel="Density",
    title="Distribution of y2(t)", linewidth=2
)
plot!(p_y2, xax_y2, pdf_gaussian_y2; label="Gaussian fit", linewidth=2)


    ########## 8. Save or Display ##########
    if save_figs == true
        base_path = "/Users/giuliodelfelice/Desktop/MIT"
        test_folder = joinpath(base_path, "TEST_$(test_number)")
        mkpath(test_folder)
        savefig(p_time, joinpath(test_folder, "time_series.pdf"))
        savefig(p_autocovy, joinpath(test_folder, "fast_signal_autocov.pdf"))
        savefig(p_score, joinpath(test_folder, "Interpolation.pdf"))
        savefig(p_pdf, joinpath(test_folder, "PDFs.pdf"))
        savefig(p_loss, joinpath(test_folder, "loss_plot.pdf"))
        display(p_time)
        display(p_autocovy)
        display(p_score)
        #display(p_pdf)
        display(p_loss)
    else
        #display(p_time)
        #display(p_autocovy)
        display(p_score)
        #display(h)
        display(p_pdf)
        #display(p_log)
        display(p_loss)
        display(p_y2)
    end

    return (loss=losses[end], centers, interpolated_score, pdf_interpolated_norm, D_eff)
end