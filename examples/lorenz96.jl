include(pwd() * "/src/ClustGen.jl")

using Main.ClustGen: f_tilde, train_clustered, train_vanilla, check_loss, sample
using Random, Distributions, Plots, Statistics, ProgressBars, KernelDensity
using FFTW, Interpolations, LinearAlgebra, Flux, Measures, HDF5
plotly()

# Lorenz 96 Model parameters
F = 8  # Forcing term

function lorenz96(u, F)
    n = length(u)
    du = similar(u)
    for i in 1:n
        du[i] = (u[mod1(i+1, n)] - u[mod1(i-2, n)]) * u[mod1(i-1, n)] - u[i] + F
    end
    return du
end

function simulate_lorenz96(T, dt, F, sigma; seed=123, u0 = rand(4))
    Random.seed!(seed)
    N = Int(T / dt)
    u = zeros(length(u0), N)
    u[:, 1] = u0
    W = randn(length(u0), N) .* sigma .* sqrt(dt)
    
    for t in 2:N
        du = lorenz96(u[:, t-1], F) * dt + W[:, t-1]
        u[:, t] = u[:, t-1] + du
    end
    
    return u
end

function autocorrelation(x)
    x = x .- mean(x)
    acf = [cor(x[1:end-k], x[k+1:end]) for k in 0:1000]
    return acf
end

# Parameters
T = 10000.0
dt = 0.01

u_lorenz96 = simulate_lorenz96(T, dt, F, sqrt(2.0))
u_lorenz96_n = (u_lorenz96 .- mean(u_lorenz96, dims=2)) ./ std(u_lorenz96, dims=2)

# Plot time series
tr = plot(u_lorenz96[1, 1:10000])

# KDE of normalized data
kde_lorenz96 = kde(u_lorenz96_n[1, :])

hist = plot(kde_lorenz96.x, kde_lorenz96.density, label="Lorenz 96", title="Histograms")

ac = plot(0:dt:1000*dt, autocorrelation(u_lorenz96[1, :]), title="Autocorrelations", xlabel="Lag", ylabel="Autocorrelation", label="")

plot(tr, hist, ac, layout=(3, 1), size=(800, 1200))
##
# Observation from longer simulation
T_obs = 1000000
obs = simulate_lorenz96(T_obs, dt, F, sqrt(2.0))
obs = obs[:, 1:100:end]
obs = (obs .- mean(obs, dims=2)) ./ std(obs, dims=2)

kde_obs = kde(u_lorenz96_n[1, :])
interp_P = interpolate((kde_obs.x,), kde_obs.density, Gridded(Linear()))
extrp_P = extrapolate(interp_P, Flat())
plt = plot(-3:0.01:5, extrp_P(-3:0.01:5), color=:red, label="Observed")
display(plt)

##
##################### CLUSTERING #####################

σ_min = 0.001
variance_exploding(t; σ_min = σ_min, σ_max = 5.0) = @. σ_min * (σ_max/σ_min)^t

n_diffs = 1
diff_times = [0.0] #[i/n_diffs for i in 1:n_diffs]
σ_values = variance_exploding.(diff_times)
μ = repeat(obs, 1, 1)

averages_values, centers_values, Nc_values = f_tilde(σ_values, μ; prob=0.0001, do_plot=true, conv_param=0.01)

##
####################### CLUSTERING CHECK #######################

ii = 1
k_norm = [norm(averages_values[ii][:, i]) for i in 1:Nc_values[ii]]
Plots.scatter(centers_values[ii][1, 1:10:end], centers_values[ii][2, 1:10:end], centers_values[ii][3, 1:10:end], marker_z=k_norm[1:10:end], color=:viridis, markersize=1)

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
    data_t = Flux.Float32.(hcat([[centers_values[t][:, i]..., averages_values_norm[t][:, i]...] for i in 1:Nc_values[t]]...))
    push!(data_clustered, [(data_t[1:Dim, i], data_t[Dim+1:2*Dim, i]) for i in 1:Nc_values[t]])
end
data_clustered = vcat(data_clustered...)

nn_clustered, loss_clustered = train_clustered(data_clustered, 500, 128, [Dim, 128, 64, Dim]; activation=tanh, opt=ADAM(0.0001))
nnc(x) = .-nn_clustered(Flux.Float32.([x...]))[:] .* (M_averages_values .- m_averages_values) .- m_averages_values
Plots.plot(loss_clustered)

##
######################## SAMPLES GENERATION ########################

function score_integral(score, N, dt, Dim)
    X = zeros(Dim, N)
    r = randn(Dim, N) .* sqrt(dt)
    for i in ProgressBar(2:N-1)
        X[:, i+1] .= X[:, i] .+ score(X[:, i]) .* dt ./ σ_min .+ sqrt(2) .* r[:, i]
        if norm(X[:, i+1]) > 100
            X[:, i+1] .= 0.0
        end
    end
    return X[:, 100:100:end]
end

sample_trj = score_integral(nnc, 5000000, 0.01, 4)
sample_trj_norm = [norm(sample_trj[:, i]) for i in 1:size(sample_trj)[2]]
plot(norm.(sample_trj_norm))

kde_samples = kde(sample_trj[1, :])
plot(kde_samples.x, kde_samples.density, color=:blue, label="Samples", xlims=(-3, 5))
plot!(kde_obs.x, kde_obs.density, color=:red, label="Observed")

##
function generate_numerical_response(model, dt, n_tau, n_ens, n_therm, F, eps, m, Obs)
    Dim = size(model(0.1, dt, F))[1]
    response_num_ens = zeros(Dim, n_tau, n_ens)
    X0 = zeros(Dim, n_ens)
    X0eps = zeros(Dim, n_ens)

    Threads.@threads for i in ProgressBar(1:n_ens)
        seed = abs(rand(Int))
        X0[:, i] = model(n_therm, dt, F, seed=seed)[:, end]
        X0eps[:, i] = copy(X0[:, i])
        X0eps[1, i] += eps
        seed = abs(rand(Int))
        X = model(n_tau*dt, dt, F, seed=seed, u0=X0[:, i])
        X_pert = model(n_tau*dt, dt, F, seed=seed, u0=X0eps[:, i])
        response_num_ens[:, :, i] = (Obs(X, m) - Obs(X_pert, m)) ./ eps
    end
    response_num = mean(response_num_ens, dims=3)

    R_num = zeros(Dim, n_tau)
    for t in 1:n_tau
        R_num[:, t] = sum(response_num[:, 1:t], dims=2) * dt
    end

    return R_num
end

function generate_linear_response(trj, dt, n_tau, m, Obs)
    Dim = size(trj, 1)
    response_lin = zeros(Dim, Dim, n_tau)
    invC0 = inv(cov(trj'))
    score_lin = zeros(Dim, size(trj)[2])
    for i in ProgressBar(1:size(trj)[2])
        score_lin[:, i] = invC0 * (trj[:, i] .- m)
    end

    Threads.@threads for i in ProgressBar(1:n_tau)
        response_lin[:, :, i] = cov(Obs(trj[:, i:end], m)', score_lin[:, 1:end-i+1]')
    end

    R_lin = zeros(Dim, Dim, n_tau)
    for t in 1:n_tau
        R_lin[:, :, t] = sum(response_lin[:, :, 1:t], dims=3) * dt
    end

    return R_lin
end

function generate_score_response(trj, score, n_tau, m, Obs)
    
    Dim = size(trj,1)
    score_gen = zeros(Dim, size(trj)[2])
    for i in ProgressBar(1:size(trj)[2])
        score_gen[:,i] = score(trj[:,i]) ./ 0.002
    end     

    response_gen = zeros(Dim,Dim,n_tau)
    Threads.@threads for i in ProgressBar(1:n_tau)
        response_gen[:,:,i] = cov(Obs(trj[:,i:end], m)', score_gen[:,1:end-i+1]')
    end

    R_gen = zeros(Dim,Dim,n_tau)
    for t in 1:n_tau
        R_gen[:,:,t] = sum(response_gen[:,:,1:t], dims=3)*dt
    end

    return R_gen
end

S_lorenz96 = std(u_lorenz96[1, :])

M_lorenz96 = mean(u_lorenz96, dims=2)
Obs_mean(x, m) = (x .- m)
Obs_std(x, m) = (x .- m) .^2
Dim = size(u_lorenz96, 1)
tau = 5
n_tau = tau * 100
n_ens = 1000
n_therm = 1000
eps = 0.01

R_lorenz96_num_mean = generate_numerical_response(simulate_lorenz96, dt, n_tau, n_ens, n_therm, F, eps, M_lorenz96, Obs_mean)
R_lorenz96_lin_mean = generate_linear_response(u_lorenz96[:, 1:1:end], dt, n_tau, M_lorenz96, Obs_mean)
R_lorenz96_gen_mean = generate_score_response(u_lorenz96, nnc, n_tau, M_lorenz96, Obs_mean)

R_lorenz96_num_std = generate_numerical_response(simulate_lorenz96, dt, n_tau, n_ens, n_therm, F, eps, M_lorenz96, Obs_std)
R_lorenz96_lin_std = generate_linear_response(u_lorenz96[:, 1:1:end], dt, n_tau, M_lorenz96, Obs_std)
R_lorenz96_gen_std = generate_score_response(u_lorenz96, nnc, n_tau, M_lorenz96, Obs_std)

##
h5write("lorenz96_F_$(F).h5", "R_lorenz96_num_mean", R_lorenz96_num_mean)
h5write("lorenz96_F_$(F).h5", "R_lorenz96_lin_mean", R_lorenz96_lin_mean)
h5write("lorenz96_F_$(F).h5", "R_lorenz96_num_std", R_lorenz96_num_std)
h5write("lorenz96_F_$(F).h5", "R_lorenz96_lin_std", R_lorenz96_lin_std)

##
#################### PLOTTING ##################
gr()


function gaussian_pdf(x::Float64, mean::Float64, variance::Float64)::Float64
    coefficient = 1 / sqrt(2 * π * variance)
    exponent = -((x - mean)^2) / (2 * variance)
    return coefficient * exp(exponent)
end

plt1 = plot(kde_lorenz96.x, kde_lorenz96.density, label="Lorenz 96", title="F = $F", ylabel="PDF", lw=2, leftmargin=5mm, xlims=(-3, 5))
plt1 = plot!(kde_samples.x, kde_samples.density, label="Generative", lw=2)
plt1 = plot!(kde_lorenz96.x, gaussian_pdf.(kde_lorenz96.x, 0.0, 1.0), label="Gaussian", lw=2)

plt2 = plot(dt:dt:n_tau*dt, abs.(R_lorenz96_num_mean[1, :]), label="Lorenz 96 numerical", lw=2, ylabel="Mean Response", legend=:bottomright)
plt2 = plot!(dt:dt:n_tau*dt, abs.(R_lorenz96_lin_mean[1, 1, :]), label="Lorenz 96 linear", lw=2)
plt2 = plot!(dt:dt:n_tau*dt, abs.(R_lorenz96_gen_mean[1, 1, :]), label="Generative", lw=2)

plt3 = plot(dt:dt:n_tau*dt, abs.(R_lorenz96_num_std[1, :]), label="", lw=2, ylabel="Variance Response")
plt3 = plot!(dt:dt:n_tau*dt, abs.(R_lorenz96_lin_std[1, 1, :]), label="", lw=2)
plt3 = plot!(dt:dt:n_tau*dt, abs.(R_lorenz96_gen_std[1, 1, :]), label="", lw=2)

plot(plt1, plt2, plt3, layout=(3, 1), size=(600, 900))

##
savefig("lorenz96_F_$(F).png")

