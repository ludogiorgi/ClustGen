include(pwd() * "/src/ClustGen.jl")

using Main.ClustGen: f_tilde, train_clustered, train_vanilla0, check_loss, sample
using Random, Distributions, Plots, Statistics, ProgressBars, KernelDensity
using FFTW, Interpolations, LinearAlgebra, Flux, Measures, HDF5
plotly()

# Lorenz 63 Model parameters
σ = 10.0
ρ = 28.0
β = 8/3

function lorenz63(u, σ, ρ, β)
    du = similar(u)
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
    return du
end

function simulate_lorenz63(T, dt, σ, ρ, β, sigma; seed=123, u0 = rand(3), res = 1)
    Random.seed!(seed)
    N = Int(T / dt)
    u = zeros(length(u0), Int(N/res))
    uOld = u0
    
    for t in 2:N
        du = lorenz63(uOld, σ, ρ, β) * dt + randn(length(u0)) .* sigma .* sqrt(dt)
        uNew = uOld + du
        if t % res == 0
            u[:, Int(t/res)] .= uNew
        end
        uOld = uNew
    end
    
    return u
end

function autocorrelation(x)
    x = x .- mean(x)
    acf = [cor(x[1:end-k], x[k+1:end]) for k in 0:1000]
    return acf
end

# Parameters
T = 10000
dt = 0.01

u_lorenz63 = simulate_lorenz63(T, dt, σ, ρ, β, 0.1)
u_lorenz63_n = (u_lorenz63 .- mean(u_lorenz63, dims=2)) ./ std(u_lorenz63, dims=2)

# Plot time series
tr = plot(u_lorenz63[1, 1:10000])

# KDE of normalized data
kde_lorenz63 = kde(u_lorenz63_n[1, :])

hist = plot(kde_lorenz63.x, kde_lorenz63.density, label="Lorenz 63", title="Histograms")

ac = plot(0:dt:1000*dt, autocorrelation(u_lorenz63[1, :]), title="Autocorrelations", xlabel="Lag", ylabel="Autocorrelation", label="")

plot(tr, hist, ac, layout=(3, 1), size=(800, 1200))
##
# Observation from longer simulation
T_obs = 1000000
obs = simulate_lorenz63(T_obs, dt, σ, ρ, β, sqrt(0.0); res=100)
obs = (obs .- minimum(obs, dims=2)) ./ (maximum(obs, dims=2) .- minimum(obs, dims=2))


kde_obs = kde(obs[1, :])
plt = plot(kde_obs.x, kde_obs.density, label="Observed", title="Observed", ylabel="PDF", lw=2, leftmargin=5mm)

scatter(obs[1, 1:50:end], obs[2, 1:50:end], obs[3, 1:50:end], markersize=1)

##
##################### CLUSTERING #####################

σ = 0.01
σ_values = [σ]
μ = repeat(obs, 1, 1)

averages_values, centers_values, Nc_values = f_tilde(σ_values, μ; prob=0.001, do_plot=true, conv_param=0.0001)

Nc_values[1]
##
####################### CLUSTERING CHECK #######################

k_norm = [norm(averages_values[1][:, i]) for i in 1:Nc_values[1]]
Plots.scatter(centers_values[1][1, 1:1:end], centers_values[1][2, 1:1:end], centers_values[1][3, 1:1:end], marker_z=k_norm[1:1:end], color=:viridis, markersize=1)

##
#################### TRAINING WITH CLUSTERING LOSS ####################

using Flux: gradient

function create_nn(neurons::Vector{Int}; activation = swish)
    layers = []
    push!(layers, Flux.Dense(neurons[1], neurons[2]))
    for i in 2:length(neurons)-2
        push!(layers, Flux.Dense(neurons[i], neurons[i+1], activation))
    end
    push!(layers, Flux.Dense(neurons[end-1], neurons[end]))
    return Flux.Chain(layers...)
end

function loss_score2(nn, data1, data2)
    return sum((nn(data1) .- data2).^10 ) #/ sum((data2).^10)
end

function train_clustered2(data, n_epochs, neurons; opt=ADAM(0.0001), activation=swish)
    nn = create_nn(neurons, activation=activation)
    losses = []
    for epoch in ProgressBar(1:n_epochs)
        loss = 0
        for i in eachindex(data)
            loss += loss_score2(nn, data[i][1], data[i][2])
            gs = gradient(Flux.params(nn)) do
                loss_score2(nn, data[i][1], data[i][2])
            end
            Flux.Optimise.update!(opt, Flux.params(nn), gs)
        end
        push!(losses, loss/length(data))
    end
    return nn, losses
end

data_clustered = []
data = Flux.Float32.(hcat([[centers_values[1][:, i]..., averages_values[1][:, i]...] for i in 1:Nc_values[1]]...))
push!(data_clustered, [(data[1:Dim, i], data[Dim+1:2*Dim, i]) for i in 1:Nc_values[1]])

data_clustered = vcat(data_clustered...)

nn_clustered, loss_clustered = train_clustered2(data_clustered, 100, [Dim, 100, 300, 100, Dim]; activation=swish, opt=ADAM(0.0001))
nnc(x) = nn_clustered(Flux.Float32.([x...]))[:] #.* (M_averages_values .- m_averages_values) .- m_averages_values
#l_check = check_loss(obs, nnc, σ)
#Plots.plot(loss_clustered)
#hline!([l_check])

nnc_x = [nnc(centers_values[1][:, i])[1] for i in 1:Nc_values[1]]
nnc_y = [nnc(centers_values[1][:, i])[2] for i in 1:Nc_values[1]]
nnc_z = [nnc(centers_values[1][:, i])[3] for i in 1:Nc_values[1]]

plt1 = scatter(centers_values[1][1, 1:1:end], averages_values[1][1, 1:1:end], label="Clustered", legend = false)
plt1 = scatter!(centers_values[1][1, 1:1:end], nnc_x, label="Clustered NN")

plt2 = scatter(centers_values[1][2, 1:1:end], averages_values[1][2, 1:1:end], label="Clustered", legend=false)
plt2 = scatter!(centers_values[1][2, 1:1:end], nnc_y, label="Clustered NN")

plt3 = scatter(centers_values[1][3, 1:1:end], averages_values[1][3, 1:1:end], label="Clustered", legend=false)
plt3 = scatter!(centers_values[1][3, 1:1:end], nnc_z, label="Clustered NN")

plot(plt1, plt2, plt3, layout=(3, 1), size=(400, 1200))


##

ii = rand(1:Nc_values[1])
println(nn_clustered(centers_values[1][:,ii]))
println(averages_values[1][1,ii])
##
loss_score2(nn_clustered, data_clustered)


loss_score3(nn_clustered, centers_values[1],  averages_values[1])

##
#################### INTERPOLATION ####################



##
#################### TRAINING WITH VANILLA LOSS ####################

Dim = 3
nn_vanilla, loss_vanilla = train_vanilla0(obs, 50, 128, [Dim, 32, 128, 32, Dim], σ; activation=swish)
nnv(x) = nn_vanilla(Flux.Float32.([x...]))[:]
Plots.plot(loss_vanilla)
# Plots.hline!([l_check])

##
####################### NNs COMPARISON ########################

av = [(averages_values[1][:,i])[2] for i in 1:Nc_values[1]]
av_c = [(nnc(centers_values[1][:,i]))[2] for i in 1:Nc_values[1]]
av_v = [norm(nnv(centers_values[1][:,i])) for i in 1:Nc_values[1]]

plt1 = Plots.scatter(centers_values[1][1, 1:1:end], centers_values[1][2, 1:1:end], centers_values[1][3, 1:1:end], marker_z=av[1:1:end], color=:viridis, markersize=3, title="Clustered", label="")
plt2 = scatter(centers_values[1][1, 1:1:end], centers_values[1][2, 1:1:end], centers_values[1][3, 1:1:end], marker_z=av_c, color=:viridis, markersize=3, title="Clustered NN", label="")
plt3 = scatter(centers_values[1][1, 1:1:end], centers_values[1][2, 1:1:end], centers_values[1][3, 1:1:end], marker_z=av_v, color=:viridis, markersize=3, title="Vanilla", label="")
plot(plt1, plt2, plt3, layout=(3, 1), size=(400, 1200), cbar=false)

##
######################## SAMPLES GENERATION ########################

function score_integral(score, N, dt, Dim)
    X = zeros(Dim, N)
    r = randn(Dim, N) .* sqrt(dt)
    for i in ProgressBar(2:N-1)
        X[:, i+1] .= X[:, i] .+ score(X[:, i]) .* dt ./ σ .+ sqrt(2) .* r[:, i]
        if norm(X[:, i+1]) > 7
            X[:, i+1] .= 0.0
        end
    end
    return X[:, 100:100:end]
end

sample_trj = score_integral(nnc, 1000000, 0.01, 3)
sample_trj_norm = [norm(sample_trj[:, i]) for i in 1:size(sample_trj)[2]]
plot(sample_trj[1,1:100])

sc1 = scatter(sample_trj[1,1:1:end], sample_trj[2,1:1:end], sample_trj[3,1:1:end], markersize=1)
sc2 = scatter(obs[1,1:50:end], obs[2,1:50:end], obs[3,1:50:end], markersize=1)
plot(sc1, sc2)
##
kde_samples = kde(sample_trj[1, :])
plot(kde_samples.x, kde_samples.density, color=:blue, label="Samples")
plot!(kde_obs.x, kde_obs.density, color=:red, label="Observed")

##
function generate_numerical_response(model, dt, n_tau, n_ens, n_therm, σ, ρ, β, eps, m, Obs)
    Dim = size(model(0.1, dt, σ, ρ, β))[1]
    response_num_ens = zeros(Dim, n_tau, n_ens)
    X0 = zeros(Dim, n_ens)
    X0eps = zeros(Dim, n_ens)

    Threads.@threads for i in ProgressBar(1:n_ens)
        seed = abs(rand(Int))
        X0[:, i] = model(n_therm, dt, σ, ρ, β, seed=seed)[:, end]
        X0eps[:, i] = copy(X0[:, i])
        X0eps[1, i] += eps
        seed = abs(rand(Int))
        X = model(n_tau*dt, dt, σ, ρ, β, seed=seed, u0=X0[:, i])
        X_pert = model(n_tau*dt, dt, σ, ρ, β, seed=seed, u0=X0eps[:, i])
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

S_lorenz63 = std(u_lorenz63[1, :])

M_lorenz63 = mean(u_lorenz63, dims=2)
Obs_mean(x, m) = (x .- m)
Obs_std(x, m) = (x .- m) .^2
Dim = size(u_lorenz63, 1)
tau = 5
n_tau = tau * 100
n_ens = 1000
n_therm = 1000
eps = 0.01

R_lorenz63_num_mean = generate_numerical_response(simulate_lorenz63, dt, n_tau, n_ens, n_therm, σ, ρ, β, eps, M_lorenz63, Obs_mean)
R_lorenz63_lin_mean = generate_linear_response(u_lorenz63[:, 1:1:end], dt, n_tau, M_lorenz63, Obs_mean)
R_lorenz63_gen_mean = generate_score_response(u_lorenz63, nnc, n_tau, M_lorenz63, Obs_mean)

R_lorenz63_num_std = generate_numerical_response(simulate_lorenz63, dt, n_tau, n_ens, n_therm, σ, ρ, β, eps, M_lorenz63, Obs_std)
R_lorenz63_lin_std = generate_linear_response(u_lorenz63[:, 1:1:end], dt, n_tau, M_lorenz63, Obs_std)
R_lorenz63_gen_std = generate_score_response(u_lorenz63, nnc, n_tau, M_lorenz63, Obs_std)

##
h5write("lorenz63.h5", "R_lorenz63_num_mean", R_lorenz63_num_mean)
h5write("lorenz63.h5", "R_lorenz63_lin_mean", R_lorenz63_lin_mean)
h5write("lorenz63.h5", "R_lorenz63_num_std", R_lorenz63_num_std)
h5write("lorenz63.h5", "R_lorenz63_lin_std", R_lorenz63_lin_std)

##
#################### PLOTTING ##################
gr()


function gaussian_pdf(x::Float64, mean::Float64, variance::Float64)::Float64
    coefficient = 1 / sqrt(2 * π * variance)
    exponent = -((x - mean)^2) / (2 * variance)
    return coefficient * exp(exponent)
end

plt1 = plot(kde_lorenz63.x, kde_lorenz63.density, label="Lorenz 63", title="Lorenz 63", ylabel="PDF", lw=2, leftmargin=5mm, xlims=(-3, 5))
plt1 = plot!(kde_samples.x, kde_samples.density, label="Generative", lw=2)
plt1 = plot!(kde_lorenz63.x, gaussian_pdf.(kde_lorenz63.x, 0.0, 1.0), label="Gaussian", lw=2)

plt2 = plot(dt:dt:n_tau*dt, abs.(R_lorenz63_num_mean[1, :]), label="Lorenz 63 numerical", lw=2, ylabel="Mean Response", legend=:bottomright)
plt2 = plot!(dt:dt:n_tau*dt, abs.(R_lorenz63_lin_mean[1, 1, :]), label="Lorenz 63 linear", lw=2)
plt2 = plot!(dt:dt:n_tau*dt, abs.(R_lorenz63_gen_mean[1, 1, :]), label="Generative", lw=2)

plt3 = plot(dt:dt:n_tau*dt, abs.(R_lorenz63_num_std[1, :]), label="", lw=2, ylabel="Variance Response")
plt3 = plot!(dt:dt:n_tau*dt, abs.(R_lorenz63_lin_std[1, 1, :]), label="", lw=2)
plt3 = plot!(dt:dt:n_tau*dt, abs.(R_lorenz63_gen_std[1, 1, :]), label="", lw=2)

plot(plt1, plt2, plt3, layout=(3, 1), size=(600, 900))

##
savefig("lorenz63.png")
