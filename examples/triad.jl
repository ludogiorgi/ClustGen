
include(pwd() * "/src/ClustGen.jl")

using Main.ClustGen: f_tilde, train_clustered, train_vanilla, check_loss, sample
using Random, Distributions, Plots, Statistics, ProgressBars, KernelDensity
using FFTW, Interpolations, LinearAlgebra, Flux, Measures, HDF5
plotly()

# Triad Model parameters
L11 = -2.0
L12 = 0.2
L13 = 0.1
g2 = 0.6
g3 = 0.4
s2 = 1.2
s3 = 0.8
F = 0.0

function simulate_triad_model(T, dt, ε, I; seed=123, u0 = [0.0, 0.0, 0.0])
    Random.seed!(seed)
    N = Int(T / dt)
    u = zeros(3, N)
    W2 = sqrt(dt) * randn(N)
    W3 = sqrt(dt) * randn(N)

    u[:, 1] = u0
    
    for t in 2:N
        u1, u2, u3 = @view u[:, t-1]
        du1 = (L11 * u1 + L12 * u2 + L13 * u3 + I * u1 * u2) * dt
        du2 = (-L12 * u1 - I * u1^2 - g2 * u2 / ε) * dt + s2 / sqrt(ε) * W2[t-1]
        du3 = (-L13 * u1 - g3 * u3 / ε) * dt + s3 / sqrt(ε) * W3[t-1]
        u[:, t] = [u1 + du1, u2 + du2, u3 + du3]
    end
    
    return u
end

function simulate_reduced_model(T, dt, ε, I; seed=123, u0 = [0.0])

    # Coefficients of the reduced model
    a = L11 + ε * ( (I^2 * s2^2) / (2 * g2^2) - (L12^2) / g2 - (L13^2) / g3 )
    b = -2 * (L12 * I) / (g2) * ε
    c = (I^2) / (g2) * ε
    B = -(I * s2) / (g2) * sqrt(ε)
    A = -(L12 * B) / I
    F_tilde = F + (A * B) / 2   
    s = (L13 * s3) / g3 * sqrt(ε)

    Random.seed!(seed)
    N = Int(T / dt)
    u = zeros(1,N)
    W2 = sqrt(dt) * randn(N)
    W3 = sqrt(dt) * randn(N)
    u[:,1] = u0
    
    for t in 2:N
        u1 = u[1,t-1]
        du1 = (-F_tilde + a * u1 + b * u1^2 - c * u1^3) *dt + (A - B * u1) * W2[t-1] + s * W3[t-1] 
        #du1 = (F_tilde + a * u1 + b * u1^2 - c * u1^3) *dt + (A - B * u1) * W2[t-1] + s * W3[t-1] 
        u[:,t] = [u1 + du1]
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
ϵ = 0.1
I = 1.0

u_triad = simulate_triad_model(T, dt, ϵ, I)
u_triad_n = (u_triad .- mean(u_triad, dims=2)) ./ std(u_triad, dims=2)
u_reduced = simulate_reduced_model(T, dt, ϵ, I)
u_reduced_n = (u_reduced.- mean(u_reduced, dims=2) ) ./ std(u_reduced, dims=2)

tr = plot(dt:dt:50, u_triad[1,1:5000])
tr = plot!(dt:dt:50, u_reduced[1,1:5000])

kde_triad = kde(u_triad_n[1,:])
kde_reduced = kde(u_reduced_n[1,:])

hist = plot(kde_triad.x, kde_triad.density, label="Triad", title="Histograms for ϵ = $ϵ")
hist = plot!(kde_reduced.x, kde_reduced.density, label="Reduced")

ac = plot(0:dt:1000*dt, autocorrelation(u_triad[1,:]), title="Autocorrelations for ϵ = $ϵ", xlabel="Lag", ylabel="Autocorrelation", label="")
ac = plot!(0:dt:1000*dt, autocorrelation(u_reduced[1,:]), label="")

plot(tr, hist, ac, layout=(3,1), size=(800,1200))

##
T_obs = 1000000
obs = simulate_triad_model(T_obs, dt, ϵ, I)
obs = obs[:,1:100:end]
obs = (obs .- mean(obs, dims=2)) ./ std(obs, dims=2)
u_triad_n = (u_triad .- mean(u_triad, dims=2)) ./ std(u_triad, dims=2)
kde_obs = kde(u_triad_n[1,:])
interp_P = interpolate((kde_obs.x,), kde_obs.density, Gridded(Linear()))
extrp_P = extrapolate(interp_P, Flat())
plt = plot(-3:0.01:5, extrp_P(-3:0.01:5), color=:red, label="Observed")
display(plt)

##
##################### CLUSTERING #####################

variance_exploding(t; σ_min = 0.002, σ_max = 5.0) = @. σ_min * (σ_max/σ_min)^t
g(t; σ_min=0.01, σ_max=5.0) = σ_min * (σ_max/σ_min)^t * sqrt(2*log(σ_max/σ_min))

n_diffs = 1
diff_times = [0.0] #[i/n_diffs for i in 1:n_diffs]
σ_values = variance_exploding.(diff_times)
μ = repeat(obs, 1, 1)

averages_values, centers_values, Nc_values = f_tilde(σ_values, μ; prob=0.002 , do_plot=true, conv_param=0.002  )
##
####################### CLUSTERING CHECK #######################

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
    data_t = Flux.Float32.(hcat([[centers_values[t][:,i]..., averages_values_norm[t][:,i]...] for i in 1:Nc_values[t]]...))
    push!(data_clustered, [(data_t[1:Dim, i], data_t[Dim+1:2*Dim, i]) for i in 1:Nc_values[t]])
end
data_clustered = vcat(data_clustered...)

nn_clustered, loss_clustered = train_clustered(data_clustered, 1000, 128, [Dim, 128, 64, Dim]; activation=tanh)
nnc(x) = .- nn_clustered(Flux.Float32.([x...]))[:] .* (M_averages_values .- m_averages_values) .- m_averages_values
Plots.plot(loss_clustered)
##
######################## SAMPLES GENERATION ########################

function score_integral(score, N, dt, Dim)
    X = zeros(Dim,N)
    r = randn(Dim,N) .* sqrt(dt)
    for i in ProgressBar(2:N-1)
        X[:,i+1] .= X[:,i] .+ score(X[:,i]) .* dt ./ 0.002 .+ sqrt(2) .* r[:,i]
        if norm(X[:,i+1]) > 5
            X[:,i+1] .= 0.0
        end
    end
    return X[:,100:100:end]
end

sample_trj = score_integral(nnc, 1000000, 0.01, 3)
plot(sample_trj[1,1:10000])
##
kde_samples = kde(sample_trj[1,:])
plot(kde_samples.x, kde_samples.density, color=:blue, label="Samples")
plot!(kde_obs.x, kde_obs.density, color=:red, label="Observed")

##

function generate_numerical_response(model, dt, n_tau, n_ens, n_therm, ϵ, I, eps, m, Obs)
    if model == simulate_triad_model
        Dim = 3
    elseif model == simulate_reduced_model
        Dim = 1
    end
    response_num_ens = zeros(Dim,n_tau,n_ens)
    X0 = zeros(Dim,n_ens)
    X0eps = zeros(Dim,n_ens)

    Threads.@threads for i in ProgressBar(1:n_ens)
        seed = abs(rand(Int))
        X0[:,i] = model(n_therm, dt, ϵ, I,seed=seed)[:,end] 
        X0eps[:,i] = copy(X0[:,i])
        X0eps[1,i] += eps
        seed = abs(rand(Int))
        X = model(tau, 0.01, ϵ, I, seed=seed,u0=X0[:,i])
        X_pert = model(tau, 0.01, ϵ, I, seed=seed,u0=X0eps[:,i])
        response_num_ens[:,:,i] = (Obs(X,m) - Obs(X_pert,m)) ./eps
    end
    response_num = mean(response_num_ens, dims=3)

    R_num = zeros(Dim, n_tau)
    for t in 1:n_tau
        R_num[:,t] = sum(response_num[:,1:t], dims=2)*dt
    end

    return R_num
end

function generate_linear_response(trj, dt, n_tau, m, Obs)
    Dim = size(trj,1)
    response_lin = zeros(Dim,Dim,n_tau)
    invC0 = inv(cov(trj'))
    score_lin = zeros(Dim,size(trj)[2])
    for i in ProgressBar(1:size(trj)[2])
        score_lin[:,i] = invC0*(trj[:,i] .- m)
    end

    Threads.@threads for i in ProgressBar(1:n_tau)
        response_lin[:,:,i] = cov(Obs(trj[:,i:end], m)', score_lin[:,1:end-i+1]')
    end

    R_lin = zeros(Dim,Dim,n_tau)
    for t in 1:n_tau
        R_lin[:,:,t] = sum(response_lin[:,:,1:t], dims=3)*dt
    end

    return R_lin
end

function generate_analytical_response(trj, dt, ε, I, m, Obs)
    a = L11 + ε * ( (I^2 * s2^2) / (2 * g2^2) - (L12^2) / g2 - (L13^2) / g3 )
    b = -2 * (L12 * I) / (g2) * ε
    c = (I^2) / (g2) * ε
    B = -(I * s2) / (g2) * sqrt(ε)
    A = -(L12 * B) / I  
    s = (L13 * s3) / g3 * sqrt(ε)
    F_tilde = F + (A * B) / 2 
    score(x) = -2* ((A*B/2 + F) + (a-B^2)*x + b*x^2 - c*x^3) / (s^2+(A-B*x)^2)
    Dim = size(trj,1)
    score_an = zeros(Dim, size(trj)[2])
    for i in ProgressBar(1:size(trj)[2])
        score_an[:,i] = score.(trj[:,i])
    end     

    response_an = zeros(Dim,Dim,n_tau)
    Threads.@threads for i in ProgressBar(1:n_tau)
        response_an[:,:,i] = cov(Obs(trj[:,i:end], m)', score_an[:,1:end-i+1]')
    end

    R_an = zeros(Dim,Dim,n_tau)
    for t in 1:n_tau
        R_an[:,:,t] = sum(response_an[:,:,1:t], dims=3)*dt
    end

    return R_an
end

function generate_analytical_response(trj, score, n_tau, m, Obs)
    
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

S_triad = std(u_triad[1,:])

M_triad = mean(u_triad, dims=2)
M_reduced = mean(u_reduced, dims=2)
Obs_mean(x, m) = (x .- m)
Obs_std(x, m) = (x .- m) .^2
Dim = 3
tau = 5
n_tau = tau*100
n_ens = 1000
n_therm = 1000
eps = 0.01
dt = 0.01


R_triad_num_mean = generate_numerical_response(simulate_triad_model, dt, n_tau, n_ens, n_therm, ϵ, I, eps, M_triad, Obs_mean)
R_reduced_num_mean = generate_numerical_response(simulate_reduced_model, dt, n_tau, n_ens, n_therm, ϵ, I, eps, M_reduced, Obs_mean)
R_triad_lin_mean = generate_linear_response(u_triad[:,1:1:end], dt, n_tau, M_triad, Obs_mean)
R_reduced_lin_mean = generate_linear_response(u_reduced[:,1:1:end], dt, n_tau, M_reduced, Obs_mean)
R_reduced_an_mean = generate_analytical_response(u_reduced[:,1:1:end], dt, ϵ, I, M_reduced, Obs_mean)
R_triad_gen_mean = generate_analytical_response(u_triad_n, nnc, n_tau, zeros(3), Obs_mean)
# R_reduced_gen_mean = generate_analytical_response(u_reduced_n, nnc, n_tau, zeros(1), Obs_mean)

R_triad_num_std = generate_numerical_response(simulate_triad_model, dt, n_tau, n_ens, n_therm, ϵ, I, eps, M_triad, Obs_std)
R_reduced_num_std = generate_numerical_response(simulate_reduced_model, dt, n_tau, n_ens, n_therm, ϵ, I, eps, M_reduced, Obs_std)
R_triad_lin_std = generate_linear_response(u_triad[:,1:1:end], dt, n_tau, M_triad, Obs_std)
R_reduced_lin_std = generate_linear_response(u_reduced[:,1:1:end], dt, n_tau, M_reduced, Obs_std)
R_reduced_an_std = generate_analytical_response(u_reduced[:,1:1:end], dt, ϵ, I, M_reduced, Obs_std)
R_triad_gen_std = generate_analytical_response(u_triad_n, nnc, n_tau, zeros(3), Obs_std)
# R_reduced_gen_std = generate_analytical_response(u_reduced_n, nnc, n_tau, zeros(1), Obs_std)

##
ϵ
h5write("ϵ_$(ϵ)_I_$(I).h5", "R_triad_num_mean", R_triad_num_mean)
h5write("ϵ_$(ϵ)_I_$(I).h5", "R_reduced_num_mean", R_reduced_num_mean)
h5write("ϵ_$(ϵ)_I_$(I).h5", "R_triad_lin_mean", R_triad_lin_mean)
h5write("ϵ_$(ϵ)_I_$(I).h5", "R_reduced_lin_mean", R_reduced_lin_mean)
h5write("ϵ_$(ϵ)_I_$(I).h5", "R_reduced_an_mean", R_reduced_an_mean)
h5write("ϵ_$(ϵ)_I_$(I).h5", "R_triad_gen_mean", R_triad_gen_mean)
# h5write("ϵ_$(ϵ)_I_$(I).h5", "R_reduced_gen_mean", R_reduced_gen_mean)

h5write("ϵ_$(ϵ)_I_$(I).h5", "R_triad_num_std", R_triad_num_std)
h5write("ϵ_$(ϵ)_I_$(I).h5", "R_reduced_num_std", R_reduced_num_std)
h5write("ϵ_$(ϵ)_I_$(I).h5", "R_triad_lin_std", R_triad_lin_std)
h5write("ϵ_$(ϵ)_I_$(I).h5", "R_reduced_lin_std", R_reduced_lin_std)
h5write("ϵ_$(ϵ)_I_$(I).h5", "R_reduced_an_std", R_reduced_an_std)
h5write("ϵ_$(ϵ)_I_$(I).h5", "R_triad_gen_std", R_triad_gen_std)
# h5write("ϵ_$(ϵ)_I_$(I).h5", "R_reduced_gen_std", R_reduced_gen_std)
##

h5read("ϵ_$(ϵ)_I_$(I).h5", "R_triad_num_mean")
h5read("ϵ_$(ϵ)_I_$(I).h5", "R_reduced_num_mean")
h5read("ϵ_$(ϵ)_I_$(I).h5", "R_triad_lin_mean")
h5read("ϵ_$(ϵ)_I_$(I).h5", "R_reduced_lin_mean")
h5read("ϵ_$(ϵ)_I_$(I).h5", "R_reduced_an_mean")
h5read("ϵ_$(ϵ)_I_$(I).h5", "R_triad_gen_mean")
# h5read("ϵ_$(ϵ)_I_$(I).h5", "R_reduced_gen_mean")

h5read("ϵ_$(ϵ)_I_$(I).h5", "R_triad_num_std")
h5read("ϵ_$(ϵ)_I_$(I).h5", "R_reduced_num_std")
h5read("ϵ_$(ϵ)_I_$(I).h5", "R_triad_lin_std")
h5read("ϵ_$(ϵ)_I_$(I).h5", "R_reduced_lin_std")
h5read("ϵ_$(ϵ)_I_$(I).h5", "R_reduced_an_std")
h5read("ϵ_$(ϵ)_I_$(I).h5", "R_triad_gen_std")
# h5read("ϵ_$(ϵ)_I_$(I).h5", "R_reduced_gen_std")
##
#################### PLOTTING ##################
gr()

function gaussian_pdf(x::Float64, mean::Float64, variance::Float64)::Float64
    coefficient = 1 / sqrt(2 * π * variance)
    exponent = -((x - mean)^2) / (2 * variance)
    return coefficient * exp(exponent)
end

plt1 = plot(kde_triad.x, kde_triad.density, label="Triad", title="ϵ = $ϵ, I = $I", ylabel="PDF", lw=2, leftmargin=5mm)
plt1 = plot!(kde_reduced.x, kde_reduced.density, label="Reduced", lw=2)
plt1 = plot!(kde_samples.x, kde_samples.density, label="Generative", lw=2)
plt1 = plot!(kde_reduced.x, gaussian_pdf.(kde_reduced.x, 0.0, 1.0), label="Gaussian", lw=2)

plt2 = plot(dt:dt:n_tau*dt, abs.(R_triad_num_mean[1,:]), label="Triad numerical", lw=2, ylabel="Mean Response", legend=:bottomright)
plt2 = plot!(dt:dt:n_tau*dt, abs.(R_reduced_num_mean[1,:]), label="Reduced numerical", lw=2)
plt2 = plot!(dt:dt:n_tau*dt, abs.(R_triad_lin_mean[1,1,:]), label="Triad linear", lw=2)
plt2 = plot!(dt:dt:n_tau*dt, abs.(R_reduced_lin_mean[1,1,:]), label="Reduced linear", lw=2)
plt2 = plot!(dt:dt:n_tau*dt, abs.(R_reduced_an_mean[1,1,:]), label="Reduced analytical", lw=2)
plt2 = plot!(dt:dt:n_tau*dt, abs.(R_triad_gen_mean[1,1,:]), label="Triad generative", lw=2)
#plt2 = plot!(dt:dt:n_tau:dt, abs.(R_reduced_gen_mean[1,1,:]), label="Reduced generative")

plt3 = plot(dt:dt:n_tau*dt, abs.(R_triad_num_std[1,:]), label="", lw=2, ylabel="Variance Response")
plt3 = plot!(dt:dt:n_tau*dt, abs.(R_reduced_num_std[1,:]), label="", lw=2)
plt3 = plot!(dt:dt:n_tau*dt, abs.(R_triad_lin_std[1,1,:]), label="", lw=2)
plt3 = plot!(dt:dt:n_tau*dt, abs.(R_reduced_lin_std[1,1,:]), label="", lw=2)
plt3 = plot!(dt:dt:n_tau*dt, abs.(R_reduced_an_std[1,1,:]), label="", lw=2)
plt3 = plot!(dt:dt:n_tau*dt, abs.(R_triad_gen_std[1,1,:]) .* S_triad, label="", lw=2)
#plt3 = plot!(dt:dt:n_tau*dt, abs.(R_reduced_gen_std[1,1,:]) .* S_triad, label="Reduced generative")

plot(plt1, plt2, plt3, layout=(3,1), size=(600,900))

##
savefig("triad_ϵ_$(ϵ)_I_$(I).png")