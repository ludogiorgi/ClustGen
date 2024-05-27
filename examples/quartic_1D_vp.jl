include(pwd() * "/src/ClustGen.jl")

using Main.ClustGen: f_tilde, train_clustered, train_vanilla, check_loss, sample
using KernelDensity
using HDF5
using Flux, Random, KernelDensity
using Plots, LinearAlgebra
using Statistics
using ProgressBars, QuadGK, ParallelKMeans,Distances, Interpolations

function play_beep()
    run(`say "Code execution finished"`)
end
##################### DATA GENERATION #####################

function evolve(x0, system, timesteps, Δt, res, σ)
    x_f = Array{Float64,2}(undef, 1, timesteps*res)
    x_f[:, 1] = x0
    for i = ProgressBar(2:timesteps*res)
        x_old = x_f[:, i-1]
        k = system(x_old)
        x_f[:, i] = x_old + Δt*k + σ * randn(1) * sqrt(Δt)
    end
    return x_f[:, 1:res:end]
end

forcing(x) = @. x * (1 - x^2)

evolve(x0, forcing, timesteps, σ; Δt=0.01, res=100) = evolve(x0, forcing, timesteps, Δt, res, σ)

b = 0.5
b2half = b^2/2
x0 = [1.0]
sol = evolve(x0, forcing, 1000000, b) # It works well also with 100000 timesteps
U(x) = - x^2/2 + x^4/4
nrm, _ = quadgk(x -> exp(-U(x)/b2half), -10, 10)
P(x) = exp(-U.(x)/b2half) / nrm
kde_sol = kde(sol[1,:])
plt = plot(kde_sol.x, kde_sol.density, color=:blue, label="Observed")
plt = plot!(kde_sol.x, P.(kde_sol.x), color=:red, label="True")
display(plt)
##
######################### DATA NORMALIZATION #########################
function normalize_to_01(series)
    min_val = minimum(series, dims=2)
    max_val = maximum(series, dims=2)
    return (series .- min_val) ./ (max_val - min_val)
end

obs = normalize_to_01(sol)

kde_obs = kde(obs[1,:])
interp_P = interpolate((kde_obs.x,), kde_obs.density, Gridded(Linear()))
extrp_P = extrapolate(interp_P, Flat())
plt = plot(-0.2:0.01:1.2, extrp_P(-0.2:0.01:1.2), color=:red, label="True")
##
#########################  #########################

function evolve_f(x0, n_diffs, f, g, diff_stop)
    dt = 1.0 / n_diffs
    x_f = Array{Float64,2}(undef, 1, diff_stop)
    x_f[:, 1] = x0
    for i = 2:diff_stop
        t_diff = i * dt
        x_old = x_f[:, i-1]
        x_f[:, i] = x_old - f(t_diff) * (x_old .- 0.5) * dt + g(t_diff) * randn(1) * sqrt(dt)
    end
    return x_f
end

STD = std(obs[1,:])
C0 = 0.01
f1 = 6
f0 = (f1 * log(1 / C0)) / (-1 + exp(f1))

f(t) = f0*exp(f1*t)
g(t) = sqrt(2*f(t)) * STD

m(t) = exp((f0 - exp(f1 * t) * f0) / f1)
σ2(t) = (1 - exp(-((2 * (-1 + exp(f1 * t)) * f0) / f1))) * STD^2
σ(t) = sqrt(σ2(t))


# f0 = 1
# f1 = 1
# fP = 1

# f(t) = f0 + f1*t^fP
# g(t) = sqrt(2*f(t)) * STD

# m(t) = exp(-f0 * t - (f1 * t^(1 + fP)) / (1 + fP))
# σ2(t) = (1 - exp(-2 * t * (f0 + (f1 * t^fP) / (1 + fP)))) * STD^2
# σ(t) = sqrt(σ2(t))

n_diffs = 1000
n_ens = 10000
diff_stop = n_diffs
ens_f = zeros(n_ens, diff_stop)

for i in ProgressBar(1:n_ens)
    start_f = [obs[1,rand(1:length(obs))]]
    ens_f[i,:] = evolve_f(start_f, n_diffs, f, g, diff_stop)
end
sigmas = zeros(diff_stop)
for i in 1:diff_stop
    sigmas[i] = std(ens_f[:,i])
end
plot(1:diff_stop, sigmas)
##
plt = plot()
for i in 1:100
    plot!(plt, 1:diff_stop, ens_f[i,:], color=:blue, legend=false, alpha=0.3)
end
display(plt)
##
G(x,mu,sigma2) = 1/sqrt(2*π*sigma2) * exp(-(x-mu)^2/2/sigma2)

kde_0 = kde(ens_f[:,1])
kde_1 = kde(ens_f[:,end])

plot(kde_0.x, kde_0.density, color=:blue, label="Initial")
plot!(kde_1.x, kde_1.density, color=:red, label="Final")
plot!(kde_1.x, G.(kde_1.x, 0.5, STD^2), color=:green, label="Gaussian")
##

function generate_xz(x0, t)
    mt = m(t)
    σ2t = σ2(t)
    σt = sqrt(σ2t)
    z = randn!(similar(x0))
    x = @. x0 * mt + σt * z + (1-mt)*0.5
    return x, z
end

function calculate_averages(X, z, x)
    Ndim, Nz = size(z)
    Nc = maximum(X)
    averages = zeros(Ndim, Nc)
    centers = zeros(Ndim, Nc)
    z_sum = zeros(Ndim, Nc)
    x_sum = zeros(Ndim, Nc)
    z_count = zeros(Ndim, Nc)
    for i in 1:Nz
        segment_index = X[i]
        for dim in 1:Ndim
            z_sum[dim, segment_index] += z[dim, i]
            x_sum[dim, segment_index] += x[dim, i]
            z_count[dim, segment_index] += 1
        end
    end
    for dim in 1:Ndim
        for i in 1:Nc
            if z_count[dim, i] != 0
                averages[dim, i] = z_sum[dim, i] / z_count[dim, i]
                centers[dim, i] = x_sum[dim, i] / z_count[dim, i]
            end
        end
    end
    return averages, centers
end

function f_tilde_t(t, μ; prob = 0.001, do_plot=false, conv_param=1e-1, i_max = 150)
    if do_plot==true
        plt = plot()
        D_avr = []
    end
    method = Tree(false, prob)
    x, z = generate_xz(μ, t)
    state_space_partitions = StateSpacePartition(x; method = method)
    Nc = maximum(state_space_partitions.partitions)
    labels = [state_space_partitions.embedding(x[:,i]) for i in 1:size(x)[2]]
    averages, centers = calculate_averages(labels, z, x)
    averages_old, centers_old = averages, centers
    D_avr_temp = 1
    i = 1
    while D_avr_temp > conv_param && i < i_max
        x, z = generate_xz(μ, t)
        labels = [state_space_partitions.embedding(x[:,i]) for i in 1:size(x)[2]]
        averages, centers = calculate_averages(labels, z, x)
        averages_new = (averages .+ i .* averages_old) ./ (i+1)
        centers_new = (centers .+ i .* centers_old) ./ (i+1)
        D_avr_temp = mean(abs2, averages_new .- averages_old) / mean(abs2, averages_new)
        if do_plot==true
            push!(D_avr, D_avr_temp)
            scatter!(plt, D_avr, label="Averages", color=:red, legend=false)
            display(plt)
        end
        averages_old, centers_old = averages_new, centers_new
        i += 1
    end
    return averages_old, centers_old, Nc
end

function f_tilde2(t_values, μ; prob = 0.001, do_plot=false, conv_param=1e-1, i_max = 150)
    averages_values = []
    centers_values = []
    Nc_values = []
    for i in eachindex(t_values)
        t_diff = t_values[i]
        averages, centers, Nc = f_tilde_t(t_diff, μ; prob=prob, do_plot=do_plot, conv_param=conv_param, i_max=i_max)
        push!(averages_values, averages)
        push!(centers_values, centers)
        push!(Nc_values, Nc)
    end
    return averages_values, centers_values, Nc_values
end

n_diffs = 20
diff_times = [i/n_diffs for i in 1:n_diffs]
μ = repeat(obs, 1, 1)

averages_values, centers_values, Nc_values = f_tilde2(diff_times, μ; prob=0.02, do_plot=false, conv_param=0.00001)

##
function f_true(x,t; tol=1e-15)
    mt = m(t)
    σ2t = σ2(t)
    σt = sqrt(σ2t)
    ext = σt * 5
    num, _ = quadgk(z -> z * G(z,0.0,σ2t) * extrp_P((x - z - (1-mt)*0.5)/mt), - ext, ext, atol=tol, rtol=tol)
    den, _ = quadgk(z -> G(z,0.0,σ2t) * extrp_P((x - z - (1-mt)*0.5)/mt), - ext, ext, atol=tol, rtol=tol)
    return - num / den / σt
end

index = 1
diff_t = diff_times[index]
Ndim, Nz = size(obs)
mt = m(diff_t)
σ2t = σ2(diff_t)
σt = σ(diff_t)

z = randn!(similar(obs))
x = @. obs * mt + σt * z + (1-mt)*0.5

averages_true = [f_true.(centers_values[index][i], diff_t) for i in 1:Nc_values[index]]

xax = [0:0.05:1...]
yax =  f_true.(xax, diff_t)

scatter(x[1,1:10000], .- z[1,1:10000], color=:blue, legend=false, markerstrokewidth=0,markersize=0.5)
plot!(xax, yax, color=:black, linewidth=3)
scatter!(centers_values[index][:], averages_true, color=:red, label="True")
scatter!(centers_values[index][:], .- averages_values[index][:], color=:blue, label="Observed")
##
rmse = 0.0
for i in 1:n_diffs
    rmse += mean(abs2, f_true.(centers_values[i], diff_times[i]) .+ averages_values[i])
end
rmse /= n_diffs
println("RMSE: ", rmse)
##
####################### TRAINING WITH CLUSTERING LOSS #######################

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

nn_clustered, loss_clustered = train_clustered(data_clustered, 6000, 16, [Dim+1, 64, 64, Dim]; activation=tanh, opt=ADAM(0.001))
nnc(x, t) = .- nn_clustered(Flux.Float32.([x..., t]))[:] .* (M_averages_values - m_averages_values) .- m_averages_values
score_clustered(x, t) = nnc(x, t) ./ σ(t)
Plots.plot(loss_clustered)
Plots.hline!([rmse])
play_beep()
##
####################### TRAINING WITH VANILLA LOSS #######################

using Flux

function generate_txz(x0, m, σ2; ϵ=0.05)
    t = rand!(similar(x0, size(x0)[1])) .* (1 - ϵ) .+ ϵ
    mt = m.(t)
    σt = sqrt.(σ2.(t))
    z = randn!(similar(x0))
    x = @. x0 * mt + σt * z + (1-mt)*0.5
    return t, x, z
end

function create_nn(neurons::Vector{Int}; activation = swish)
    layers = []
    for i in 1:length(neurons)-1
        push!(layers, Flux.Dense(neurons[i], neurons[i+1], activation))
    end
    return Flux.Chain(layers...)
end

function loss_score(nn, batch)
    l = 0.0
    for i in eachindex(batch)
        x, y = batch[i]
        l +=  Flux.mse(nn(x), y)
    end
    return l/length(batch)
end

function train_vanilla2(obs, n_epochs, batch_size, neurons, m, σ2; opt=ADAM(0.001), activation=swish, ϵ=0.05)
    nn = create_nn(neurons, activation=activation)
    losses = []
    μ = obs'
    Dim = size(obs)[1]
    losses = []
    for epoch in ProgressBar(1:n_epochs)
        t, x, z = generate_txz(μ, m, σ2, ϵ=ϵ)
        data = Flux.Float32.(hcat([[x[i,:]..., t[i], z[i,:]...] for i in 1:size(μ)[1]]...))
        data_inout = [(data[1:Dim+1, i], .- data[Dim+2:2*Dim+1, i]) for i in 1:size(μ)[1]]
        data_loader = Flux.DataLoader(data_inout, batchsize=batch_size, shuffle=true)
        loss = 0
        for batch in data_loader
            loss += loss_score(nn, batch)
            gs = Flux.gradient(Flux.params(nn)) do
                loss_score(nn, batch)
            end
            Flux.Optimise.update!(opt, Flux.params(nn), gs)
        end
        push!(losses, loss/length(data_loader))
    end
    return nn, losses
end

function check_loss2(obs, nn, m, σ2; ϵ=0.05)
    Dim = size(obs)[1]
    μ = obs'
    t, x, z = generate_txz(μ, m, σ2; ϵ=ϵ)
    data = Flux.Float32.(hcat([[x[i,:]..., t[i], z[i,:]...] for i in 1:size(μ)[1]]...))
    data_inout = [(data[1:Dim+1, i], .- data[Dim+2:2*Dim+1, i]) for i in 1:size(μ)[1]]
    loss = 0.0
    for (x, y) in data_inout
        loss += Flux.mse(nn(x[1:Dim], x[Dim+1]), y)
    end
    return loss/length(data_inout)
end

nn_vanilla, loss_vanilla = train_vanilla2(obs[:,1:10:end], 1000, 16, [Dim+1, 64, 64, Dim], m, σ2; activation=tanh)
cluster_loss = check_loss2(obs, nnc, m, σ2)
nnv(x, t) = nn_vanilla(Flux.Float32.([x..., t]))[:]
score_vanilla(x, t) = nnv(x, t) ./ σ(t)
plt = Plots.plot(loss_vanilla, label="Vanilla", xlabel="Epochs", ylabel="Loss")
plt = Plots.hline!([cluster_loss], label="Clustered")
play_beep()
savefig(plt, "losses.png")
##
####################### SCORES COMPARISON #######################

index = 
tt = diff_times[index]

xax = [minimum(centers_values[index]):0.01:maximum(centers_values[index])...]

yax =  f_true.(xax, tt)
y_nn1 = [nnc(x, tt)[1] for x in xax]
y_nn2 = [nnv(x, tt)[1] for x in xax]

println("Error clustering: ", mean(abs2,yax .- y_nn1))
println("Error vanilla: ", mean(abs2,yax .- y_nn2))

plt = plot(xax, yax, color=:blue, title="t=$(tt)",label="True")
plt = plot!(xax, y_nn1, color=:red, label="Clustered")
plt = scatter!(centers_values[index], .- averages_values[index], markersize=3, label="", xlabel="x", ylabel="f(x)", markerstrokewidth=0, color=:red)
plt = plot!(xax, y_nn2, color=:green, label="Vanilla")
savefig(plt, "scores_t$(tt).png")
##
######################## SAMPLES GENERATION #########################

function system(x, score, t, f, g)
    return score(x ,t) .* g(t) .^2 .+ f(t) .* (x .- 0.5)    
end

function sample3(Dim, score, n_samples, n_diffs, f, g; system=system)
    dt = 1.0 / n_diffs
    ens = zeros(Dim, n_samples)
    for i in ProgressBar(1:n_samples)
        xOld = STD*randn(Dim) .+ 0.5
        for t in 1:n_diffs
            t_diff = (n_diffs - t + 1) / n_diffs

            k1 = system(xOld, score, t_diff, f, g)
            k2 = system(xOld + k1 * (dt/2), score, t_diff+dt/2, f, g)
            k3 = system(xOld + k2 * (dt/2), score, t_diff+dt/2, f, g)
            k4 = system(xOld + k3 * dt, score, t_diff+dt, f, g)
    
            xNew = xOld .+ (dt/6) * (k1 + 2*k2 + 2*k3 + k4) .+ randn(Dim) .* sqrt(dt) .* g(t_diff)
            xOld = xNew
        end
        ens[:,i] = xOld
    end
    return ens
end

n_diffs_sampling = 100
n_ens = 50000

ens_clustered = sample3(Dim, score_clustered, n_ens, n_diffs_sampling, f, g)
ens_vanilla = sample3(Dim, score_vanilla, n_ens, n_diffs_sampling, f, g)

kde_clustered = kde(ens_clustered[1,:])
kde_vanilla = kde(ens_vanilla[1,:])

##
plt = plot(kde_clustered.x, kde_clustered.density, color=:red, label="Clustered", xlims=(-0.5,1.5), size=(800,600))
plt = plot!(kde_vanilla.x, kde_vanilla.density, color=:green, label="Vanilla", xlims=(-0.5,1.5))
plt = plot!(kde_obs.x, kde_obs.density, color=:blue, label="Observed", xlims=(-0.5,1.5))
savefig(plt, "pdfs.png")
