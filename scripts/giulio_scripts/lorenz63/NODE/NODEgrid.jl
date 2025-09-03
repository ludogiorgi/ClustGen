using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Distributed
addprocs(Sys.CPU_THREADS - 1)
@everywhere using Revise, MarkovChainHammer, ClustGen, KernelDensity, HDF5, BSON, Plots, LinearAlgebra, Distributions
@everywhere using DifferentialEquations, Zygote, CUDA, SciMLSensitivity, Flux, OrdinaryDiffEq, Random
@everywhere using Statistics, StatsBase, ProgressBars, Optimisers, ComponentArrays, Dates
@everywhere using BSON: @save, @load


# Define all functions and constants inside @everywhere to share with all workers
@everywhere begin
    function F(x, t, ε ; µ=10.0, ρ=28.0, β=8/3)
        dy1 = µ/ε^2 * (x[2] - x[1])
        dy2 = 1/ε^2 * (x[1] * (ρ - x[3]) - x[2])
        dy3 = 1/ε^2 * (x[1] * x[2] - β * x[3])
        return [dy1, dy2, dy3]
    end

    function sigma(x, t; noise = 0.0)
        return [noise, noise, noise]
    end

    function delay_embedding(x; τ, m)
        q = round(Int, τ / dt)
        start_idx = 1 + (m - 1) * q
        Z = [ [x[i - j*q] for j in 0:m-1] for i in start_idx:length(x) ]
        return hcat(Z...)
    end

    function gen_batches(x::Matrix, batch_len::Int, n_batch::Int)
        datasize = size(x, 2)
        r = rand(1:datasize - batch_len, n_batch)
        return [x[:, i+1:i+batch_len] for i in r]
    end

    function create_nn(layers::Vector{Int}; activation_hidden=swish, activation_output=identity)
        layer_list = []
        for i in 1:(length(layers) - 2)
            push!(layer_list, Dense(layers[i], layers[i+1], activation_hidden))
        end
        push!(layer_list, Dense(layers[end-1], layers[end], activation_output))
        return Chain(layer_list...)
    end

    function dudt!(du, u, p, t)
        du .= re(p)(u)
    end

    function predict_neuralode(u0, p, tspan, t)
        prob = ODEProblem(dudt!, u0, tspan, p)
        sol = solve(prob, Tsit5(), saveat=t, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
        return hcat(sol.u...)
    end

    function loss_neuralode(p, data_sample, tspan, t)
        loss = 0.0f0
        for i in 1:100
            u = data_sample[rand(1:length(data_sample))]
            pred = predict_neuralode(u[:, 1], p, tspan, t)
            loss += sum(abs2, u[:, 2:end] .- pred[:, 1:end])
        end
        return loss / 100
    end

    function estimate_tau(y, dt; threshold=0.2)
        y_centered = y .- mean(y)
        acf = autocor(y_centered)
        for i in 2:length(acf)
            if abs(acf[i]) < threshold
                return i * dt, acf
            end
        end
        return dt * length(acf), acf
    end
end

# Shared data
@everywhere dt = 0.001f0
@everywhere Nsteps = 1000000
@everywhere ε = 0.5
@everywhere f = (x, t) -> F(x, t, ε)
@everywhere obs_nn = evolve(randn(3), dt, Nsteps, f, sigma; timestepper=:rk4, resolution=1)
@everywhere y = Float32.(obs_nn[2, :])
@everywhere μ, σ = mean(y), std(y)
@everywhere y_norm = (y .- μ) ./ σ
@everywhere base_path = "/Users/giuliodelfelice/Desktop/MIT"


# Parameter grid
@everywhere begin
    m_values = [10, 12, 14, 16]
    n_steps_values = [40, 45, 50, 55]
    grid_params = [(m, n) for m in m_values for n in n_steps_values]
end
@sync @distributed for i in 1:length(params)
    m, n_steps = grid_params[i]
    start_time = now()
    println("[$(myid())] Starting model m=$m, n_steps=$n_steps")

    # Delay embedding
    τ_opt, _ = estimate_tau(obs_nn[2, :], dt)
    τ = 0.25 * τ_opt
    Z = Float32.(delay_embedding(y_norm; τ=τ, m=m))

    # Batching
    batch_size = n_steps + 1
    local_data_sample = gen_batches(Z, batch_size, 2000)

    # Time setup
    t = collect(0.0f0:dt:dt*(n_steps - 1))
    tspan = (t[1], t[end])

    # Model + training
    model = create_nn([m, 256, 256, m], activation_hidden=tanh, activation_output=identity)
    flat_p0, re = Flux.destructure(model)
    p = flat_p0
    opt = Optimisers.Adam(0.01)
    state = Optimisers.setup(opt, p)

    for epoch in 1:500
        loss_val, back = Flux.withgradient(p) do p
            loss = 0.0f0
            for i in 1:100
                u = local_data_sample[rand(1:end)]
                pred = predict_neuralode(u[:, 1], p, tspan, t)
                loss += sum(abs2, u[:, 2:end] .- pred[:, 1:end])
            end
            loss / 100
        end
        state, p = Optimisers.update(state, p, back[1])

        if epoch % 100 == 0
            println("[$(myid())][m=$m n=$n_steps] Epoch $epoch/500 — Loss: $(round(loss_val, digits=4))")
        end
    end

    elapsed = round(Dates.now() - start_time, RoundNearest, Millisecond)
    println("[$(myid())] Finished model m=$m, n_steps=$n_steps in $elapsed")

    # Prediction
    model_trained = re(p)
    u0 = Z[:, 1]
    n_long = 200
    t_long = collect(0.0f0:dt:dt*(n_long - 1))
    tspan_long = (t_long[1], t_long[end])
    pred_long = predict_neuralode(u0, flat_p0, tspan_long, t_long)

    max_steps = min(size(pred_long, 2), size(Z, 2))
    y_pred_long = pred_long[1, 1:max_steps]
    y_true_long = Z[1, 1:max_steps]
    t_plot = t_long[1:max_steps]

    # Plot time series
    plotlyjs()
    p_time = plot(t_plot, y_true_long, label="True y2(t)", lw=2)
    plot!(p_time, t_plot, y_pred_long, label="Predicted y(t)", lw=2, ls=:dash)

    # Plot PDFs
    kde_pred = kde(y_pred_long)
    kde_obs = kde(y_true_long)
    p_pdf = plot(kde_pred.x, kde_pred.density, label="predicted", color=:red)
    plot!(p_pdf, kde_obs.x, kde_obs.density, label="observed", color=:blue)

    # Save all
    test_folder = joinpath(base_path, "NODE_tests_m$(m)_nsteps$(n_steps)")
    mkpath(test_folder)

    savefig(p_time, joinpath(test_folder, "time_series.svg"))
    savefig(p_pdf, joinpath(test_folder, "PDFs.svg"))

    @save joinpath(test_folder, "y_true_long.bson") data=y_true_long
    @save joinpath(test_folder, "y_pred_long.bson") data=y_pred_long
end
# ========= Summary Plots ========= #
results = []
for m in m_values
    for n_steps in n_steps_values
        test_folder = joinpath(base_path, "NODE_tests_m$(m)_nsteps$(n_steps)")
        y_true_path = joinpath(test_folder, "y_true_long.bson")
        y_pred_path = joinpath(test_folder, "y_pred_long.bson")
        if isfile(y_true_path) && isfile(y_pred_path)
            y_true_long = BSON.load(y_true_path)["data"]
            y_pred_long = BSON.load(y_pred_path)["data"]
            corr = cor(y_pred_long, y_true_long)
            rmse = sqrt(mean((y_pred_long .- y_true_long).^2))
            nrmse = 1 - rmse / std(y_true_long)
            push!(results, (m=m, n=n_steps, corr=corr, nrmse=nrmse))
        end
    end
end


# Save results
for m in m_values
    subset = filter(r -> r.m == m, results)
    ns = [r.n for r in subset]
    rmses = [r.nrmse for r in subset]
    corrs = [r.corr for r in subset]

    plotlyjs()
    plt_perf = plot(ns, rmses, label="NRMSE", lw=2, marker=:circle)
    plot!(plt_perf, ns, corrs, label="Correlation", lw=2, ls=:dash, marker=:diamond)
    xlabel!("n_steps")
    ylabel!("Score")
    title!("Performance vs n_steps for m = $m")

    m_folder = joinpath(base_path, "NODE_tests_m$(m)_summary")
    mkpath(m_folder)
    savefig(plt_perf, joinpath(m_folder, "performance_vs_nsteps.svg"))
end
