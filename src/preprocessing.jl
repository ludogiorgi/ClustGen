function generate_xz(y, sigma)
    z = randn!(similar(y))
    x = @. y + sigma * z
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

function f_tilde_σ(σ, μ; prob = 0.001, do_plot=false, conv_param=1e-1, i_max = 150)
    if do_plot==true
        plt = plot()
        D_avr = []
    end
    method = Tree(false, prob)
    x, z = generate_xz(μ, σ)
    state_space_partitions = StateSpacePartition(x; method = method)
    Nc = maximum(state_space_partitions.partitions)
    labels = [state_space_partitions.embedding(x[:,i]) for i in 1:size(x)[2]]
    averages, centers = calculate_averages(labels, z, x)
    averages_old, centers_old = averages, centers
    D_avr_temp = 1
    i = 1
    while D_avr_temp > conv_param && i < i_max
        x, z = generate_xz(μ, σ)
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

function f_tilde(σ_values, μ; prob = 0.001, do_plot=false, conv_param=1e-1, i_max = 150)
    averages_values = []
    centers_values = []
    Nc_values = []
    for i in eachindex(σ_values)
        σ = σ_values[i]
        averages, centers, Nc = f_tilde_σ(σ, μ; prob=prob, do_plot=do_plot, conv_param=conv_param, i_max=i_max)
        push!(averages_values, averages)
        push!(centers_values, centers)
        push!(Nc_values, Nc)
    end
    return averages_values, centers_values, Nc_values
end
