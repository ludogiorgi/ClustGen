
# Generate data to be clustered
function generate_xz(y, sigma)
    z = randn!(similar(y))
    x = @. y + sigma * z
    return x, z
end

# Clustering of the data and calculation of the cluster centers and value of the score function on the cluster centers 
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

# Generate inputs and targets to train the NN with the clustering loss and reverse sampling method
function generate_inputs_targets(diff_times, averages_values, centers_values, Nc_values; normalization=true)
    inputs = []
    targets = []
    if normalization == true # Normalization of the targets between 0 and 1
        M_averages_values = maximum(hcat(averages_values...))
        m_averages_values = minimum(hcat(averages_values...))
        for (i, t) in enumerate(diff_times)
            averages_values_norm = (averages_values[i] .- m_averages_values_temp) ./ (M_averages_values_temp - m_averages_values_temp)
            inputs_t = hcat([[centers_values[i][:,j]..., t] for j in 1:Nc_values[i]]...)
            targets_t = hcat([[averages_values_norm[:,j]...] for j in 1:Nc_values[i]]...)
            push!(inputs, inputs_t)
            push!(targets, targets_t)
        end
    else
        for (i, t) in enumerate(diff_times)
            inputs_t = hcat([[centers_values[i][:,j]..., t] for j in 1:Nc_values[i]]...)
            targets_t = hcat([[averages_values[i][:,j]...] for j in 1:Nc_values[i]]...)
            push!(inputs, inputs_t)
            push!(targets, targets_t)   
        end
    end
    inputs = vcat(inputs'...)
    targets = vcat(targets'...)
    if normalization == true
        return (Matrix(inputs'), Matrix(targets'), M_averages_values, m_averages_values)
    else
        return (Matrix(inputs'), Matrix(targets'))
    end
end

# Generate inputs and targets to train the NN with the clustering loss and Langevin sampling method
function generate_inputs_targets(averages_values, centers_values, Nc_values; normalization=true)
    if normalization == true
        M_averages_values = maximum(averages_values)
        m_averages_values = minimum(averages_values)
        averages_values_norm = (averages_values .- m_averages_values) ./ (M_averages_values - m_averages_values)
        inputs = hcat([[centers_values[:,j]...] for j in 1:Nc_values]...)
        targets = hcat([[averages_values_norm[:,j]...] for j in 1:Nc_values]...)
    else
        inputs = centers_values
        targets = averages_values
    end
    if normalization == true
        return (inputs, targets, M_averages_values, m_averages_values)
    else
        return (inputs, targets)
    end
end

# Application of the function "calculate_averages" iteratively until convergence for a given value of σ
function f_tilde_σ(σ::Float64, μ; prob = 0.001, do_print=false, conv_param=1e-1, i_max = 150)
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
        if do_print==true
            println("Iteration: $i, Δ: $D_avr_temp")
        end
        averages_old, centers_old = averages_new, centers_new
        i += 1
    end
    return averages, centers, Nc, labels
end

# Application of the function "f_tilde_σ" for a vector of σ values
function f_tilde(σ_values::Vector{Float64}, diff_times::Vector{Float64}, μ; prob = 0.001, do_print=false, conv_param=1e-1, i_max = 150, normalization=true)
    averages_values = []
    centers_values = []
    Nc_values = []
    for i in eachindex(σ_values)
        averages, centers, Nc, labels = f_tilde_σ(σ_values[i], μ; prob=prob, do_print=do_print, conv_param=conv_param, i_max=i_max)
        push!(averages_values, averages)
        push!(centers_values, centers)
        push!(Nc_values, Nc)
    end
    return generate_inputs_targets(diff_times, averages_values, centers_values, Nc_values; normalization=normalization)
end

# Application of the function "f_tilde_σ" for a single value of σ
function f_tilde(σ_value::Float64, μ; prob = 0.001, do_print=false, conv_param=1e-1, i_max = 150, normalization=true)
    averages, centers, Nc, labels = f_tilde_σ(σ_value, μ; prob=prob, do_print=do_print, conv_param=conv_param, i_max=i_max)
    return generate_inputs_targets(averages, centers, Nc; normalization=normalization)
end

# # Application of the function "f_tilde_σ" for a vector of σ values
# function f_tilde_labels(σ_values::Vector{Float64}, diff_times::Vector{Float64}, μ; prob = 0.001, do_print=false, conv_param=1e-1, i_max = 150, normalization=true)
#     averages_values = []
#     centers_values = []
#     Nc_values = []
#     labels_values = []
#     for i in eachindex(σ_values)
#         averages, centers, Nc, labels = f_tilde_σ(σ_values[i], μ; prob=prob, do_print=do_print, conv_param=conv_param, i_max=i_max)
#         push!(averages_values, averages)
#         push!(centers_values, centers)
#         push!(Nc_values, Nc)
#         push!(labels_values, labels)
#     end
#     return generate_inputs_targets(diff_times, averages_values, centers_values, Nc_values; normalization=normalization), labels_values
# end

# Application of the function "f_tilde_σ" for a single value of σ
function f_tilde_labels(σ_value::Float64, μ; prob = 0.001, do_print=false, conv_param=1e-1, i_max = 150, normalization=true)
    averages, centers, Nc, labels = f_tilde_σ(σ_value, μ; prob=prob, do_print=do_print, conv_param=conv_param, i_max=i_max)
    return averages, centers, Nc, labels
end

