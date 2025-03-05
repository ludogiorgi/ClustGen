"""
    generate_xz(y, sigma)

Generate perturbed data and noise vectors for clustering.

# Arguments
- `y`: Original data points
- `sigma`: Noise standard deviation

# Returns
- Tuple of (perturbed data, noise vectors)
"""
function generate_xz(y, sigma)
    z = randn!(similar(y))  # Generate random noise with same shape as y
    x = @. y + sigma * z    # Add scaled noise to original data
    return x, z
end

"""
    calculate_averages(X, z, x, y)

Calculate cluster centers and average values for each cluster.

# Arguments
- `X`: Cluster labels/indices for each point
- `z`: Noise vectors
- `x`: Perturbed data points
- `y`: Original data points

# Returns
- Tuple of (average noise, average residuals, cluster centers)
"""
function calculate_averages(X, z, x, y)
    # Get dimensions
    Ndim, Nz = size(z)
    Nc = maximum(X)
    
    # Initialize arrays to store results
    averages = zeros(Ndim, Nc)
    averages_residual = zeros(Ndim, Nc)
    centers = zeros(Ndim, Nc)
    
    # Initialize accumulators for sums
    z_sum = zeros(Ndim, Nc)
    x_sum = zeros(Ndim, Nc)
    y_sum = zeros(Ndim, Nc)
    count = zeros(Ndim, Nc)
    
    # Accumulate sums for each cluster
    for i in 1:Nz
        segment_index = X[i]
        for dim in 1:Ndim
            z_sum[dim, segment_index] += z[dim, i]
            x_sum[dim, segment_index] += x[dim, i]
            y_sum[dim, segment_index] += y[dim, i]
            count[dim, segment_index] += 1
        end
    end
    
    # Calculate averages per cluster
    for dim in 1:Ndim
        for i in 1:Nc
            if count[dim, i] != 0
                averages[dim, i] = z_sum[dim, i] / count[dim, i]
                averages_residual[dim, i] = y_sum[dim, i] / count[dim, i]
                centers[dim, i] = x_sum[dim, i] / count[dim, i]
            end
        end
    end
    
    return averages, averages_residual, centers
end

"""
    generate_inputs_targets(diff_times, averages_values, centers_values, Nc_values; normalization=true)

Generate inputs and targets for neural network training with reverse sampling method.

# Arguments
- `diff_times`: Vector of diffusion times
- `averages_values`: List of average values for each diffusion time
- `centers_values`: List of cluster centers for each diffusion time
- `Nc_values`: List of number of clusters for each diffusion time
- `normalization`: Whether to normalize the targets between 0 and 1

# Returns
- Tuple containing (inputs, targets) matrices, and optional normalization parameters
"""
function generate_inputs_targets(diff_times, averages_values, centers_values, Nc_values; normalization=true)
    inputs = []
    targets = []
    
    if normalization == true  # Normalization of the targets between 0 and 1
        # Calculate global min and max across all diffusion times
        M_averages_values = maximum(hcat(averages_values...))
        m_averages_values = minimum(hcat(averages_values...))
        
        for (i, t) in enumerate(diff_times)
            # Normalize current time's averages
            averages_values_norm = (averages_values[i] .- m_averages_values) ./ (M_averages_values - m_averages_values)
            
            # Format inputs with time parameter appended
            inputs_t = hcat([[centers_values[i][:,j]..., t] for j in 1:Nc_values[i]]...)
            
            # Format normalized targets
            targets_t = hcat([[averages_values_norm[:,j]...] for j in 1:Nc_values[i]]...)
            
            push!(inputs, inputs_t)
            push!(targets, targets_t)
        end
    else
        for (i, t) in enumerate(diff_times)
            # Format inputs with time parameter appended
            inputs_t = hcat([[centers_values[i][:,j]..., t] for j in 1:Nc_values[i]]...)
            
            # Format original targets without normalization
            targets_t = hcat([[averages_values[i][:,j]...] for j in 1:Nc_values[i]]...)
            
            push!(inputs, inputs_t)
            push!(targets, targets_t)   
        end
    end
    
    # Vertically concatenate all time steps
    inputs = vcat(inputs'...)
    targets = vcat(targets'...)
    
    if normalization == true
        return (Matrix(inputs'), Matrix(targets'), M_averages_values, m_averages_values)
    else
        return (Matrix(inputs'), Matrix(targets'))
    end
end

"""
    generate_inputs_targets(averages_values, centers_values, Nc_values; normalization=true)

Generate inputs and targets for neural network training with Langevin sampling method.

# Arguments
- `averages_values`: Average values for score estimation
- `centers_values`: Cluster centers
- `Nc_values`: Number of clusters
- `normalization`: Whether to normalize the targets between 0 and 1

# Returns
- Tuple containing (inputs, targets), and optional normalization parameters
"""
function generate_inputs_targets(averages_values, centers_values, Nc_values; normalization=true)
    if normalization == true
        # Calculate global min and max
        M_averages_values = maximum(averages_values)
        m_averages_values = minimum(averages_values)
        
        # Normalize targets
        averages_values_norm = (averages_values .- m_averages_values) ./ (M_averages_values - m_averages_values)
        
        inputs = reshape(centers_values, :, size(centers_values, 2))
        targets = reshape(averages_values_norm, :, size(averages_values_norm, 2))
    else
        inputs = centers_values
        targets = averages_values
    end
    
    if normalization == true
        return (inputs, targets), M_averages_values, m_averages_values
    else
        return (inputs, targets)
    end
end

"""
    f_tilde_σ(σ::Float64, μ; prob = 0.001, do_print=false, conv_param=1e-1, i_max = 150)

Iteratively estimate score function at σ until convergence.

# Arguments
- `σ`: Noise standard deviation
- `μ`: Original data points
- `prob`: Probability threshold for state space partitioning
- `do_print`: Whether to print progress
- `conv_param`: Convergence threshold
- `i_max`: Maximum number of iterations

# Returns
- Tuple containing (average noise, average residuals, centers, number of clusters, partition)
"""
function f_tilde_σ(σ::Float64, μ; prob = 0.001, do_print=false, conv_param=1e-1, i_max = 150)
    # Initialize state space partitioning
    method = Tree(false, prob)
    
    # Generate initial perturbed data and noise
    x, z = generate_xz(μ, σ)
    
    # Create partition of state space
    state_space_partitions = StateSpacePartition(x; method = method)
    Nc = maximum(state_space_partitions.partitions)
    
    # Get cluster labels for each point
    labels = [state_space_partitions.embedding(x[:,i]) for i in 1:size(x)[2]]
    
    # Calculate initial averages
    averages, averages_residual, centers = calculate_averages(labels, z, x, μ)
    averages_old, averages_residual_old, centers_old = averages, averages_residual, centers
    
    # Iterative refinement
    D_avr_temp = 1
    i = 1
    while D_avr_temp > conv_param && i < i_max
        # Generate new perturbed data
        x, z = generate_xz(μ, σ)
        
        # Apply partition
        labels = [state_space_partitions.embedding(x[:,i]) for i in 1:size(x)[2]]
        
        # Calculate new averages
        averages, averages_residual, centers = calculate_averages(labels, z, x, μ)
        
        # Update running averages
        averages_new = (averages .+ i .* averages_old) ./ (i+1)
        averages_residual_new = (averages_residual .+ i .* averages_residual_old) ./ (i+1)
        centers_new = (centers .+ i .* centers_old) ./ (i+1)
        
        # Check convergence
        D_avr_temp = mean(abs2, averages_new .- averages_old) / mean(abs2, averages_new)
        
        if do_print==true
            println("Iteration: $i, Δ: $D_avr_temp")
        end
        
        # Update old values for next iteration
        averages_old, averages_residual_old, centers_old = averages_new, averages_residual_new, centers_new
        i += 1
    end
    
    return averages_old, averages_residual_old, centers_old, Nc, state_space_partitions
end

"""
    f_tilde(σ_values::Vector{Float64}, diff_times::Vector{Float64}, μ; kwargs...)

Apply f_tilde_σ for multiple noise levels with diffusion times.

# Arguments
- `σ_values`: Vector of noise standard deviations
- `diff_times`: Vector of diffusion times (must match length of σ_values)
- `μ`: Original data points
- `kwargs`: Additional parameters passed to f_tilde_σ and generate_inputs_targets

# Returns
- Neural network inputs and targets for training
"""
function f_tilde(σ_values::Vector{Float64}, diff_times::Vector{Float64}, μ; 
                prob = 0.001, do_print=false, conv_param=1e-1, i_max = 150, normalization=true, residual=false)
    # Initialize storage
    averages_values = []
    averages_residual_values = []
    centers_values = []
    Nc_values = []
    
    # Process each noise level
    for i in eachindex(σ_values)
        averages, averages_residual, centers, Nc, _ = f_tilde_σ(σ_values[i], μ; 
                                                           prob=prob, 
                                                           do_print=do_print, 
                                                           conv_param=conv_param, 
                                                           i_max=i_max)
        push!(averages_values, averages)
        push!(averages_residual_values, averages_residual)
        push!(centers_values, centers)
        push!(Nc_values, Nc)
    end
    
    # Generate inputs and targets based on residual flag
    if residual == true
        return generate_inputs_targets(diff_times, averages_residual_values, centers_values, Nc_values; normalization=normalization)
    else
        return generate_inputs_targets(diff_times, averages_values, centers_values, Nc_values; normalization=normalization)
    end
end

"""
    f_tilde(σ_value::Float64, μ; kwargs...)

Apply f_tilde_σ for a single noise level.

# Arguments
- `σ_value`: Noise standard deviation
- `μ`: Original data points
- `kwargs`: Additional parameters passed to f_tilde_σ and generate_inputs_targets

# Returns
- Neural network inputs and targets for training
"""
function f_tilde(σ_value::Float64, μ; 
                prob = 0.001, do_print=false, conv_param=1e-1, i_max = 150, normalization=true, residual=false)
    # Process single noise level
    averages, averages_residual, centers, Nc, _ = f_tilde_σ(σ_value, μ; 
                                                       prob=prob, 
                                                       do_print=do_print, 
                                                       conv_param=conv_param, 
                                                       i_max=i_max)
    
    # Generate inputs and targets based on residual flag
    if residual == true
        return generate_inputs_targets(averages_residual, centers, Nc; normalization=normalization)
    else
        return generate_inputs_targets(averages, centers, Nc; normalization=normalization)
    end
end

"""
    f_tilde_ssp(σ_value::Float64, μ; kwargs...)

Apply f_tilde_σ for a single noise level and return the state space partition.

# Arguments
- `σ_value`: Noise standard deviation
- `μ`: Original data points
- `kwargs`: Additional parameters passed to f_tilde_σ

# Returns
- Tuple containing (averages, averages_residual, centers, Nc, state_space_partition)
"""
function f_tilde_ssp(σ_value::Float64, μ; 
                    prob = 0.001, do_print=false, conv_param=1e-1, i_max = 150, normalization=true)
    # Process single noise level and return full results including state space partition
    averages, averages_residual, centers, Nc, ssp = f_tilde_σ(σ_value, μ; 
                                                         prob=prob, 
                                                         do_print=do_print, 
                                                         conv_param=conv_param, 
                                                         i_max=i_max)
    return averages, averages_residual, centers, Nc, ssp
end

"""
    f_tilde_labels(σ_value::Float64, μ; kwargs...)

Apply f_tilde_σ and return cluster labels along with averages and centers.

# Arguments
- `σ_value`: Noise standard deviation
- `μ`: Original data points
- `kwargs`: Additional parameters passed to f_tilde_σ

# Returns
- Tuple containing (averages, centers, Nc, labels)
"""
function f_tilde_labels(σ_value::Float64, μ; 
                       prob = 0.001, do_print=false, conv_param=1e-1, i_max = 150, normalization=true)
    # Process single noise level
    averages, averages_residual, centers, Nc, ssp = f_tilde_σ(σ_value, μ; 
                                                         prob=prob, 
                                                         do_print=do_print, 
                                                         conv_param=conv_param, 
                                                         i_max=i_max)
    
    # Generate perturbed data for labeling
    x, _ = generate_xz(μ, σ_value)
    
    # Apply partition to get labels
    labels = [ssp.embedding(x[:,i]) for i in 1:size(x)[2]]
    
    return averages, centers, Nc, labels
end