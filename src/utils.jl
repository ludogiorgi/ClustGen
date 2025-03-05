"""
    covariance(x1, x2; timesteps=length(x), progress=false)

Compute the empirical time-lagged covariance between two time series.

# Arguments
- `x1`: First time series
- `x2`: Second time series
- `timesteps`: Maximum number of time lags to compute (default: length of series)
- `progress`: Whether to show progress bar (default: false)

# Returns
- Vector of covariance values at each lag
"""
function covariance(x1, x2; timesteps=length(x1), progress=false)
    # Calculate means for centering
    μ1 = mean(x1)
    μ2 = mean(x2)
    
    # Initialize covariance array
    autocor = zeros(timesteps)
    
    # Set up iterator with optional progress bar
    progress ? iter = ProgressBar(1:timesteps) : iter = 1:timesteps
    
    # Compute time-lagged covariance for each lag
    for i in iter
        autocor[i] = mean(x1[i:end] .* x2[1:end-i+1]) - μ1 * μ2
    end
    
    return autocor
end

"""
    covariance(g⃗1, g⃗2, Q::Eigen, timelist; progress=false)

Compute covariance using eigenvector decomposition of generator matrix.

# Arguments
- `g⃗1`: First observable vector
- `g⃗2`: Second observable vector
- `Q`: Eigensystem of generator matrix (eigenvalues and eigenvectors)
- `timelist`: List of time points at which to evaluate covariance
- `progress`: Whether to show progress bar (default: false)

# Returns
- Covariance evaluated at each time point
"""
function covariance(g⃗1, g⃗2, Q::Eigen, timelist; progress=false)
    # Uncomment to check for ergodicity
    # @assert all(real.(Q.values[1:end-1]) .< 0) "Did not pass an ergodic generator matrix"
    
    # Initialize covariance array
    autocov = zeros(length(timelist))
    
    # Unpack eigenvalues and eigenvectors
    Λ, V = Q
    
    # Compute stationary distribution from eigenvector of zero eigenvalue
    p = real.(V[:, end] ./ sum(V[:, end]))
    
    # Project observables onto eigenbasis
    v1 = V \ (p .* g⃗1)
    w2 = g⃗2' * V
    
    # Calculate means for centering
    μ1 = sum(p .* g⃗1)
    μ2 = sum(p .* g⃗2)
    
    # Set up iterator with optional progress bar
    progress ? iter = ProgressBar(eachindex(timelist)) : iter = eachindex(timelist)
    
    # Compute covariance at each time point
    for i in iter
        autocov[i] = real(w2 * (exp.(Λ .* timelist[i]) .* v1) - μ1 * μ2)
    end
    
    return autocov
end

"""
    covariance(g⃗1, g⃗2, Q, timelist; progress=false)

Compute covariance using generator matrix directly (computes eigendecomposition).

# Arguments
- `g⃗1`: First observable vector
- `g⃗2`: Second observable vector
- `Q`: Generator matrix
- `timelist`: List of time points at which to evaluate covariance
- `progress`: Whether to show progress bar (default: false)

# Returns
- Covariance evaluated at each time point
"""
covariance(g⃗1, g⃗2, Q, timelist; progress=false) = covariance(g⃗1, g⃗2, eigen(Q), timelist; progress=progress)

"""
    cleaning(averages, centers, labels)

Clean up cluster assignments by renumbering to ensure contiguous indices.

# Arguments
- `averages`: Matrix of average values for each cluster
- `centers`: Matrix of cluster centers
- `labels`: Vector of cluster assignments

# Returns
- Tuple containing:
  - `averages_new`: Filtered average values
  - `centers_new`: Filtered cluster centers
  - Number of unique clusters
  - `labels_new`: Remapped cluster assignments
"""
function cleaning(averages, centers, labels)
    # Find unique cluster indices sorted in ascending order
    unique_clusters = sort(unique(labels))
    
    # Create mapping from old cluster indices to new contiguous indices
    mapping = Dict(old_cluster => new_cluster for (new_cluster, old_cluster) in enumerate(unique_clusters))
    
    # Apply mapping to labels
    labels_new = [mapping[cluster] for cluster in labels]
    
    # Select columns corresponding to valid clusters
    averages_new = averages[:, unique_clusters]
    centers_new = centers[:, unique_clusters]
    
    return averages_new, centers_new, length(unique_clusters), labels_new
end

"""
    meshgrid(x, y)

Create meshgrid arrays for 2D plotting and evaluation.

# Arguments
- `x`: Vector of x-coordinates
- `y`: Vector of y-coordinates

# Returns
- Tuple containing meshgrid matrices (X, Y)
"""
function meshgrid(x, y)
    X = [i for i in x, j in y]
    Y = [j for i in x, j in y]
    return X, Y
end

"""
    vectorfield2d(score_model, x_range, y_range)

Compute a 2D vector field from a score model.

# Arguments
- `score_model`: Function that computes the score (gradient of log density)
- `x_range`: Range of x-coordinates
- `y_range`: Range of y-coordinates

# Returns
- Tuple containing vector components (u, v) on the meshgrid
"""
function vectorfield2d(score_model, x_range, y_range)
    # Create meshgrid
    X, Y = meshgrid(x_range, y_range)
    
    # Initialize vector field components
    u = zeros(size(X))
    v = zeros(size(Y))
    
    # Evaluate score function at each grid point
    for i in 1:size(X, 1)
        for j in 1:size(X, 2)
            score = score_model([X[i,j], Y[i,j]])
            u[i,j] = score[1]
            v[i,j] = score[2]
        end
    end
    
    return u, v
end

"""
    save_model(nn, filename)

Save a Flux model and its parameters to disk.

# Arguments
- `nn`: Neural network model
- `filename`: Output filename (will append .bson if not present)
"""
function save_model(nn, filename)
    # Ensure filename has .bson extension
    if !endswith(filename, ".bson")
        filename = filename * ".bson"
    end
    
    # Extract model parameters
    model_params = Flux.params(nn)
    
    # Save model architecture and parameters
    BSON.@save filename model=nn params=model_params
    
    println("Model saved to $filename")
end

"""
    load_model(filename)

Load a Flux model from disk.

# Arguments
- `filename`: Input filename

# Returns
- Loaded neural network model
"""
function load_model(filename)
    # Ensure filename has .bson extension
    if !endswith(filename, ".bson")
        filename = filename * ".bson"
    end
    
    # Load model data
    model_data = BSON.load(filename)
    
    # Extract model and parameters
    model = model_data[:model]
    params = model_data[:params]
    
    # Restore parameters
    Flux.loadparams!(model, params)
    
    println("Model loaded from $filename")
    return model
end

"""
    model_to_score(nn, dim_in, normalization_params=nothing)

Convert a trained neural network to a score function.

# Arguments
- `nn`: Neural network model
- `dim_in`: Input dimension
- `normalization_params`: Optional tuple (max, min) for denormalizing outputs

# Returns
- Score function that takes a state vector and returns gradient of log density
"""
function model_to_score(nn, dim_in, normalization_params=nothing)
    # Create score function based on whether normalization was used
    if normalization_params === nothing
        # No normalization case
        function score(x)
            return nn([x...])
        end
    else
        # Denormalize outputs using provided parameters
        M, m = normalization_params
        
        function score(x)
            # Normalize neural network output to recover original scale
            return nn([x...]) .* (M - m) .+ m
        end
    end
    
    return score
end