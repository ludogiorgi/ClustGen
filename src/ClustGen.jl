"""
ClustGen

A comprehensive toolkit for clustering-based generative modeling of dynamical systems.
This package provides tools for analyzing, clustering, and generating trajectories 
from dynamical systems using various machine learning and statistical techniques.
"""
module ClustGen

# Core dependencies
using StateSpacePartitions
using LinearAlgebra
using Random
using Distributed
using SharedArrays
using StaticArrays

# Statistics and data processing
using Statistics
using StatsBase
using KernelDensity
using Optim

# I/O and visualization
using HDF5
using BSON
using Plots
using ProgressBars
using ProgressMeter
using GLMakie

# Numerical utilities
using QuadGK
using SpecialFunctions

# Deep learning
using Flux
using CUDA

# ===== Component modules =====
include("autoencoder.jl")       # Neural network autoencoders
include("preprocessing.jl")     # Data preprocessing utilities
include("training.jl")          # Model training functions
include("sampling.jl")          # Sampling methods
include("noising_schedules.jl") # Noise schedules for diffusion models
include("generate.jl")          # Data generation utilities
include("utils.jl")             # General utility functions
include("io.jl")                # I/O functions for saving/loading models and data
include("diffusion_matrix.jl")  # Diffusion matrix estimation
include("responses.jl")         # Response function generators
include("KSE_integrate.jl")     # KSE integration functions
include("control.jl")          # Control and optimal forcing

# ===== Exported functions =====
# Autoencoder functionality
export apply_autoencoder, read_autoencoder

# Clustering and preprocessing
export f_tilde, f_tilde_ssp, f_tilde_labels, generate_inputs_targets

# Model training
export train, check_loss

# Sampling methods
export sample_reverse, sample_langevin, sample_langevin_Σ

# Noise schedules
export σ_variance_exploding, g_variance_exploding

# Visualization
export vectorfield2d, meshgrid

# Data generation
export evolve, evolve_ens

# Statistical utilities
export covariance, cleaning
export computeSigma
export compute_density_from_score, maxent_distribution


# Response functions
export generate_numerical_response, generate_score_response, generate_numerical_response_f,
       generate_numerical_response_HO, generate_numerical_response_f_HO

# KSE integration
export KSE_integrate, dealias, domain, field2vector, vector2field, create_ks_animation, 
       reduce_fourier_energy, reconstruct_physical_from_reduced

# Control and optimal forcing
export create_linear_interpolator, f_star, compute_M_matrix, compute_C_matrices, find_optimal_u

# ===== I/O functions =====
export save_variables_to_hdf5, read_variables_from_hdf5, save_current_workspace, 
       load_workspace_from_file, save_model, load_model, save_model_safe, load_model_safe

# ===== Main score estimation functions =====
export calculate_score_kgmm, calculate_score_vanilla


"""
    calculate_score_kgmm(raw_timeseries; kwargs...)

Estimate the score function using K-Gaussian Mixture Model (K-GMM) clustering approach.

This function performs score estimation by first clustering the data using state space partitioning,
then training a neural network to approximate the score function based on cluster statistics.

# Arguments
- `raw_timeseries::AbstractMatrix`: Input time series data with shape (dimensions, time_points)

## Clustering & Score Parameters
- `σ_value::Float64 = 0.05`: Noise standard deviation for score estimation
- `clustering_prob::Float64 = 0.0005`: Probability threshold for state space partitioning
- `clustering_conv_param::Float64 = 0.002`: Convergence parameter for clustering iterations
- `clustering_max_iter::Int = 150`: Maximum iterations for clustering convergence
- `use_normalization_for_clustering::Bool = true`: Whether to normalize data before clustering

## Neural Network Training Parameters
- `epochs::Int = 2000`: Number of training epochs
- `batch_size::Int = 32`: Mini-batch size for training
- `hidden_layers::Vector{Int} = [100, 50]`: Hidden layer sizes (input/output dims determined from clustering data)
- `activation = swish`: Activation function for hidden layers
- `last_activation = identity`: Activation function for output layer
- `optimizer = Adam(0.001)`: Optimizer for training
- `use_gpu::Bool = false`: Whether to use GPU acceleration

## Control Parameters
- `verbose::Bool = false`: Whether to show progress information and training details

# Returns
- `NamedTuple` containing:
  - `score_function`: The estimated score function
  - `loss_history`: Training loss history
  - `averages`: Cluster average values
  - `centers`: Cluster center locations
  - `Nc`: Number of clusters found
  - `ssp`: State space partition object
  - `normalization_params`: Tuple of (mean, std) used for normalization

# Example
```julia
# Basic usage
result = calculate_score_kgmm(data; verbose=true)
score_fn = result.score_function

# Advanced usage with custom parameters
result = calculate_score_kgmm(data; 
    σ_value=0.1, 
    epochs=5000, 
    hidden_layers=[200, 100],
    use_gpu=true,
    verbose=true
)
```
"""
function calculate_score_kgmm(
    raw_timeseries::AbstractMatrix;
    # Clustering & Score parameters
    σ_value::Float64 = 0.05,
    clustering_prob::Float64 = 0.0005,
    clustering_conv_param::Float64 = 0.002,
    clustering_max_iter::Int = 150,
    use_normalization_for_clustering::Bool = true,
    # NN Training parameters
    epochs::Int = 2000,
    batch_size::Int = 32,
    hidden_layers::Vector{Int} = [100, 50],
    activation = swish,
    last_activation = identity,
    optimizer = Flux.Adam(0.001),
    use_gpu::Bool = false,
    # Control parameters
    verbose::Bool = false
    )

    if verbose
        println("🎯 Starting K-GMM Score Estimation")
        println("=" ^ 50)
    end

    # === 1. Data Validation and Normalization ===
    if verbose
        println("📊 [Step 1/3] Normalizing input time series...")
        println("   • Input shape: $(size(raw_timeseries))")
    end
    
    # Validate input dimensions
    if size(raw_timeseries, 1) < 1 || size(raw_timeseries, 2) < 2
        throw(ArgumentError("Input data must have at least 1 dimension and 2 time points"))
    end
    
    # For K-GMM clustering, we need to determine the actual input/output dimensions
    # from the training data that will be generated later, but we can estimate it now
    data_dim = size(raw_timeseries, 1)
    
    if verbose
        println("   • Data dimension: $data_dim")
    end
    
    # Compute normalization parameters
    M = mean(raw_timeseries, dims=2)
    S = std(raw_timeseries, dims=2)
    
    # Check for zero standard deviation
    if any(S .≈ 0)
        @warn "Some dimensions have zero standard deviation. Adding small regularization."
        S = S .+ 1e-8
    end
    
    obs = (raw_timeseries .- M) ./ S
    
    if verbose
        println("   • Normalization parameters computed")
        println("✅ Normalization complete.")
    end

    # === 2. Clustering ===
    if verbose
        println("🔗 [Step 2/3] Performing clustering...")
        println("   • σ_value: $σ_value")
        println("   • Clustering probability threshold: $clustering_prob")
        println("   • Convergence parameter: $clustering_conv_param")
    end
    
    # Determine which data to use for clustering
    data_for_clustering = use_normalization_for_clustering ? obs : raw_timeseries
    
    # Perform clustering with score estimation
    averages, centers, Nc, ssp = f_tilde_ssp(
        σ_value, 
        data_for_clustering; 
        prob=clustering_prob, 
        verbose=verbose, 
        conv_param=clustering_conv_param, 
        i_max=clustering_max_iter,
        normalization=use_normalization_for_clustering
    )

    if verbose
        println("   • Number of clusters found: $Nc")
    end

    # Generate inputs and targets for neural network
    local inputs_targets, M_averages_values, m_averages_values
    
    if use_normalization_for_clustering
        inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(
            averages, centers, Nc; normalization=true
        )
    else
        inputs_targets = generate_inputs_targets(
            averages, centers, Nc; normalization=false
        )
        M_averages_values, m_averages_values = nothing, nothing
    end
    
    if verbose
        println("   • Training data prepared: $(size(inputs_targets[1])) inputs, $(size(inputs_targets[2])) targets")
    end
    
    # Now we can determine the actual neural network architecture based on the training data
    actual_input_dim = size(inputs_targets[1], 1)
    actual_output_dim = size(inputs_targets[2], 1)
    
    # Construct full neural network architecture
    nn_architecture = [actual_input_dim, hidden_layers..., actual_output_dim]
    
    if verbose
        println("   • Neural network architecture: $nn_architecture")
        println("   • Hidden layers: $hidden_layers")
        println("✅ Clustering complete.")
    end

    # === 3. Neural Network Training ===
    if verbose
        println("🧠 [Step 3/3] Training neural network...")
        println("   • Architecture: $nn_architecture")
        println("   • Hidden layers: $hidden_layers")
        println("   • Epochs: $epochs, Batch size: $batch_size")
        println("   • Using GPU: $use_gpu")
    end
    
    nn_clustered, loss_clustered = train(
        inputs_targets, 
        epochs, 
        batch_size, 
        nn_architecture; 
        use_gpu=use_gpu, 
        activation=activation, 
        last_activation=last_activation,
        opt=optimizer,
        verbose=verbose
    )

    # Post-process the neural network and ensure it's on CPU
    local nn_clustered_cpu
    if use_normalization_for_clustering && M_averages_values !== nothing
        # Create denormalization chain
        denorm_layer = x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values
        nn_clustered_cpu = Chain(nn_clustered, denorm_layer) |> cpu
    else
        nn_clustered_cpu = nn_clustered |> cpu
    end
    
    if verbose
        println("   • Final training loss: $(loss_clustered[end])")
        println("✅ Training complete.")
    end

    # === 4. Create Score Function ===
    if verbose
        println("🎯 Creating final score function...")
    end
    
    # Define the raw score function
    function final_score_function(x)
        return .-nn_clustered_cpu(Float32.(x)) ./ σ_value
    end

    # Define unnormalization helper
    function unnormalize_score(f_norm, x, M, S)
        x_norm = (x .- M) ./ S
        return f_norm(x_norm) ./ S
    end

    # Create the final score function that handles unnormalization
    function score_clustered_nn(x)
        return unnormalize_score(final_score_function, x, M, S)
    end

    if verbose
        println("✅ Score function created.")
        println("🎉 K-GMM Score Estimation Complete!")
        println("=" ^ 50)
    end

    # Return comprehensive results
    return (
        score_function = score_clustered_nn,
        loss_history = loss_clustered,
        averages = averages,
        centers = centers,
        Nc = Nc,
        ssp = ssp
    )
end
"""
    calculate_score_vanilla(raw_timeseries; kwargs...)

Estimate the score function using vanilla score matching without clustering.

This function trains a neural network directly on noisy samples of the data to approximate 
the score function using standard score matching techniques.

# Arguments
- `raw_timeseries::AbstractMatrix`: Input time series data with shape (dimensions, time_points)

## Score Estimation Parameters
- `σ_value::Float64 = 0.05`: Noise standard deviation for score estimation

## Neural Network Training Parameters
- `epochs::Int = 2000`: Number of training epochs
- `batch_size::Int = 32`: Mini-batch size for training
- `hidden_layers::Vector{Int} = [128, 64]`: Hidden layer sizes (input/output dims added automatically)
- `activation = swish`: Activation function for hidden layers
- `last_activation = identity`: Activation function for output layer
- `optimizer = Adam(0.001)`: Optimizer for training
- `use_gpu::Bool = false`: Whether to use GPU acceleration

## Control Parameters
- `verbose::Bool = false`: Whether to show progress information and training details

# Returns
- `NamedTuple` containing:
  - `score_function`: The estimated score function
  - `loss_history`: Training loss history
  - `normalization_params`: Tuple of (mean, std) used for normalization
  - `neural_network`: The trained neural network (on CPU)

# Example
```julia
# Basic usage
result = calculate_score_vanilla(data; verbose=true)
score_fn = result.score_function

# Advanced usage with custom parameters
result = calculate_score_vanilla(data; 
    σ_value=0.1, 
    epochs=5000, 
    hidden_layers=[256, 128],
    use_gpu=true,
    verbose=true
)
```
"""
function calculate_score_vanilla(
    raw_timeseries::AbstractMatrix;
    # Score estimation parameters
    σ_value::Float64 = 0.05,
    # NN Training parameters
    epochs::Int = 2000,
    batch_size::Int = 32,
    hidden_layers::Vector{Int} = [128, 64],
    activation = swish,
    last_activation = identity,
    optimizer = Flux.Adam(0.001),
    use_gpu::Bool = false,
    # Control parameters
    verbose::Bool = false
    )

    if verbose
        println("🎯 Starting Vanilla Score Estimation")
        println("=" ^ 50)
    end

    # === 1. Data Validation and Normalization ===
    if verbose
        println("📊 [Step 1/2] Normalizing input time series...")
        println("   • Input shape: $(size(raw_timeseries))")
    end
    
    # Validate input dimensions
    if size(raw_timeseries, 1) < 1 || size(raw_timeseries, 2) < 2
        throw(ArgumentError("Input data must have at least 1 dimension and 2 time points"))
    end
    
    dim = size(raw_timeseries, 1)
    
    # Construct full neural network architecture
    nn_architecture = [dim, hidden_layers..., dim]
    
    if verbose
        println("   • Neural network architecture: $nn_architecture")
        println("   • Hidden layers: $hidden_layers")
    end
    
    # Compute normalization parameters
    M = mean(raw_timeseries, dims=2)
    S = std(raw_timeseries, dims=2)
    
    # Check for zero standard deviation
    if any(S .≈ 0)
        @warn "Some dimensions have zero standard deviation. Adding small regularization."
        S = S .+ 1e-8
    end
    
    obs = (raw_timeseries .- M) ./ S
    
    if verbose
        println("   • Normalization parameters computed")
        println("   • σ_value: $σ_value")
        println("✅ Normalization complete.")
    end

    # === 2. Neural Network Training ===
    if verbose
        println("🧠 [Step 2/2] Training neural network...")
        println("   • Architecture: $nn_architecture")
        println("   • Hidden layers: $hidden_layers")
        println("   • Epochs: $epochs, Batch size: $batch_size")
        println("   • Using GPU: $use_gpu")
    end
    
    # Train the neural network using vanilla score matching
    nn_vanilla, loss_vanilla = train(
        obs, 
        epochs, 
        batch_size, 
        nn_architecture, 
        σ_value; 
        use_gpu=use_gpu,
        activation=activation,
        last_activation=last_activation,
        opt=optimizer,
        verbose=verbose
    )
    
    # Ensure the network is on CPU for final use
    nn_vanilla_cpu = nn_vanilla |> cpu
    
    if verbose
        println("   • Final training loss: $(loss_vanilla[end])")
        println("✅ Training complete.")
    end

    # === 3. Create Score Function ===
    if verbose
        println("🎯 Creating final score function...")
    end
    
    # Define the raw score function (operates on normalized data)
    function score_normalized(x)
        return .-nn_vanilla_cpu(Float32.(x)) ./ σ_value
    end

    # Define unnormalization helper
    function unnormalize_score(f_norm, x, M, S)
        x_norm = (x .- M) ./ S
        return f_norm(x_norm) ./ S
    end

    # Create the final score function that handles unnormalization
    function score_vanilla_nn(x)
        return unnormalize_score(score_normalized, x, M, S)
    end

    if verbose
        println("✅ Score function created.")
        println("🎉 Vanilla Score Estimation Complete!")
        println("=" ^ 50)
    end

    # Return comprehensive results
    return (
        score_function = score_vanilla_nn,
        loss_history = loss_vanilla,
        normalization_params = (mean=M, std=S),
        neural_network = nn_vanilla_cpu
    )
end

end # module ClustGen