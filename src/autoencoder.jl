#=
   AUTOENCODER MODULE
   
   This module provides functionality for creating, training, and utilizing autoencoders
   for dimensionality reduction in dynamical systems data. It includes functions for 
   encoder/decoder creation, training, application, and model persistence.
=#

"""
    create_encoder(neurons::Vector{Int}; activation = tanh)

Creates a feed-forward neural network encoder with the specified architecture.

# Arguments
- `neurons::Vector{Int}`: Vector specifying number of neurons in each layer
- `activation`: Activation function to use in all layers (default: tanh)

# Returns
- A Flux.Chain object representing the encoder network
"""
function create_encoder(neurons::Vector{Int}; activation = tanh)
    layers = []
    for i in 1:length(neurons)-1
        push!(layers, Flux.Dense(neurons[i], neurons[i+1], activation))
    end
    return Flux.Chain(layers...)
end


"""
    create_decoder(neurons::Vector{Int}; activation = tanh, output_activation = sigmoid)

Creates a feed-forward neural network decoder with the specified architecture.

# Arguments
- `neurons::Vector{Int}`: Vector specifying number of neurons in each layer
- `activation`: Activation function to use in hidden layers (default: tanh)
- `output_activation`: Activation function for the output layer (default: sigmoid)

# Returns
- A Flux.Chain object representing the decoder network
"""
function create_decoder(neurons::Vector{Int}; activation = tanh, output_activation = sigmoid)
    layers = []
    
    # Add all hidden layers with the specified activation function
    for i in 1:length(neurons)-2
        push!(layers, Flux.Dense(neurons[i], neurons[i+1], activation))
    end
    
    # Add the output layer with a potentially different activation
    push!(layers, Flux.Dense(neurons[end-1], neurons[end], output_activation))
    
    return Flux.Chain(layers...)
end


"""
    loss_autoencoder(autoencoder, x)

Computes the mean squared error loss between autoencoder output and input.

# Arguments
- `autoencoder`: The autoencoder model (encoder + decoder)
- `x`: Input data to the autoencoder

# Returns
- Mean squared error between autoencoder(x) and x
"""
loss_autoencoder(autoencoder, x) = Flux.mse(autoencoder(x), x)


"""
    train_autoencoder!(autoencoder, train_loader, optimizer, epochs)

Trains an autoencoder using the provided data loader and optimizer.

# Arguments
- `autoencoder`: The autoencoder model to train
- `train_loader`: DataLoader containing training data batches
- `optimizer`: Flux optimizer to use (e.g., ADAM)
- `epochs`: Number of training epochs

# Side Effects
- Updates the autoencoder parameters in-place
- Prints training progress after each epoch
"""
function train_autoencoder!(autoencoder, train_loader, optimizer, epochs)
    for epoch in 1:epochs
        for batch in train_loader
            grads = Flux.gradient(Flux.params(autoencoder)) do
                loss = loss_autoencoder(autoencoder, batch)
            end
            Flux.Optimise.update!(optimizer, Flux.params(autoencoder), grads)
        end
        
        # Print progress after each epoch
        println("Epoch: $epoch, Loss: $(loss_autoencoder(autoencoder, train_loader |> first))")
    end
end


"""
    apply_autoencoder(data, neurons, epochs, batch_size; kwargs...)

Creates, trains, and optionally saves an autoencoder model for the input data.

# Arguments
- `data`: Training data for the autoencoder
- `neurons`: Vector specifying network architecture (e.g., [input_dim, hidden_dim, latent_dim])
- `epochs`: Number of training epochs
- `batch_size`: Batch size for training

# Keyword Arguments
- `activation`: Activation function for hidden layers (default: tanh)
- `output_activation`: Activation function for output layer (default: sigmoid)
- `optimizer`: Flux optimizer (default: Adam(0.001))
- `file_name`: If provided, saves encoder/decoder to disk with this name

# Returns
- Tuple of (encoder, decoder) networks
"""
function apply_autoencoder(data, neurons, epochs, batch_size; 
                          activation = tanh, 
                          output_activation = sigmoid, 
                          optimizer = Adam(0.001), 
                          file_name = false)
    # Create encoder and decoder networks
    encoder = create_encoder(neurons; activation=activation)
    decoder = create_decoder(reverse(neurons); activation=activation, output_activation=output_activation)
    
    # Combine into autoencoder
    autoencoder = Flux.Chain(encoder, decoder)
    
    # Create data loader for batching
    data_loader = Flux.DataLoader(data, batchsize=batch_size, shuffle=true)
    
    # Train the autoencoder
    train_autoencoder!(autoencoder, data_loader, optimizer, epochs)
    
    # Optionally save the model components
    if file_name != false
        BSON.@save pwd() * "/NNs/encoder_$(file_name).bson" encoder
        BSON.@save pwd() * "/NNs/decoder_$(file_name).bson" decoder
    end
    
    return encoder, decoder
end


"""
    read_autoencoder(file_name)

Loads a previously saved encoder and decoder from disk.

# Arguments
- `file_name`: Base name used when saving the encoder/decoder

# Returns
- Tuple of (encoder, decoder) networks loaded from disk
"""
function read_autoencoder(file_name)
    encoder = BSON.load(pwd() * "/NNs/encoder_$(file_name).bson")[:encoder]
    decoder = BSON.load(pwd() * "/NNs/decoder_$(file_name).bson")[:decoder]
    return encoder, decoder
end