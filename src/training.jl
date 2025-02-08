# Generate training data for vanilla loss and reverse sampling method
function generate_txz(y, σ; ϵ=0.05)
    t = rand!(similar(y, size(y)[1])) .* (1 - ϵ) .+ ϵ
    σ_t = σ(t)
    z = randn!(similar(y))
    x = @. y + σ_t * z
    return t, x, z
end

# Generate training data for the other cases
function generate_xz(y, σ)
    z = randn!(similar(y))
    x = @. y + σ * z
    return x, z
end

# Generate inputs and targets for vanilla loss and reverse sampling
function generate_data_t(obs, σ; ϵ=0.05)
    t, x, z = generate_txz(obs', σ, ϵ=ϵ)
    inputs = hcat([[x[i, :]...,t[i]] for i in 1:size(obs, 2)]...)
    targets = hcat([z[i, :] for i in 1:size(obs, 2)]...)
    return inputs, targets
end

# Generate inputs and targets for the other cases
function generate_data(obs, σ)
    x, z = generate_xz(obs', σ)
    inputs = hcat([[x[i, :]...] for i in 1:size(obs, 2)]...)
    targets = hcat([z[i, :] for i in 1:size(obs, 2)]...)
    return inputs, targets
end

# Create a NN with the given number of neurons and activation functions
function create_nn(neurons::Vector{Int}; activation = swish, last_activation = identity)
    layers = Vector{Any}(undef, length(neurons) - 1)
    for i in 1:length(neurons)-2
        layers[i] = Flux.Dense(neurons[i], neurons[i+1], activation)
    end
    layers[end] = Flux.Dense(neurons[end-1], neurons[end], last_activation)
    return Flux.Chain(layers...)
end

# Loss function
function loss_score(nn, inputs, targets)
    predictions = nn(inputs)  
    return Flux.mse(predictions, targets) 
end

# Train the NN with vanilla loss
function train(obs, n_epochs, batch_size, neurons::Vector{Int}, σ; opt=Adam(0.001), activation=swish, last_activation = identity, ϵ=0.05, use_gpu=true)
    device = (use_gpu && CUDA.functional()) ? gpu : cpu
    println("Using $(device === gpu ? "GPU" : "CPU")")
    nn = create_nn(neurons, activation=activation, last_activation=last_activation) |> device
    losses = []
    for epoch in ProgressBar(1:n_epochs) 
        if isa(σ, Float64)      # Langevin sampling method
            inputs, targets = generate_data(obs, σ)
        else                    # Reverse sampling method
            inputs, targets = generate_data_t(obs, σ, ϵ=ϵ)
        end
        data_loader = Flux.DataLoader((inputs, targets), batchsize=batch_size, shuffle=true) 
        epoch_loss = 0.0
        for (batch_inputs, batch_targets) in data_loader
            batch_inputs = batch_inputs |> device
            batch_targets = batch_targets |> device
            gs = Flux.gradient(() -> loss_score(nn, batch_inputs, batch_targets), Flux.params(nn))
            for (param, grad) in zip(Flux.params(nn), gs)
                Flux.Optimisers.update!(opt, param, grad)
            end
            epoch_loss += loss_score(nn, batch_inputs, batch_targets)
        end
        push!(losses, epoch_loss / length(data_loader))
    end
    return nn, losses
end

# Further training of the NN with vanilla loss
function train(obs, n_epochs, batch_size, nn::Chain, σ; opt=Adam(0.001), ϵ=0.05, use_gpu=true)
    device = (use_gpu && CUDA.functional()) ? gpu : cpu
    println("Using $(device === gpu ? "GPU" : "CPU")")
    nn |> device
    losses = []
    for epoch in ProgressBar(1:n_epochs) 
        if isa(σ, Float64)     # Langevin sampling method
            inputs, targets = generate_data(obs, σ)
        else                  # Reverse sampling method
            inputs, targets = generate_data_t(obs, σ, ϵ=ϵ)
        end
        data_loader = Flux.DataLoader((inputs, targets), batchsize=batch_size, shuffle=true) 
        epoch_loss = 0.0
        for (batch_inputs, batch_targets) in data_loader
            batch_inputs = batch_inputs |> device
            batch_targets = batch_targets |> device
            gs = Flux.gradient(() -> loss_score(nn, batch_inputs, batch_targets), Flux.params(nn))
            for (param, grad) in zip(Flux.params(nn), gs)
                Flux.Optimisers.update!(opt, param, grad)
            end
            epoch_loss += loss_score(nn, batch_inputs, batch_targets)
        end
        push!(losses, epoch_loss / length(data_loader))
    end
    return nn, losses
end

# Train the NN with clustering loss
function train(obs, n_epochs, batch_size, neurons; opt=Adam(0.001), activation=swish, last_activation = identity, use_gpu=true)
    device = (use_gpu && CUDA.functional()) ? gpu : cpu
    println("Using $(device === gpu ? "GPU" : "CPU")")
    nn = create_nn(neurons, activation=activation, last_activation=last_activation) |> device
    losses = []
    for epoch in ProgressBar(1:n_epochs) 
        inputs, targets = obs
        data_loader = Flux.DataLoader((inputs, targets), batchsize=batch_size, shuffle=true) 
        epoch_loss = 0.0
        for (batch_inputs, batch_targets) in data_loader
            batch_inputs = batch_inputs |> device
            batch_targets = batch_targets |> device
            gs = Flux.gradient(() -> loss_score(nn, batch_inputs, batch_targets), Flux.params(nn))
            for (param, grad) in zip(Flux.params(nn), gs)
                Flux.Optimisers.update!(opt, param, grad)
            end
            epoch_loss += loss_score(nn, batch_inputs, batch_targets)
        end
        push!(losses, epoch_loss / length(data_loader))
    end
    return nn, losses
end

# Check vanilla loss of a NN 
function check_loss(obs, nn, σ; ϵ=0.05, n_samples=1)
    loss = 0.0
    for _ in ProgressBar(1:n_samples) 
        if isa(σ, Float64)
            inputs, targets = generate_data(obs, σ)
        else
            inputs, targets = generate_data_t(obs, σ, ϵ=ϵ)
        end
        loss += loss_score(nn, inputs, targets)
    end
    return loss/n_samples
end