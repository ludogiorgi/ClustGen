
##################### AUTOENCODERS TO REDUCE DATA DIMENSIONALITY #####################


function create_encoder(neurons::Vector{Int}; activation = tanh)
    layers = []
    for i in 1:length(neurons)-1
        push!(layers, Flux.Dense(neurons[i], neurons[i+1], activation))
    end
    return Flux.Chain(layers...)
end

function create_decoder(neurons::Vector{Int}; activation = tanh, output_activation = sigmoid)
    layers = []
    for i in 1:length(neurons)-2
        push!(layers, Flux.Dense(neurons[i], neurons[i+1], activation))
    end
    push!(layers, Flux.Dense(neurons[end-1], neurons[end], output_activation))
    return Flux.Chain(layers...)
end

loss_autoencoder(autoencoder, x) = Flux.mse(autoencoder(x), x)

function train_autoencoder!(autoencoder, train_loader, optimizer, epochs)
    for epoch in 1:epochs
        for batch in train_loader
            grads = Flux.gradient(Flux.params(autoencoder)) do
                loss = loss_autoencoder(autoencoder, batch)
            end
            Flux.Optimise.update!(optimizer, Flux.params(autoencoder), grads)
        end
        println("Epoch: $epoch, Loss: $(loss_autoencoder(autoencoder, train_loader |> first))")
    end
end

function apply_autoencoder(data, neurons, epochs, batch_size; activation = tanh, output_activation = sigmoid, optimizer = Flux.ADAM(0.001), file_name = false)
    encoder = create_encoder(neurons; activation=activation)
    decoder = create_decoder(reverse(neurons); activation=activation, output_activation=output_activation)
    autoencoder = Flux.Chain(encoder, decoder)
    data_loader = Flux.DataLoader(data, batchsize=batch_size, shuffle=true)
    train_autoencoder!(autoencoder, data_loader, optimizer, epochs)
    if file_name != false
        BSON.@save pwd() * "/NNs/encoder_$(file_name).bson" encoder
        BSON.@save pwd() * "/NNs/decoder_$(file_name).bson" decoder
    end
    return encoder, decoder
end

function read_autoencoder(file_name)
    encoder = BSON.load(pwd() * "/NNs/encoder_$(file_name).bson")[:encoder]
    decoder = BSON.load(pwd() * "/NNs/decoder_$(file_name).bson")[:decoder]
    return encoder, decoder
end
