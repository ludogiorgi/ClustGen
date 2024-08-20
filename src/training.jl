function generate_txz(y, σ; ϵ=0.05)
    t = rand!(similar(y, size(y)[1])) .* (1 - ϵ) .+ ϵ
    σ_t = σ(t)
    z = randn!(similar(y))
    x = @. y + σ_t * z
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

function train_clustered(data, n_epochs, batch_size, neurons; opt=ADAM(0.001), activation=swish)
    nn = create_nn(neurons, activation=activation)
    losses = []
    data_loader = Flux.DataLoader(data, batchsize=batch_size, shuffle=true)
    for epoch in ProgressBar(1:n_epochs)
        loss = 0
        for batch in data_loader
            loss += loss_score(nn, batch)
            gs = gradient(Flux.params(nn)) do
                loss_score(nn, batch)
            end
            Flux.Optimise.update!(opt, Flux.params(nn), gs)
        end
        push!(losses, loss/length(data_loader))
    end
    return nn, losses
end

function train_vanilla(obs, n_epochs, batch_size, neurons, σ; opt=ADAM(0.001), activation=swish, ϵ=0.05)
    nn = create_nn(neurons, activation=activation)
    losses = []
    μ = obs'
    Dim = size(obs)[1]
    losses = []
    for epoch in ProgressBar(1:n_epochs)
        t, x, z = generate_txz(μ, σ, ϵ=ϵ)
        data = Flux.Float32.(hcat([[x[i,:]..., t[i], z[i,:]...] for i in 1:size(μ)[1]]...))
        data_inout = [(data[1:Dim+1, i], .- data[Dim+2:2*Dim+1, i]) for i in 1:size(μ)[1]]
        data_loader = Flux.DataLoader(data_inout, batchsize=batch_size, shuffle=true)
        loss = 0
        for batch in data_loader
            loss += loss_score(nn, batch)
            gs = gradient(Flux.params(nn)) do
                loss_score(nn, batch)
            end
            Flux.Optimise.update!(opt, Flux.params(nn), gs)
        end
        push!(losses, loss/length(data_loader))
    end
    return nn, losses
end

function check_loss(obs, nn, σ; ϵ=0.05)
    Dim = size(obs)[1]
    μ = obs'
    t, x, z = generate_txz(μ, σ; ϵ=ϵ)
    data = Flux.Float32.(hcat([[x[i,:]..., t[i], z[i,:]...] for i in 1:size(μ)[1]]...))
    data_inout = [(data[1:Dim+1, i], .- data[Dim+2:2*Dim+1, i]) for i in 1:size(μ)[1]]
    loss = 0.0
    for (x, y) in data_inout
        loss += Flux.mse(nn(x[1:Dim], x[Dim+1]), y)
    end
    return loss/length(data_inout)
end