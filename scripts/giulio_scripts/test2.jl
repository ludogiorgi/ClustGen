using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using MarkovChainHammer
using ClustGen
using KernelDensity
using HDF5
using Flux
using BSON
using Plots
using LinearAlgebra
using ProgressBars
using Distributions

function F(x, t, ε ; µ=10.0, ρ=28.0, β=8/3)
    dy1 = µ/ε^2 * (x[2] - x[1])
    dy2 = 1/ε^2 * (x[1] * (ρ - x[3]) - x[2])
    dy3 = 1/ε^2 * (x[1] * x[2] - β * x[3])
    return [dy1, dy2, dy3]
end

function sigma(x, t; noise = 0.0)
    sigma1 = noise
    sigma2 = noise
    sigma3 = noise #Added: This is for the 4th variable
    return [sigma1, sigma2, sigma3]
end

function delay_embedding(x; τ, m)

    #number of time steps per every delay embedding step
    q =  round(Int, τ / dt)  

    start_idx = 1 + (m - 1) * q 
    Z = []
    for i in start_idx:length(x)
            z_i = [x[i - j*q] for j in 0:m-1]
            push!(Z, z_i)
    end
    
    return transpose(hcat(Z...))
end

function generate_inputs_targets_y2(Z, obs_nn; τ, dt, m)

    q =  round(Int, τ / dt)

    inputs = Z

    targets = obs_nn[2, 1 + (m-1)*q:end]

    return (inputs, targets)
end


function  predict_trajectory(model, y0, dt, n_steps)
    n_feat = length(y0)
    y_pred=zeros(eltype(y0), n_feat, n_steps+1)
    y_pred[:,i] = y0
    for i in 1:n_steps
        dy = model(y_pred[:,i])
        y_pred[:,i+1] = y_pred[:,i] + dt*dy
    end
    return y_pred
end

function predict_scalar_timeseries(model, y0_embed, dt, n_steps)
    m = length(y0_embed)             # dimensione dell'embedding
    y_pred = zeros(eltype(y0_embed), n_steps+1)
    y_pred[1] = y0_embed[1]        # valore più recente dell'embedding

    current_embed = copy(y0_embed)

    for i in 1:n_steps
        next_embed = model(current_embed)              # output è embedding predetto
        y_next = next_embed[1]                       # prendi l’ultimo elemento = y(t+T)
        y_pred[i+1] = y_next

        # aggiorna embedding: shift a sinistra e aggiungi nuovo valore
        current_embed = vcat(current_embed[2:end], y_next)
    end

    return y_pred
end

function create_nn_giulio(layers_size::Vector{Int}; activation_hidden=swish, activation_output=identity)
    #create an array to store the layers
    layers = Vector{Any}()

    #create hidden layers
    for i in 1:length(layers_size)-2
        push!(layers, Dense(layers_size[i], layers_size[i+1], activation_hidden))
    end

    #create output layer
    push!(layers, Dense(layers_size[end-1], layers_size[end], activation_output))


    return Chain(
        layers...,
        x -> reshape(x, :)   # <-- rende output sempre vettore (m,) piatto
        )
end

function make_batches_NODE(X, batch_size)
    n_total = size(X, 2)
    idx = shuffle(1:n_total)

    batches = Vector{Matrix{Float32}}()

    for i in 1:batch_size:length(idx)
        batch_idx = idx[i:min(i + batch_size - 1, end)]
        inputs_batch = hcat([Float32.(X[:, j]) for j in batch_idx]...)  # m × batch_size
        push!(batches, inputs_batch)
    end

    return batches  # List of m × batch_size matrices
end


function debug_integral(x, i)
    println("Step $i - integral_old = ", x)
end

Zygote.@nograd debug_integral


function loss_fn_y2(model, inputs_batch; dt=dt, n_steps=n_steps)

    batch_size = size(inputs_batch, 2) #number of columns in inputs_batch
    diffs_sq = Float32[]
    integral_new = Array{Float32}(undef, size(inputs_batch, 1)) #initialize the array for the delay embedding of y(t+dt)

    for i in 1:(batch_size - n_steps)
        integral_old = inputs_batch[:,i] #delay embedding of  y(t), first delay embedding in the batch
        
        for i in 1:n_steps #integrate over n_steps steps
            dy = model(integral_old) #compute the delay embedding of \dot y(t) using the model
            integral_new = integral_old .+ dt .* dy #compute the delay embedding of y(t+dt) integrating the model
            integral_old = integral_new #update the delay embedding of y(t) to the new value
            debug_integral(integral_old, i) #debugging line
        end
        diff = sum((integral_old .- inputs_batch[:, i + n_steps]).^2)
        push!(diffs_sq, diff) #compute the difference between the delay embedding of y(t+dt) and the delay embedding of y(t+dt) observed by integrating lorenz63 as the quadratic distance between the two vectors
    end
    loss = sqrt(mean(diffs_sq)) #the loss is the square root mean over the batch of the difference squared between the delay embedding of y(t+dt) and the delay embedding of y(t+dt) observed by integrating lorenz63

    return loss 
end




#integrate the first 3 variables of the lorenz system 
ndim = 1
dt = 0.01
Nsteps=10000000
ε=0.5
f = (x, t) -> F(x, t, ε)
obs_nn = evolve(randn(3), dt, Nsteps, f, sigma; timestepper=:rk4, resolution=100)
size(obs_nn)
@show length(obs_nn[:, 2])


#delay embedding of y2(t)
m = 10
τ=0.1
#q = round(Int, τ / dt)
Z = delay_embedding(obs_nn[2,:]; τ=τ, m=m)

#generate data for training and validation

#NB before that we need to create a variant of generate_inputs_targets that creates a Tuple with the delay embeddig as 1st entry and the y_obs(t) as 2nd entry.

#inputs in this case will be m-dimensional vectors containing the delay embedding of y2(t) for every t. Also we want to create a training set with 80% of the data and a validation set with 20% of the data.
inputs_targets = generate_inputs_targets_y2(Z, obs_nn; τ=τ, dt=dt, m=m)
@show size(inputs_targets[2])
inputs_train = inputs_targets[1][1:floor(Int, 0.8*size(inputs_targets[1], 1)), :]
inputs_validation = inputs_targets[1][floor(Int, 0.8*size(inputs_targets[1], 1))+1:end, :]

#targets in this case will be 1-dimensional vectors containing y_obs(t)
targets_train = inputs_targets[2][1:floor(Int, 0.8*length(inputs_targets[2]))]
targets_validation = inputs_targets[2][floor(Int, 0.8*length(inputs_targets[2]))+1:end]

inputs_train  = permutedims(inputs_train) 
inputs_train = Float32.(inputs_train) # (m, Ntrain)
#targets_train = reshape(targets_train, 1, :)       # (1, Ntrain)
length(inputs_train[1,:])
targets_train[end]
# inputs_val    = permutedims(inputs_validation)    
# targets_val   = reshape(targets_validation, 1, :)
#set nn parameters
n_epochs = 500
batch_size = 128
n_steps = 50

use_gpu = false
device = (use_gpu && CUDA.functional()) ? gpu : cpu
    println("Using $(device === gpu ? "GPU" : "CPU")")


nn = create_nn_giulio([m, 50, 25,  m]; activation_hidden=swish, activation_output=identity)|> device
loss = loss_fn_y2(nn, inputs_train; dt=dt, n_steps=n_steps)