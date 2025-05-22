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



#Define the rhs of the Lorenz system for later integration with evolve function, changes wrt GMM_Lorenz63: x is a 4 dimensional vector that contains x, y1,y2 and y3.
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

inputs_train  = permutedims(inputs_train)          # (m, Ntrain)
targets_train = reshape(targets_train, 1, :)       # (1, Ntrain)
length(inputs_train[1,:])
targets_train[end]
# inputs_val    = permutedims(inputs_validation)    
# targets_val   = reshape(targets_validation, 1, :)
#set nn parameters
n_epochs = 5000
batch_size = 32
n = 10
#Now I train the model
@time nn, losses = train_y2(inputs_train, targets_train, n_epochs, batch_size, [m, 50, 25,  m]; opt=Flux.Adam(0.001), activation_hidden=swish, activation_output=identity, use_gpu=false, dt=dt, m=m)

nn_clustered_cpu = nn |> cpu
