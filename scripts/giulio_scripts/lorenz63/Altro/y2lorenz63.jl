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
using Base.Threads
using DifferentialEquations, Zygote, Optimisers


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
n_epochs = 100
batch_size = 256
n_steps = 10
@show typeof(batch_size)
@show typeof(n_steps)
integral_new = Array{Float32}(undef, size(inputs_train, 1))

#Now I train the model
@time nn, losses = train_y2(inputs_train, n_epochs, batch_size, [m, 50, 25,  m]; opt=Flux.Adam(0.001), activation_hidden=swish, activation_output=identity, use_gpu=false, dt=dt, m=m, n_steps=n_steps)

nn_clustered_cpu = nn |> cpu
losses
p_loss = Plots.plot(losses, xlabel="Epoch", ylabel="Loss", title="NN training loss", linewidth=2, label="Loss")
display(p_loss)

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
        current_embed = vcat(y_next, current_embed[1:end-1])
    end

    return y_pred
end

y0_embed = inputs_validation[1, :] #initial condition delay embedding      
@show size(targets_validation)
dt      = 0.01   
T_total= 100.0                
n_steps = Int(T_total / dt) 

y_pred = predict_scalar_timeseries(nn_clustered_cpu, y0_embed, dt, n_steps)
#@show(y_pred[1:4345])

# Plottiamo la prima componente in funzione del tempo
t = 0:dt:T_total
# plt1 = plot(t[1:2], y_pred[1:2],
#      xlabel="t", ylabel="y₁(t)",
#      title="Predicted trajectory for y₂(t)")
targets_trimmed = targets_validation[1:length(y_pred[1:100])]
plot!(plt1, t[1:100], targets_trimmed,
     label="observed trajectory")

#plot!(plt1, t, , label="observed trajectory", color=:red)


#now I integrate the trained model to compute the trajectory


# while length(y_predicted) - length(inputs[:,1]) < q 
#     for i in 0:m -1
#         push!(new_embedding, inputs[end - i*q])
#     end
# end

# for i in 0:m-1
#     push!(new_embedding, y_predicted[end - i*q])
# end
# current_embedding =  new_embedding
# end
# return y_predicted[start_row:end]
# end
# q = round(Int, τ / dt)
#     n_steps = round(Int, T_final / dt)

#     y_obs = inputs[:, 1]
    
#     #initialize predicted array 
#     y_predicted = copy(y_obs) #prendo la prima colonna dei miei input, che non arà altro che la mia time series osservata
    
#     #initialize the embedding array
#     current_embedding = copy(inputs[start_row, :])
    
#     #initialize the current index
#     current_index = start_row
#     for step in 1:n_steps
#         dy = nn(current_embedding)[1]

#     #integrate dy yielded by the model to get the y predicted at the next time step
#         y_next = current_embedding[1] + dt*dy #questo è y(t+\Delta t)
#         push!(y_predicted, y_next) #vettore che contiene y(t + i \Delt t) al crescere di i fino a T finale, li attacco in coda alla time series osservata per i passaggi successivi del codice

#         if current_index + 1 <= length(y_obs)
#             current_embedding = copy(inputs[current_index + 1, :])
#             current_index += 1
#         else
#             new_embedding =Float64[]
#             push!(new_embedding, y_next)

#             for i in 1:m -1
#                 idx_in_pred = length(y_predicted) - i*q
#                 #if I ended up outside of the initially observed trajectory, I draw (also) from the predicted values
#                 if idx_in_pred > length(y_obs)
#                     push!(new_embedding, y_predicted[idx_in_pred])
#                     #if I'm still covered by the observed trajectory, I draw from the observed values
#                 else
#                     push!(new_embedding, y_obs[idx_in_pred])
#                 end
#             end
#             current_embedding = new_embedding
#             current_index +=1
#         end
#     end
#     return y_predicted[length(y_obs)+1:end]