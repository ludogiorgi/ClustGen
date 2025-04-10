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

##

# function F(x, t; F0=4.0, nu=1.0, c=10.0, b=10.0, Nk=4, Nj=8)
#     # Coupling constant: c1 = c/b
#     c1 = c / b

#     # Allocate arrays for the derivatives of the slow and fast variables.
#     dx = zeros(Nk)
#     dy = zeros(Nk, Nj)
    
#     # Extract the slow variables xₖ from the state vector.
#     x_slow = x[1:Nk]
    
#     # Extract the fast variables yₖ,ⱼ.
#     # They are assumed to be stored after the slow variables.
#     # Reshape them into an Nk×Nj matrix, where the k-th row corresponds to the block for xₖ.
#     x_fast = reshape(x[Nk+1:end], (Nj, Nk))'  # now x_fast[k, j] corresponds to yₖ,ⱼ

#     # Compute the forcing for the slow variables (Eq. 10):
#     # dxₖ/dt = - xₖ₋₁ (xₖ₋₂ - xₖ₊₁) - nu*xₖ + F0 + c1 * (sum of fast variables for mode k)
#     for k in 1:Nk
#         # Use periodic boundary conditions:
#         # For index arithmetic in Julia (1-indexed):
#         im1 = mod(k - 2, Nk) + 1  # index for xₖ₋₁
#         im2 = mod(k - 3, Nk) + 1  # index for xₖ₋₂
#         ip1 = mod(k, Nk) + 1      # index for xₖ₊₁
#         dx[k] = - x_slow[im1]*(x_slow[im2] - x_slow[ip1]) - nu*x_slow[k] + F0 + c1*sum(x_fast[k, :])
#     end

#     # Compute the forcing for the fast variables (Eq. 11):
#     # dyₖ,ⱼ/dt = - c*b * yₖ,ⱼ₊₁ (yₖ,ⱼ₊₂ - yₖ,ⱼ₋₁) - c*nu*yₖ,ⱼ + c1*xₖ
#     for k in 1:Nk
#         for j in 1:Nj
#             # Periodic indices in the fast sub-block:
#             jm1 = mod(j - 2, Nj) + 1   # index for yₖ,ⱼ₋₁
#             jp1 = mod(j, Nj) + 1       # index for yₖ,ⱼ₊₁
#             jp2 = mod(j + 1, Nj) + 1     # index for yₖ,ⱼ₊₂
#             dy[k, j] = - c*b * x_fast[k, jp1]*(x_fast[k, jp2] - x_fast[k, jm1]) -
#                        c*nu * x_fast[k, j] + c1*x_slow[k]
#         end
#     end

#     # Combine the slow and fast derivatives into a single vector.
#     # The slow derivatives come first, then the fast derivatives (flattened in row-major order).
#     return vcat(dx, vec(transpose(dy)))
# end

function F(x, t; F0=4.0, nu=1.0, c=10.0, b=1.0, Nk=4, Nj=0)
    # Coupling constant: c1 = c/b
    c1 = 1.0

    # Allocate arrays for the derivatives of the slow and fast variables.
    dx = zeros(Nk)
    dy = zeros(Nk, Nj)
    
    # Extract the slow variables xₖ from the state vector.
    x_slow = x[1:Nk]
    
    # Extract the fast variables yₖ,ⱼ.
    # They are assumed to be stored after the slow variables.
    # Reshape them into an Nk×Nj matrix, where the k-th row corresponds to the block for xₖ.
    x_fast = reshape(x[Nk+1:end], (Nj, Nk))'  # now x_fast[k, j] corresponds to yₖ,ⱼ

    # Compute the forcing for the slow variables (Eq. 10):
    # dxₖ/dt = - xₖ₋₁ (xₖ₋₂ - xₖ₊₁) - nu*xₖ + F0 + c1 * (sum of fast variables for mode k)
    for k in 1:Nk
        # Use periodic boundary conditions:
        # For index arithmetic in Julia (1-indexed):
        im1 = mod(k - 2, Nk) + 1  # index for xₖ₋₁
        im2 = mod(k - 3, Nk) + 1  # index for xₖ₋₂
        ip1 = mod(k, Nk) + 1      # index for xₖ₊₁
        dx[k] = - x_slow[im1]*(x_slow[im2] - x_slow[ip1]) - nu*x_slow[k] + F0 - c1*sum(x_fast[k, :])
    end

    # Compute the forcing for the fast variables (Eq. 11):
    # dyₖ,ⱼ/dt = - c*b * yₖ,ⱼ₊₁ (yₖ,ⱼ₊₂ - yₖ,ⱼ₋₁) - c*nu*yₖ,ⱼ + c1*xₖ
    for k in 1:Nk
        for j in 1:Nj
            # Periodic indices in the fast sub-block:
            jm1 = mod(j - 2, Nj) + 1   # index for yₖ,ⱼ₋₁
            jp1 = mod(j, Nj) + 1       # index for yₖ,ⱼ₊₁
            jp2 = mod(j + 1, Nj) + 1     # index for yₖ,ⱼ₊₂
            dy[k, j] = - c*b * x_fast[k, jp1]*(x_fast[k, jp2] - x_fast[k, jm1]) -
                       c*nu * x_fast[k, j] + c1*x_slow[k]
        end
    end

    # Combine the slow and fast derivatives into a single vector.
    # The slow derivatives come first, then the fast derivatives (flattened in row-major order).
    return vcat(dx, vec(transpose(dy)))
end


function sigma(x, t; noise = 1.0)
    return noise
end

function normalize_f(f, x, t, M, S)
    return f(x .* S .+ M, t) .* S
end

dim = 4
dim_full = 4
dt = 0.01
Nsteps = 10000000
obs_nn = evolve(0.01 .* randn(dim_full), dt, Nsteps, F, sigma; resolution = 1)[1:dim,:]

M = mean(obs_nn, dims=2)
S = std(obs_nn, dims=2)
obs = (obs_nn .- M) ./ S

# kde_true_12 = kde((obs[1,:], obs[2,:]))
# kde_true_13 = kde((obs[1,:], obs[3,:]))
# kde_true_14 = kde((obs[1,:], obs[4,:]))

# plt1 = Plots.heatmap(kde_true_12.x, kde_true_12.y, kde_true_12.density, xlabel="X", ylabel="Y", title="True PDF")
# plt2 = Plots.heatmap(kde_true_13.x, kde_true_13.y, kde_true_13.density, xlabel="X", ylabel="Y", title="True PDF")
# plt3 = Plots.heatmap(kde_true_14.x, kde_true_14.y, kde_true_14.density, xlabel="X", ylabel="Y", title="True PDF")
# Plots.plot(plt1, plt2, plt3, layout=(1, 3), size=(1200, 400))
# ##

autocov_obs = zeros(dim, 1000)
for i in 1:dim
    autocov_obs[i,:] = autocovariance(obs[i,:]; timesteps=1000)
end

autocov_obs_mean = mean(autocov_obs, dims=1)

plotly()
Plots.plot(autocov_obs_mean[1,:], label="X", xlabel="Lag", ylabel="Autocovariance", title="Autocovariance of Observed Trajectory")

##
obs_uncorr = obs[:, 1:1:end]

plotly()
Plots.scatter(obs_uncorr[1,1:100:end], obs_uncorr[2,1:100:end], obs_uncorr[3,1:100:end], markersize=1, label="", xlabel="X", ylabel="Y", title="Observed Trajectory")

##
############################ CLUSTERING ####################

normalization = false
σ_value = 0.05

averages, _, centers, Nc, ssp = f_tilde_ssp(σ_value, obs_uncorr; prob=0.00005, do_print=true, conv_param=0.2, normalization=normalization)

if normalization == true
    inputs_targets, M_averages_values, m_averages_values = generate_inputs_targets(averages, centers, Nc; normalization=true)
else
    inputs_targets = generate_inputs_targets(averages, centers, Nc; normalization=false)
end

targets_norm = [norm(averages[:,i]) for i in eachindex(centers[1,:])]
Plots.scatter(centers[1,:], centers[2,:], marker_z=targets_norm, color=:viridis)

##
#################### TRAINING WITH CLUSTERING LOSS ####################

@time nn_clustered, loss_clustered = train(inputs_targets, 1000, 64, [dim, 128, 64, dim]; use_gpu=true, activation=swish, last_activation=identity)
if normalization == true
    nn_clustered_cpu  = Chain(nn_clustered, x -> x .* (M_averages_values .- m_averages_values) .+ m_averages_values) |> cpu
else
    nn_clustered_cpu = nn_clustered |> cpu
end
score_clustered(x) = .- nn_clustered_cpu(Float32.([x...]))[:] ./ σ_value
Plots.plot(loss_clustered)


##
#################### SAMPLES GENERATION ####################

score_gen(x) = score_clustered(x)

score_gen_xt(x,t) = score_gen(x)
sigma_I(x,t) = 1.0

trj_clustered = evolve(zeros(dim), 0.1*dt, 1000000, score_gen_xt, sigma_I; timestepper=:rk4, resolution=2, boundary=[-100,100])
# trj_score = evolve([0.0, 0.0], dt, 1000000, score_true, sigma_I; timestepper=:rk4, resolution=10, boundary=[-100,100])

kde_clustered_1 = kde(trj_clustered[1,:])
kde_true_1 = kde(obs[1,:])

kde_clustered_2 = kde(trj_clustered[2,:])
kde_true_2 = kde(obs[2,:])

kde_clustered_3 = kde(trj_clustered[3,:])
kde_true_3 = kde(obs[3,:])

kde_clustered_4 = kde(trj_clustered[4,:])
kde_true_4 = kde(obs[4,:])

kde_clustered_y = (kde_clustered_1.density .+ kde_clustered_2.density .+ kde_clustered_3.density .+ kde_clustered_4.density) ./ 4
kde_clustered_x = ([kde_clustered_1.x...] .+ [kde_clustered_2.x...] .+ [kde_clustered_3.x...] .+ [kde_clustered_4.x...]) ./ 4

kde_true_y = (kde_true_1.density .+ kde_true_2.density .+ kde_true_3.density .+ kde_true_4.density) ./ 4
kde_true_x = ([kde_true_1.x...] .+ [kde_true_2.x...] .+ [kde_true_3.x...] .+ [kde_true_4.x...]) ./ 4

Plots.plot(kde_clustered_x, kde_clustered_y, label="Observed", xlabel="X", ylabel="Density", title="Observed PDF")
Plots.plot!(kde_true_x, kde_true_y, label="True", xlabel="X", ylabel="Density", title="True PDF")

##
# Compute bivariate PDFs for consecutive variables
kde_true_12 = kde((obs[1,:], obs[2,:]))
kde_clustered_12 = kde((trj_clustered[1,:], trj_clustered[2,:]))

kde_true_23 = kde((obs[2,:], obs[3,:]))
kde_clustered_23 = kde((trj_clustered[2,:], trj_clustered[3,:]))

kde_true_34 = kde((obs[3,:], obs[4,:]))
kde_clustered_34 = kde((trj_clustered[3,:], trj_clustered[4,:]))

kde_true_41 = kde((obs[4,:], obs[1,:]))
kde_clustered_41 = kde((trj_clustered[4,:], trj_clustered[1,:]))

# Compute bivariate PDFs for variables with one in between
kde_true_13 = kde((obs[1,:], obs[3,:]))
kde_clustered_13 = kde((trj_clustered[1,:], trj_clustered[3,:]))

kde_true_24 = kde((obs[2,:], obs[4,:]))
kde_clustered_24 = kde((trj_clustered[2,:], trj_clustered[4,:]))

# Compute average PDFs for consecutive variables
kde_true_consecutive_density = (kde_true_12.density + kde_true_23.density + kde_true_34.density + kde_true_41.density) ./ 4
kde_clustered_consecutive_density = (kde_clustered_12.density + kde_clustered_23.density + kde_clustered_34.density + kde_clustered_41.density) ./ 4

# Use one of the grids for plotting (they should be similar)
kde_consecutive_x = kde_true_12.x
kde_consecutive_y = kde_true_12.y

# Compute average PDFs for variables with one in between
kde_true_skip_density = (kde_true_13.density + kde_true_24.density) ./ 2
kde_clustered_skip_density = (kde_clustered_13.density + kde_clustered_24.density) ./ 2

# Use one of the grids for plotting
kde_skip_x = kde_true_13.x
kde_skip_y = kde_true_13.y

# Plot the results
plt1 = Plots.heatmap(kde_consecutive_x, kde_consecutive_y, kde_true_consecutive_density, 
                     xlabel="X", ylabel="Y", title="True PDF (Consecutive)")
plt2 = Plots.heatmap(kde_consecutive_x, kde_consecutive_y, kde_clustered_consecutive_density, 
                    xlabel="X", ylabel="Y", title="Generated PDF (Consecutive)")
plt3 = Plots.heatmap(kde_skip_x, kde_skip_y, kde_true_skip_density, 
                    xlabel="X", ylabel="Y", title="True PDF (Skip-One)")
plt4 = Plots.heatmap(kde_skip_x, kde_skip_y, kde_clustered_skip_density, 
                    xlabel="X", ylabel="Y", title="Generated PDF (Skip-One)")
Plots.plot(plt1, plt2, plt3, plt4, layout=(2, 2), size=(800, 800))

##

using ClustGen
f(t) = 1.0

res_trj = 4
steps_trj = 100000
trj = obs[:,1:res_trj:steps_trj*res_trj]

ϵ = 0.05

function u(x)
    U = zeros(dim_full)
    U[1] = ϵ 
    return U
end

function u_score(x)
    U = zeros(dim)
    U[1] = ϵ 
    return U
end

div_u(x) = 0.0
invC0 = inv(cov(obs'))
score_qG(x) = - invC0*x

dim_Obs = 4
n_tau = 100

R_num, δObs_num = zeros(4, dim_Obs, n_tau+1), zeros(4, dim_Obs, n_tau+1)
R_lin, δObs_lin = zeros(4, dim_Obs, n_tau+1), zeros(4, dim_Obs, n_tau+1)
R_gen, δObs_gen = zeros(4, dim_Obs, n_tau+1), zeros(4, dim_Obs, n_tau+1)

for i in 1:2
    Obs(x) = x[1:dim, :] .^i
    Obs_num(x) = (x[1:dim, :] .- M) .^i
    R_num[i,:,:] = generate_numerical_response(F, u, dim_full, dt, n_tau, 10000, sigma, Obs_num, dim_Obs; n_ens=1000, resolution=res_trj)
    R_lin[i,:,:], δObs_lin[i,:,:] = generate_score_response(trj, u_score, div_u, f, score_qG, res_trj*dt, n_tau, Obs, dim_Obs)
    R_gen[i,:,:], δObs_gen[i,:,:] = generate_score_response(trj, u_score, div_u, f, score_gen, res_trj*dt, n_tau, Obs, dim_Obs)
end


##

R0_gen = zeros(dim,dim)
for j in 1:steps_trj
    R0_gen .+= trj[:,j] * score_gen(trj[:,j])'
end
R0_gen ./= (steps_trj)
invR0_gen = .-inv(R0_gen)
    
R_gen_hack = zeros(4, dim, n_tau+1)
for i in 1:4
    for j in 1:n_tau+1
        R_gen_hack[i,:,j] = 0.5 .* (R_gen[i,:,j]'*invR0_gen .+ (invR0_gen*R_gen[i,:,j])')
    end
end

R_num_cum = zeros(4, n_tau+1)
R_lin_cum = zeros(4, n_tau+1)
R_gen_cum = zeros(4, n_tau+1)

for i in 1:4
    R_num_cum[i,:] = cumsum(R_num[i,1,:]) .* (res_trj*dt)
    R_lin_cum[i,:] = cumsum(R_lin[i,1,:]) .* (res_trj*dt)
    R_gen_cum[i,:] = cumsum(R_gen_hack[i,1,:]) .* (res_trj*dt)
end

##
index = 1
plot(R_num_cum[index,:] ./ ϵ, label="Numerical", linewidth=2, legend=:best,
     xlabel="Time", ylabel="Response", title="Response function for moment 1")
plot!(R_lin_cum[index,:] .* S[1]^(index-1)./ ϵ, label="Linear", linewidth=2, linestyle=:dash)
plot!(R_gen_cum[index,:] .* S[1]^(index-1)./ ϵ, label="Generated", linewidth=2, linestyle=:dot)

##
index = 1
plot(R_num[index,1,:] ./ ϵ, label="Numerical", linewidth=2, legend=:best,
     xlabel="Time", ylabel="Response", title="Response function for moment 1")
plot!(R_lin[index,1,:] .* S[1]^(index-1)./ ϵ, label="Linear", linewidth=2, linestyle=:dash)
plot!(R_gen_hack[index,1,:] .* S[1]^(index-1)./ ϵ, label="Generated", linewidth=2, linestyle=:dot)

##
using Plots: gr
gr()  # Use GR backend for better performance with multiple plots

# Create plots for each moment (i=1,2,3,4)
plots = []
for i in 1:4
    plt = plot(layout=(2,2), size=(1000,800), 
               title="Response function for moment $(i)")
    
    for dim_idx in 1:4
        # Timepoints for x-axis
        t = (0:n_tau) .* (res_trj*dt)
        
        # Plot the three responses for this dimension
        plot!(plt[dim_idx], t, R_num[i,dim_idx,:], 
              label="Numerical", linewidth=2, legend=:best,
              xlabel="Time", ylabel="Response", title="Dimension $dim_idx")
        
        plot!(plt[dim_idx], t, R_lin[i,dim_idx,:], 
              label="Linear", linewidth=2, linestyle=:dash)
        
        plot!(plt[dim_idx], t, R_gen[i,dim_idx,:], 
              label="Generated", linewidth=2, linestyle=:dot)
    end
    
    push!(plots, plt)
end

# Display the plots
display(plots[1])
display(plots[2])
display(plots[3])
display(plots[4])

# Save the plots if needed
# savefig(plots[1], "response_moment1.png")
# savefig(plots[2], "response_moment2.png")
# savefig(plots[3], "response_moment3.png")
# savefig(plots[4], "response_moment4.png")