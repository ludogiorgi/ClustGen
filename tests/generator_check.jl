using Pkg
Pkg.activate(".")
Pkg.instantiate()
##
using Revise
using ClustGen
using StateSpacePartitions
using LinearAlgebra
using Random 
using ProgressBars
using Statistics
using KernelDensity
using HDF5
using Flux
using QuadGK
using BSON
using Plots
using StatsBase
using MarkovChainHammer
import MarkovChainHammer.Trajectory: ContinuousTimeEmpiricalProcess
import LaTeXStrings


function ∇U_c(x; A1=1.0, A2=1.0, B1=0.6, B2=0.3, C=1.0, D=0.0)
    # Conservative gradient terms
    ∇U1 = 2 * (x[1] + A1) * (x[1] - A1)^2 + 2 * (x[1] - A1) * (x[1] + A1)^2 + B1 + C * (x[1] * x[2])^2
    ∇U2 = 2 * (x[2] + A2) * (x[2] - A2)^2 + 2 * (x[2] - A2) * (x[2] + A2)^2 + B2 + C * (x[1] * x[2])^2
    
    # Non-conservative term (e.g., rotational flow)
    F1 = -D * x[2]
    F2 = D * x[1]
    
    # Total force
    return [∇U1 + F1, ∇U2 + F2]
end

function ∇U_nc(x; A1=1.0, A2=1.0, B1=0.6, B2=0.3, C=1.0, D=3.0)
    # Conservative gradient terms
    ∇U1 = 2 * (x[1] + A1) * (x[1] - A1)^2 + 2 * (x[1] - A1) * (x[1] + A1)^2 + B1 + C * (x[1] * x[2])^2
    ∇U2 = 2 * (x[2] + A2) * (x[2] - A2)^2 + 2 * (x[2] - A2) * (x[2] + A2)^2 + B2 + C * (x[1] * x[2])^2
    
    # Non-conservative term (e.g., rotational flow)
    F1 = -D * x[2]
    F2 = D * x[1]
    
    # Total force
    return [∇U1 + F1, ∇U2 + F2]
end

function potential_data(x0, timesteps, dt, Σ, ∇U; res = 1)
    dim = length(x0)
    force(x) = -∇U(x)
    x = []
    x_temp = x0
    for i in ProgressBar(2:timesteps)
        Σ2_force(x) = Σ^2 * force(x)
        rk4_step!(x_temp, dt, Σ2_force)
        @inbounds x_temp .+= Σ * randn(dim) * sqrt(2dt)
        if i % res == 0
            push!(x, copy(x_temp))
        end
    end
    x = hcat(x...)
    return x
end

dt = 0.025
Σ_true = [1.0 0.5; 0.5 1.0]
obs_c = potential_data([0.0,0.0], 10000000, dt, Σ_true, ∇U_c)
obs_nc = potential_data([0.0,0.0], 10000000, dt, Σ_true, ∇U_nc)
dim = size(obs_c)[1]

normalization = false
σ_value = 0.05

μ_c = repeat(obs_c[:,1:100:end], 1, 1)
μ_nc = repeat(obs_nc[:,1:100:end], 1, 1)

averages_c, centers_c, Nc_c, labels_c = f_tilde_labels(σ_value, μ_c; prob=0.001, do_print=true, conv_param=0.001, normalization=normalization)
averages_nc, centers_nc, Nc_nc, labels_nc = f_tilde_labels(σ_value, μ_nc; prob=0.001, do_print=true, conv_param=0.001, normalization=normalization)

Qc = generator(labels_c)
Qnc = generator(labels_nc)

Qc_c, Qc_nc = decomposition(Qc)
Qnc_c, Qnc_nc = decomposition(Qnc)

lc,vc = eigen(Qc)
lnc,vnc = eigen(Qnc)

Plots.plot(imag.(lc))
Plots.plot!(imag.(lnc))

Plots.plot(real.(lc))
Plots.plot!(real.(lnc))

tsteps = 81
res = 10

##

auto_obs_c = zeros(dim, tsteps)
auto_obs_nc = zeros(dim, tsteps)
auto_Qc = zeros(dim, tsteps)
auto_Qnc_c = zeros(dim, tsteps)

for i in 1:dim
    auto_obs_c[i,:] = autocovariance(obs_c[i,1:res:end]; timesteps=tsteps) 
    auto_obs_nc[i,:] = autocovariance(obs_nc[i,1:res:end]; timesteps=tsteps)
    auto_Qc[i,:] = autocovariance(centers_c[i,:], Qc, [0:dt*res:Int(res * (tsteps-1) * dt)...])
    auto_Qnc_c[i,:] = autocovariance(centers_nc[i,:], Qnc_c, [0:dt*res:Int(res * (tsteps-1) * dt)...])
end

Plots.plot(auto_obs_c[2,:])
Plots.plot!(auto_obs_nc[2,:])
Plots.plot!(auto_Qc[2,:])
Plots.plot!(auto_Qnc_c[2,:])