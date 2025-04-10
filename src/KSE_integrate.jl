using FFTW
using Statistics

"""
    dealias(u, dealias_factor=3)

Apply dealiasing to Fourier coefficients by zeroing out high-frequency modes.
The dealias_factor controls how many modes are kept (N/dealias_factor).
Default is 3 (the "2/3 rule").
"""
function dealias(u; dealias_factor=3)
    N = length(u)
    Nd = round(Int, N/dealias_factor)
    
    v = copy(u)  # Use copy to avoid modifying the input
    v[Nd+2:N-Nd] .= 0

    return v
end

function domain(L, N)
    x = L*(0:N-1)'/N .- L/2
    k = (2*pi/L)*[0:N/2; -N/2+1:-1]'

    return x, k
end

"""
    field2vector(f, N, s, dealias_factor=3)

Convert full Fourier field to a reduced vector representation.
"""
function field2vector(f, N, s; dealias_factor=3)
    Nd = round(Int, N/dealias_factor)
    v = imag(f[2:Nd+1])

    if !s
        v = [v; real(f[2:Nd+1]); real(f[1])]
    end

    return v
end

"""
    vector2field(v, N, s, dealias_factor=3)

Convert reduced vector representation back to full Fourier field.
"""
function vector2field(v, N, s; dealias_factor=3)
    Nd = round(Int, N/dealias_factor)

    f = zeros(Complex{Float64}, N, 1)
    f[2:Nd+1] = v[1:Nd]*im
    if !s
        f[2:Nd+1] = f[2:Nd+1] + v[Nd+1:2*Nd]
        f[1] = v[2*Nd+1]
    end

    f[N÷2+2:end] = conj(f[N÷2:-1:2])

    return f
end

"""
    KSE_integrate(L, dt_ref, t_max, t_store, u0, s, zero_mean=true, dealias_factor=3)

Integrate the Kuramoto-Sivashinsky equation.

Parameters:
- L: Domain length
- dt_ref: Reference time step
- t_max: Maximum integration time
- t_store: How often to store snapshots (0 for final state only)
- u0: Initial condition in Fourier space
- s: Symmetry flag (true for odd symmetry)
- zero_mean: Whether to enforce zero mean
- dealias_factor: Dealiasing factor (N/dealias_factor modes are kept)
"""
function KSE_integrate(L, dt_ref, t_max, t_store, u0, s, zero_mean=true; dealias_factor=3)
    ## adjust time step size
    N_steps = floor(Int, t_max/dt_ref) + 1
    dt = t_max / N_steps

    ## grid and initial condition
    N = length(u0)
    _, k = domain(L, N)

    v = dealias(u0; dealias_factor=dealias_factor)

    if s
        v = im*imag(v)
    end

    ## precompute ETDRK4 scalars
    Linear = k.^2 - k.^4
    
    E = exp.(dt*Linear)
    E2 = exp.(dt*Linear/2)
    
    M = 32
    r = exp.(im*pi*((1:M)' .- 0.5)/M)
    LR = dt*repeat(transpose(Linear),1,M) + repeat(r,N,1)
    Q = dt*real(mean((exp.(LR/2) .- 1)./LR, dims = 2))'
    f1 = dt*real(mean((-4 .- LR + exp.(LR).*(4 .- 3*LR+LR.^2))./LR.^3, dims = 2))'
    f2 = dt*real(mean((2 .+ LR + exp.(LR).*(-2 .+ LR))./LR.^3, dims = 2))'
    f3 = dt*real(mean((-4 .- 3*LR - LR.^2 + exp.(LR).*(4 .- LR))./LR.^3, dims = 2))'
    
    ## time-stepping loop:
    g = -0.5im*k

    if t_store == 0
        n = 1
    else
        n = floor(Int, t_store/dt)
        snapshot = field2vector(v, N, s; dealias_factor=dealias_factor)
        snapshots = zeros(length(snapshot), N_steps÷n)
        snapshots[:,1] = snapshot
        t_grid = zeros(N_steps÷n)
    end

    F! = plan_fft!(v, flags=FFTW.MEASURE)
    IF! = plan_ifft!(v, flags=FFTW.MEASURE)
    IF = plan_ifft(v, flags=FFTW.MEASURE)

    t = 0;

    for q = 1:N_steps
        t = t+dt
        
        Nv = g.*dealias(F!*((IF*v).^2); dealias_factor=dealias_factor)
        a = E2.*v + Q.*Nv
        Na = g.*dealias(F!*((IF*a).^2); dealias_factor=dealias_factor)
        
        b = E2.*v + Q.*Na
        Nb = g.*dealias(F!*((IF!*b).^2); dealias_factor=dealias_factor)
        
        c = E2.*a + Q.*(2*Nb-Nv)
        Nc = g.*dealias(F!*((IF!*c).^2); dealias_factor=dealias_factor)
        
        v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3

        # maintain conjugate symmetry
        v[N÷2+2:end] = conj(v[N÷2:-1:2])

        if s
            v = im*imag(v)
        end

        if zero_mean
            v[1] = 0
        else
            v[1] = real(v[1])
        end
        
        if rem(q,n) == 0 && t_store != 0
            snapshots[:,q÷n] = field2vector(v, N, s; dealias_factor=dealias_factor)
            t_grid[q÷n] = t
        end
    end

    if t_store == 0
        snapshots = field2vector(v, N, s; dealias_factor=dealias_factor)
        t_grid = t
    end

    return snapshots, t_grid
end

function create_ks_animation(t::Vector{Float64}, u::Matrix{Float64}, x::Vector{Float64}, 
    filename::String; framerate::Int64=30)
# Create directory structure if it doesn't exist
mkpath(dirname(filename))

# Clean up any non-finite values
u_clean = copy(u)
for i in eachindex(u_clean)
if !isfinite(u_clean[i])
u_clean[i] = 0
end
end

# Calculate y-axis limits for consistent plotting
y_min = minimum(u_clean)
y_max = maximum(u_clean)
y_padding = 0.1 * (y_max - y_min)
y_limits = (y_min - y_padding, y_max + y_padding)

# Set up figure and axis with proper limits
fig = Figure(resolution=(900, 600))
ax = Axis(fig[1, 1], 
title="Kuramoto-Sivashinsky Equation", 
xlabel="x", 
ylabel="u(x,t)",
limits=(extrema(x), y_limits))

# Create animation
nframes = size(u_clean, 2)
println("Creating animation with $nframes frames...")

GLMakie.record(fig, filename, 1:nframes; framerate=framerate) do i
empty!(ax)
GLMakie.lines!(ax, x, u_clean[:, i], linewidth=2, color=:blue)
ax.title = "KS Equation (t = $(round(t[i], digits=2)))"

# Show progress every 10%
if i % max(1, div(nframes, 10)) == 0
println("Processing frame $i of $nframes ($(round(100*i/nframes; digits=1))%)")
end
end

println("Animation saved to $filename")
return filename
end

function reduce_fourier_energy(u_fourier, energy_threshold::Float64=0.99)
    # u_fourier should be a matrix where each column is a time snapshot
    # and each row is a Fourier coefficient
    
    # Calculate average energy in each mode
    mode_energy = mean(abs2.(u_fourier), dims=2)
    
    # Sort modes by energy
    sorted_indices = sortperm(vec(mode_energy), rev=true)
    sorted_energy = mode_energy[sorted_indices]
    
    # Calculate cumulative energy
    total_energy = sum(sorted_energy)
    cumulative_energy = cumsum(sorted_energy) ./ total_energy
    
    # Find how many modes are needed to reach the threshold
    n_modes = findfirst(cumulative_energy .>= energy_threshold)
    
    # Get the indices of the most energetic modes
    important_indices = sorted_indices[1:n_modes]
    
    # Return the reduced coefficients and the indices
    return u_fourier[important_indices, :], important_indices
end

"""
    reduce_fourier_energy(u_fourier, n_modes::Int)

Reduce the dimensionality of Fourier coefficients by keeping the specified
number of most energetic modes.

# Arguments
- `u_fourier`: Matrix where each column is a time snapshot and each row is a Fourier coefficient
- `n_modes::Int`: Number of modes to keep

# Returns
- Reduced Fourier coefficients
- Indices of the kept modes
- Relative energy for each kept mode (not cumulative)
"""
function reduce_fourier_energy(u_fourier, n_modes::Int)
    # Calculate average energy in each mode
    mode_energy = mean(abs2.(u_fourier), dims=2)
    
    # Sort modes by energy
    sorted_indices = sortperm(vec(mode_energy), rev=true)
    sorted_energy = mode_energy[sorted_indices]
    
    # Calculate total energy
    total_energy = sum(sorted_energy)
    
    # Ensure n_modes doesn't exceed the number of available modes
    n_modes = min(n_modes, length(sorted_indices))
    
    # Get the indices of the most energetic modes
    important_indices = sorted_indices[1:n_modes]
    
    # Calculate relative energy for each kept mode (not cumulative)
    relative_energies = sorted_energy[1:n_modes] ./ total_energy
    
    # Return the reduced coefficients, indices, and relative energies
    return u_fourier[important_indices, :], important_indices, relative_energies
end

"""
    reconstruct_physical_from_reduced(u_reduced, kept_modes, N, symm)

Convert reduced Fourier coefficients back to physical space representation.

# Arguments
- `u_reduced`: Reduced Fourier coefficients (subset of modes)
- `kept_modes`: Indices of the modes that were kept
- `N`: Size of the full domain
- `symm`: Symmetry flag (true for odd symmetry)

# Returns
- Matrix where each row is a time snapshot in physical space
"""
function reconstruct_physical_from_reduced(u_reduced, kept_modes, N, symm)
    # Get dimensions
    n_modes = length(kept_modes)
    n_times = size(u_reduced, 2)
    
    # Initialize output array for physical space
    uu_physical = zeros(n_times, N)
    
    # Process each time snapshot
    for i in 1:n_times
        # Start with zeros for the full vector representation
        if symm
            full_vector = zeros(N÷3)
        else
            full_vector = zeros(2*N÷3 + 1)
        end
        
        # Place the reduced coefficients back in their original positions
        for j in 1:n_modes
            full_vector[kept_modes[j]] = u_reduced[j, i]
        end
        
        # Convert to full Fourier field
        u_spectral = vector2field(full_vector, N, symm)
        
        # Transform to physical space
        u_physical = real(ifft(u_spectral))
        
        # Store the result
        uu_physical[i, :] = u_physical
    end
    
    return uu_physical
end