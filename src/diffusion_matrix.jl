"""
    computeSigma(X, piVec, Q, s, Ntau, dtau, dt; iterations=0)

Compute the diffusion matrix Σ using statistical estimates from data.

# Arguments
- `X`: Data matrix with shape (D, N) where D is dimension and N is sample count
- `piVec`: Vector of stationary distribution weights, length N
- `Q`: Generator matrix with shape (N, N)
- `s`: Score function that calculates gradient of log probability density
- `Ntau`: Number of time steps for trajectory comparison (used when iterations > 0)
- `dtau`: Time step resolution for trajectory sampling
- `dt`: Base time step size
- `iterations`: Number of optimization iterations (0 for no optimization)

# Returns
- The estimated diffusion matrix Σ
"""
# function computeSigma(X, piVec, Q, s, Ntau, dtau, dt; iterations=0)
#     # Extract dimensions from data
#     (D, N) = size(X)
    
#     # Input validation
#     @assert length(piVec) == N "piVec must have length N"
#     @assert size(Q) == (N, N) "Q must be N×N"

#     # Compute gradient of log probability (score) at each data point
#     gradLogp = zeros(D, N)
#     for i in 1:N
#         gradLogp[:,i] = s(X[:,i])
#     end

#     # Weight data points by stationary distribution
#     Xp = X * Diagonal(piVec)

#     # Compute correlation matrices
#     C_Q = X * Q * Xp'
#     C_grad = gradLogp * Xp'
    
#     # Make the matrix symmetric and ensure positive definiteness
#     M = C_Q * inv(C_grad)  # Symmetrize
        
#     # Add small regularization for numerical stability
#     ϵ = 1e-6
#     M += ϵ * I(size(M,1))

#     # Initialize Σ outside try block for scope reasons
#     local Σ
        
#     # Try Cholesky decomposition with error handling
#     try
#         if size(M, 1) == 1
#             # Special case for 1x1 matrix
#             Σ = sqrt(M)
#         else
#             Σ = cholesky(M).L
#         end
#     catch e
#         # Fall back to eigendecomposition if Cholesky fails
#         λ, V = eigen(M)
#         λ .= max.(λ, 0)  # Ensure positive eigenvalues
#         Σ = V * Diagonal(sqrt.(λ)) * V'
#     end
    
#     # Convert to standard matrix type and prepare for optimization
#     Σ = Matrix(Σ)
#     sigma_vec = vec(Σ)
#     dim_sigma = size(Σ)

#     # Return early if no optimization requested
#     if iterations == 0
#         return Σ
#     else
#         # Compute reference trajectories using the generator Q
#         trj_Q = zeros(Ntau, D, N)
#         for i in 0:Ntau-1
#             trj_Q[i+1,:,:] = X * exp(Q*i*dtau*dt)
#         end

#         # Define optimization objective function
#         function objective(sigma_vec)
#             # Reshape vector back to matrix for computations
#             Sigma = reshape(sigma_vec, dim_sigma)
            
#             # Define drift function using score and diffusion
#             Σ2s(x, t) = (Sigma * Sigma') * s(x)
            
#             # Generate trajectories using the current diffusion matrix
#             trj_gen = zeros(Ntau, D, N)
#             Threads.@threads for i in 1:N
#                 trj_gen[:,:,i] = evolve(X[:,i], dt, Ntau*dtau, Σ2s, Sigma; resolution=dtau)'
#             end
            
#             # Compute loss by comparing reference and generated trajectories
#             loss = 0.0
#             for tau in 1:Ntau
#                 C_Q = trj_Q[tau,:,:] * Xp'
#                 C_grad = trj_gen[tau,:,:] * Xp'
#                 loss += norm(C_Q - C_grad, 2)
#             end
            
#             # Report progress
#             println("Loss: ", loss, " Sigma: ", Sigma)
#             return loss
#         end

#         # Setup and run optimization
#         opts = Optim.Options(iterations=iterations)
#         result = optimize(objective, sigma_vec, LBFGS(), opts)
        
#         # Reshape result back to matrix and return
#         Σ_fin = reshape(Optim.minimizer(result), dim_sigma)

#         return Σ_fin
#     end
# end

function computeSigma(X, piVec, Q, gradLogp)
    # Extract dimensions from data
    (D, N) = size(X)
    
    # Input validation
    @assert length(piVec) == N "piVec must have length N"
    @assert size(Q) == (N, N) "Q must be N×N"
    @assert size(gradLogp) == (D, N) "gradLogp must have shape (D, N)"

    # Weight data points by stationary distribution
    Xp = X * Diagonal(piVec)

    # Compute correlation matrices
    C_Q = X * Q * Xp'
    C_grad = gradLogp * Xp'
    
    # Make the matrix symmetric and ensure positive definiteness
    M = C_Q * inv(C_grad)  # Symmetrize

    return cholesky(0.5*(M .+ M')).L
end