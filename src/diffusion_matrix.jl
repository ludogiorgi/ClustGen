
function computeSigma(X, piVec, Q, s, Ntau, dtau, dt; iterations=0)

(D, N) = size(X)
@assert length(piVec) == N "piVec must have length N"
@assert size(Q) == (N, N) "Q must be N×N"

gradLogp = zeros(D, N)
for i in 1:N
    gradLogp[:,i] = s(X[:,i])
end

Xp = X * Diagonal(piVec)

C_Q = X * Q * Xp'
C_grad = gradLogp * Xp'
# Make the matrix symmetric and ensure positive definiteness
M = (C_Q * inv(C_grad) + (C_Q * inv(C_grad))') / 2  # Symmetrize
    
# Add small regularization if needed
ϵ = 1e-6
M += ϵ * I(size(M,1))

# Initialize Σ outside try block
local Σ
    
# Try Cholesky decomposition with error handling
try
    if size(M, 1) == 1
        # Special case for 1x1 matrix
        Σ = sqrt(M)
    else
        Σ = cholesky(M).L
    end
catch e
    # Always use eigendecomposition if Cholesky fails
    λ, V = eigen(M)
    λ .= max.(λ, 0)  # Ensure positive eigenvalues
    Σ = V * Diagonal(sqrt.(λ)) * V'
end
    Σ = Matrix(Σ)
    sigma_vec = vec(Σ)
    dim_sigma = size(Σ)

    if iterations == 0
        return Σ
    else
        trj_Q = zeros(Ntau, D, N)
        for i in 0:Ntau-1
            trj_Q[i+1,:,:] = X * exp(Q*i*dtau*dt)
        end

        function objective(sigma_vec)
            # Reshape vector back to matrix for computations
            Sigma = reshape(sigma_vec, dim_sigma)
            Σ2s(x, t) = (Sigma * Sigma') * s(x)
            trj_gen = zeros(Ntau, D, N)
            Threads.@threads for i in 1:N
                trj_gen[:,:,i] = evolve(X[:,i], dt, Ntau*dtau, Σ2s, Sigma; resolution=dtau)'
            end
            loss = 0.0
            for tau in 1:Ntau
                C_Q = trj_Q[tau,:,:] * Xp'
                C_grad = trj_gen[tau,:,:] * Xp'
                loss += norm(C_Q - C_grad, 2)
            end
            println("Loss: ", loss, " Sigma: ", Sigma)
            return loss
        end

        opts = Optim.Options(iterations=iterations)
        result = optimize(objective, sigma_vec, LBFGS(), opts)
        
        # Reshape result back to matrix
        Σ_fin = reshape(Optim.minimizer(result), dim_sigma)

        return Σ_fin
end
end