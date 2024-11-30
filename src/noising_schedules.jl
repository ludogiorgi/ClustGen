# Variance exploding protocol
 
σ_variance_exploding(t; σ_min = 0.01, σ_max = 1.0) = @. σ_min * (σ_max/σ_min)^t

g_variance_exploding(t; σ_min=0.01, σ_max=1.0) = σ_min * (σ_max/σ_min)^t * sqrt(2*log(σ_max/σ_min))