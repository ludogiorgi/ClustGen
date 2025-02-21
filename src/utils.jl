function rk4_step!(u, dt, f)
    k1 = f(u)
    k2 = f(u .+ 0.5 .* dt .* k1)
    k3 = f(u .+ 0.5 .* dt .* k2)
    k4 = f(u .+ dt .* k3)
    @inbounds u .= u .+ (dt / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
end


function covariance(x1, x2; timesteps=length(x), progress = false)
    μ1 = mean(x1)
    μ2 = mean(x2)
    autocor = zeros(timesteps)
    progress ? iter = ProgressBar(1:timesteps) : iter = 1:timesteps
    for i in iter
        autocor[i] = mean(x1[i:end] .* x2[1:end-i+1]) - μ1 * μ2
    end
    return autocor
end


function covariance(g⃗1, g⃗2, Q::Eigen, timelist; progress=false)
   #  @assert all(real.(Q.values[1:end-1]) .< 0) "Did not pass an ergodic generator matrix"
    autocov = zeros(length(timelist))
    # Q  = V Λ V⁻¹
    Λ, V = Q
    p = real.(V[:, end] ./ sum(V[:, end]))
    v1 = V \ (p .* g⃗1)
    w2 = g⃗2' * V
    μ1 = sum(p .* g⃗1)
    μ2 = sum(p .* g⃗2)
    progress ? iter = ProgressBar(eachindex(timelist)) : iter = eachindex(timelist)
    for i in iter
        autocov[i] = real(w2 * (exp.(Λ .* timelist[i]) .* v1) - μ1 * μ2)
    end
    return autocov
end

covariance(g⃗1, g⃗2, Q, timelist; progress = false) = covariance(g⃗1, g⃗2, eigen(Q), timelist; progress = progress)

function cleaning(averages, centers, labels)
    unique_clusters = sort(unique(labels))
    mapping = Dict(old_cluster => new_cluster for (new_cluster, old_cluster) in enumerate(unique_clusters))
    labels_new = [mapping[cluster] for cluster in labels]
    averages_new = averages[:, unique_clusters]
    centers_new = centers[:, unique_clusters]
    return averages_new, centers_new, length(unique_clusters), labels_new
end