function sample(Dim, nn, n_samples, n_diffs, σ, g)
    dt = 1.0 / n_diffs
    ens = zeros(Dim, n_samples)
    for i in 1:n_samples
        xOld = randn(Dim)
        for t in 1:n_diffs
            t_diff = (n_diffs - t + 1) / n_diffs
            s = σ(t_diff)
            score = nn(xOld, t_diff) ./ s
            xNew = xOld .+ score .* g(t_diff) .^2 .* dt .+ randn(Dim) .* sqrt(dt) .* g(t_diff)
            xOld = xNew
        end
        ens[:,i] = xOld
    end
    return ens
end