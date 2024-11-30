function decorrelation_times(time_series::Array{Float64, 2}, threshold::Float64)
    D, N = size(time_series)
    autocorr_times = zeros(Int, D)
    
    for d in 1:D
        for t in 2:N
            autocorr = cor(time_series[d, 1:end-t+1], time_series[d, t:end])
            if autocorr < threshold
                autocorr_times[d] = t
                break
            end
        end
    end
    
    return autocorr_times
end