using Pkg
Pkg.activate(".")
Pkg.instantiate()

##
using NetCDF
using Statistics  # For mean function in PCA
using LinearAlgebra  # For eigen decomposition
using HDF5
using Glob

# === Memory monitoring function ===
function print_memory_usage(label="")
    total_mem = Sys.total_memory() / 1024^3  # GB
    free_mem = Sys.free_memory() / 1024^3    # GB
    used_mem = total_mem - free_mem
    
    println("$label - Memory: Used $(round(used_mem, digits=2)) GB / Total $(round(total_mem, digits=2)) GB ($(round(used_mem/total_mem*100, digits=1))% used)")
end

print_memory_usage("Initial")


# === Read and stack all PlaSim_data NetCDF files (Memory Optimized, robust to variable time dimension) ===

# Get file list
data_dir = joinpath(@__DIR__, "PlaSim_data")
println("Looking for NetCDF files in: $data_dir")
println("Directory exists: $(isdir(data_dir))")
if isdir(data_dir)
    all_files = readdir(data_dir)
    println("All files in directory: $all_files")
end

files = sort(Glob.glob("TS_HistoryC_m*.nc", data_dir))
println("Found $(length(files)) files matching pattern TS_HistoryC_m*.nc")
if length(files) == 0
    error("No NetCDF files found in directory: $data_dir\nLooking for pattern: TS_HistoryC_m*.nc")
end

# First pass: get spatial dimensions and total time
nlon, nlat = 0, 0
file_time_lengths = Int[]
for file in files
    ds = NetCDF.open(file)
    ts = NetCDF.readvar(ds, "ts")
    nlon, nlat, ntime = size(ts)
    push!(file_time_lengths, ntime)
    NetCDF.close(ds)
end

total_time = sum(file_time_lengths)
println("Total time steps across all files: $total_time")

# Pre-allocate arrays
data = Array{Float32}(undef, nlon, nlat, total_time)
times = Vector{Float64}(undef, total_time)

println("Allocated arrays: data = $(sizeof(data) / 1024^3) GB, times = $(sizeof(times) / 1024^3) GB")

# Second pass: fill arrays
curr_idx = 1
for (i, file) in enumerate(files)
    ds = NetCDF.open(file)
    ts = Float32.(NetCDF.readvar(ds, "ts"))
    t = NetCDF.readvar(ds, "time")
    ntime = size(ts, 3)
    data[:, :, curr_idx:curr_idx+ntime-1] .= ts
    times[curr_idx:curr_idx+ntime-1] .= t
    curr_idx += ntime
    NetCDF.close(ds)
    if i % 5 == 0
        GC.gc()
    end
    println("Processed file $i/$(length(files)) (time steps: $ntime)")
end

println("Final data shape: ", size(data))
println("Final times shape: ", size(times))
print_memory_usage("After loading NetCDF data")

# Verify data was loaded successfully
if data === nothing || size(data, 3) == 0
    error("Data loading failed - data array is empty or undefined")
end

##
# Remove seasonal cycle (climatological mean for each day of year)
println("\n=== Removing seasonal cycle (day-of-year climatology) ===")
year_length = 360
n_years = div(total_time, year_length)
remaining_points = total_time - n_years * year_length

# Pre-allocate array for climatological mean (mean for each day of year)
year_mean = Array{Float32}(undef, nlon, nlat, year_length)

# Compute climatological mean for each day of the year
println("Computing climatological means for each day of year...")
for day_of_year in 1:year_length
    # Collect all data points for this specific day across all years
    day_indices = day_of_year:year_length:total_time
    if length(day_indices) > 0
        # Extract data for this day across all years - handle the case safely
        day_data = data[:, :, day_indices]
        if ndims(day_data) == 3
            year_mean[:, :, day_of_year] = mean(day_data, dims=3)
        else
            # If only one time point, day_data is 2D
            year_mean[:, :, day_of_year] = day_data
        end
    else
        # Fill with zeros if no data for this day (shouldn't happen but safety check)
        year_mean[:, :, day_of_year] .= 0.0f0
        println("Warning: No data found for day $day_of_year")
    end
end

# Subtract climatological mean from each data point
println("Removing seasonal cycle from data...")
for t in 1:total_time
    day_of_year = mod1(t, year_length)  # Get day of year (1-360)
    data[:, :, t] .-= year_mean[:, :, day_of_year]
end

# Handle remaining incomplete year if any (use available data for climatology)
if remaining_points > 0
    println("Processed $n_years complete years and $remaining_points remaining points")
else
    println("Processed $n_years complete years")
end

# Do not free year_mean here; it is needed for saving later

println("Removed seasonal cycle (day-of-year climatological means)")
print_memory_usage("After removing seasonal cycle")
##
# PCA Analysis Section (Memory Optimized)
println("\n=== Performing Memory-Optimized PCA Analysis ===")
if ndims(data) == 3
    sz = size(data)
    n_components = 32  # Number of principal components to retain
    
    # Determine time dimension and reshape data
    if sz[3] > sz[1]  # Time is last dimension (lon, lat, time)
        nlon, nlat, ntime = sz
        time_dim = 3
        # Reshape to (space, time) matrix directly without copying
        data_matrix = reshape(data, nlon*nlat, ntime)
        println("Data shape: (lon=$nlon, lat=$nlat, time=$ntime)")
    else  # Time is first dimension (time, lat, lon)
        ntime, nlat, nlon = sz
        time_dim = 1
        # Reshape to (space, time) matrix - use view to avoid copying
        data_matrix = reshape(permutedims(data, [2, 3, 1]), nlat*nlon, ntime)
        println("Data shape: (time=$ntime, lat=$nlat, lon=$nlon)")
    end
    
    # Remove spatial mean (already removed yearly averages)
    println("Computing spatial means...")
    data_mean = mean(data_matrix, dims=2)
    data_matrix .-= data_mean  # In-place subtraction to save memory
    
    # Free original data array to save memory
    data = nothing
    GC.gc()  # Force garbage collection
    print_memory_usage("After freeing original data array")
    
    # Compute covariance matrix using more memory-efficient approach
    println("Computing spatial covariance matrix (memory efficient)...")
    # Use Float32 for covariance to save memory
    C = zeros(Float32, size(data_matrix, 1), size(data_matrix, 1))
    batch_size = min(2000, ntime)  # Increased batch size for better efficiency
    
    # Process batches with progress indicator
    for i in 1:batch_size:ntime
        end_idx = min(i + batch_size - 1, ntime)
        batch = @view data_matrix[:, i:end_idx]
        
        # Use BLAS for efficient matrix multiplication
        BLAS.gemm!('N', 'T', Float32(1.0), batch, batch, Float32(1.0), C)
        
        # Periodic garbage collection to prevent memory buildup
        if i % (batch_size * 10) == 1
            GC.gc()
        end
    end
    C ./= Float32(ntime - 1)
    
    # Eigenvalue decomposition with optimized settings
    println("Performing eigenvalue decomposition...")
    # Use Symmetric wrapper to ensure numerical stability
    C_sym = Symmetric(C)
    eigenvals, eigenvecs = eigen(C_sym)
    
    # Free covariance matrix immediately after eigendecomposition
    C = nothing
    C_sym = nothing
    GC.gc()
    
    # Sort by decreasing eigenvalue (most important first)
    idx = sortperm(eigenvals, rev=true)
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    # Save all eigenvalues for correct explained variance calculation
    all_eigenvals = eigenvals

    # Keep only needed components to save memory
    eigenvals = eigenvals[1:n_components]
    eigenvecs = eigenvecs[:, 1:n_components]

    # Compute explained variance using all eigenvalues
    total_variance = sum(all_eigenvals)
    explained_variance = eigenvals ./ total_variance
    cumulative_variance = cumsum(explained_variance)

    println("Explained variance by first $n_components components (relative to total variance):")
    for i in 1:n_components
        println("  PC$i: $(round(explained_variance[i]*100, digits=2))% (cumulative: $(round(cumulative_variance[i]*100, digits=2))%)")
    end
    
    # Project data onto first n_components (batch processing to save memory)
    println("Computing PC scores (batch processing)...")
    pc_scores = zeros(Float32, n_components, ntime)
    batch_size = min(10000, ntime)  # Larger batch size for better efficiency
    
    # Pre-compute the projection matrix for efficiency
    projection_matrix = eigenvecs[:, 1:n_components]'
    
    for i in 1:batch_size:ntime
        end_idx = min(i + batch_size - 1, ntime)
        batch_range = i:end_idx
        
        # Use BLAS for efficient matrix multiplication
        batch_data = @view data_matrix[:, batch_range]
        batch_scores = @view pc_scores[:, batch_range]
        
        # Compute projection: PC_scores = eigenvecs' * data
        mul!(batch_scores, projection_matrix, batch_data)
        
        # Periodic garbage collection
        if i % (batch_size * 5) == 1
            GC.gc()
        end
    end
    
    # Free data_matrix and projection matrix to save memory
    data_matrix = nothing
    projection_matrix = nothing
    GC.gc()

    # === Save all variables needed for reconstruction to file in PlaSim_data ===
    output_file = joinpath(data_dir, "pc_scores.h5")
    println("Saving pc_scores, year_mean, data_mean, eigenvecs, times to $output_file ...")
    h5open(output_file, "w") do f
        f["pc_scores"] = pc_scores
        f["year_mean"] = year_mean
        f["data_mean"] = data_mean
        f["eigenvecs"] = eigenvecs
        f["times"] = times
    end
    println("All reconstruction variables saved to $output_file")

    # Now free memory
    year_mean = nothing
    data_mean = nothing
    eigenvecs = nothing
    pc_scores = nothing
    GC.gc()

else
    println("PCA analysis requires 3D data (with time dimension)")
end


