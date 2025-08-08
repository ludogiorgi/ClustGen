"""
    save_model(nn, filename)

Save a Flux model and its parameters to disk.

# Arguments
- `nn`: Neural network model
- `filename`: Output filename (will append .bson if not present)
"""
function save_model(nn, filename)
    # Ensure filename has .bson extension
    if !endswith(filename, ".bson")
        filename = filename * ".bson"
    end
    
    # Create directory if it doesn't exist
    mkpath(dirname(filename))
    
    try
        # Try the modern approach with state dict
        model_state = Flux.state(nn)
        BSON.@save filename model_state=model_state
        println("Model saved to $filename (using state dict)")
    catch e1
        try
            # Fallback: Extract just the parameters
            model_params = [p for p in Flux.params(nn)]
            BSON.@save filename model_params=model_params
            println("Model saved to $filename (using parameters only)")
            @warn "Saved parameters only. You'll need to reconstruct the model architecture manually."
        catch e2
            # Final fallback: try with HDF5
            try
                h5_filename = replace(filename, ".bson" => ".h5")
                save_model_hdf5(nn, h5_filename)
                println("Model saved to $h5_filename (using HDF5)")
            catch e3
                error("Failed to save model with all methods. BSON error: $e1, Params error: $e2, HDF5 error: $e3")
            end
        end
    end
end

"""
    save_model_hdf5(nn, filename)

Save a Flux model to HDF5 format (more robust alternative).
"""
function save_model_hdf5(nn, filename)
    # Create directory if it doesn't exist
    mkpath(dirname(filename))
    
    # Extract parameters as arrays
    params = Flux.params(nn)
    param_arrays = [Array(p) for p in params]
    
    # Save to HDF5
    h5open(filename, "w") do file
        file["num_params"] = length(param_arrays)
        for (i, param) in enumerate(param_arrays)
            file["param_$i"] = param
            file["param_$(i)_size"] = collect(size(param))
        end
    end
end

"""
    load_model(filename)

Load a Flux model from disk.

# Arguments
- `filename`: Input filename

# Returns
- Loaded neural network model
"""
function load_model(filename)
    # Check if file exists with .bson extension
    bson_filename = endswith(filename, ".bson") ? filename : filename * ".bson"
    h5_filename = replace(bson_filename, ".bson" => ".h5")
    
    if isfile(bson_filename)
        try
            # Try loading from BSON
            model_data = BSON.load(bson_filename)
            
            if haskey(model_data, :model_state)
                # Modern approach with state dict
                @warn "Loading with state dict not fully implemented. Please reconstruct model manually."
                return model_data[:model_state]
            elseif haskey(model_data, :model_params)
                # Parameters only
                @warn "Loaded parameters only. You need to reconstruct the model architecture manually."
                return model_data[:model_params]
            else
                # Legacy approach
                model = model_data[:model]
                if haskey(model_data, :params)
                    params = model_data[:params]
                    Flux.loadparams!(model, params)
                end
                println("Model loaded from $bson_filename")
                return model
            end
        catch e
            @warn "Failed to load BSON file: $e. Trying HDF5..."
        end
    end
    
    if isfile(h5_filename)
        @warn "HDF5 loading returns parameters only. Reconstruct model architecture manually."
        return load_model_hdf5(h5_filename)
    end
    
    error("Neither $bson_filename nor $h5_filename exists")
end

"""
    load_model_hdf5(filename)

Load model parameters from HDF5 format.
"""
function load_model_hdf5(filename)
    param_arrays = []
    
    h5open(filename, "r") do file
        num_params = read(file["num_params"])
        for i in 1:num_params
            param_data = read(file["param_$i"])
            push!(param_arrays, param_data)
        end
    end
    
    return param_arrays
end

"""
    save_model_safe(nn, filename, architecture)

Save a Flux model with architecture information for safe reconstruction.

# Arguments
- `nn`: Neural network model
- `filename`: Output filename
- `architecture`: Array describing layer sizes, e.g., [1, 50, 25, 1]
"""
function save_model_safe(nn, filename, architecture; activation=swish, last_activation=identity)
    # Create directory if it doesn't exist
    mkpath(dirname(filename))
    
    # Ensure filename has .h5 extension for this safe method
    if !endswith(filename, ".h5")
        filename = filename * ".h5"
    end
    
    # Extract parameters as arrays
    params = Flux.params(nn)
    param_arrays = [Array(p) for p in params]
    
    # Save to HDF5 with architecture info
    h5open(filename, "w") do file
        # Save architecture
        file["architecture"] = architecture
        file["num_params"] = length(param_arrays)
        
        # Save activation function names
        file["activation"] = string(activation)
        file["last_activation"] = string(last_activation)
        
        # Save parameters
        for (i, param) in enumerate(param_arrays)
            file["param_$i"] = param
            file["param_$(i)_size"] = collect(size(param))
        end
    end
    
    println("Model safely saved to $filename")
end

"""
    load_model_safe(filename)

Load a model saved with save_model_safe, reconstructing the full architecture.
"""
function load_model_safe(filename)
    # Ensure filename has .h5 extension
    if !endswith(filename, ".h5")
        filename = filename * ".h5"
    end
    
    if !isfile(filename)
        error("File $filename does not exist")
    end
    
    param_arrays = []
    architecture = nothing
    activation_str = "swish"
    last_activation_str = "identity"
    
    h5open(filename, "r") do file
        # Load architecture and activation info
        architecture = read(file["architecture"])
        num_params = read(file["num_params"])
        
        if haskey(file, "activation")
            activation_str = read(file["activation"])
        end
        if haskey(file, "last_activation")
            last_activation_str = read(file["last_activation"])
        end
        
        # Load parameters
        for i in 1:num_params
            param_data = read(file["param_$i"])
            push!(param_arrays, param_data)
        end
    end
    
    # Reconstruct the model
    activation_fn = activation_str == "swish" ? swish : 
                   activation_str == "relu" ? relu :
                   activation_str == "tanh" ? tanh :
                   activation_str == "sigmoid" ? sigmoid : swish
                   
    last_activation_fn = last_activation_str == "identity" ? identity :
                        last_activation_str == "relu" ? relu :
                        last_activation_str == "tanh" ? tanh :
                        last_activation_str == "sigmoid" ? sigmoid : identity
    
    # Build the model layers
    layers = []
    param_idx = 1
    
    for i in 1:(length(architecture)-1)
        # Add dense layer
        W = param_arrays[param_idx]
        b = param_arrays[param_idx + 1]
        
        dense_layer = Dense(W, b)
        push!(layers, dense_layer)
        
        # Add activation (except for last layer)
        if i < length(architecture) - 1
            push!(layers, activation_fn)
        else
            push!(layers, last_activation_fn)
        end
        
        param_idx += 2
    end
    
    model = Chain(layers...)
    println("Model safely loaded from $filename")
    return model
end

"""
    save_variables_to_hdf5(filename::String, vars::Dict; group_path="/")

Save multiple variables to an HDF5 file with proper type handling.

# Arguments
- `filename`: Path to the HDF5 file to save
- `vars`: Dictionary mapping variable names to their values
- `group_path`: Optional group path within the HDF5 file
"""
function save_variables_to_hdf5(filename::String, vars::Dict; group_path="/")
    # Create directory if it doesn't exist
    mkpath(dirname(filename))
    
    # Save variables to HDF5 file
    h5open(filename, "w") do file
        # Create group if it's not the root
        group = group_path == "/" ? file : create_group(file, group_path)
        
        # Save each variable with type handling
        for (name, value) in vars
            if value isa AbstractArray
                # For arrays, save attributes to track dimensions and type
                dataset = group[name] = value
                if eltype(value) <: Complex
                    # Store complex arrays as tuple of real and imaginary parts
                    group["$(name)_complex"] = true
                    group["$(name)_real"] = real(value)
                    group["$(name)_imag"] = imag(value)
                end
            elseif value isa Number
                # For scalar values
                group[name] = [value]
                attrs(group[name])["scalar"] = true
            elseif value isa String
                # For strings
                group[name] = value
            elseif value isa Bool
                # For booleans
                group[name] = [value]
                attrs(group[name])["boolean"] = true
            elseif value isa Symbol
                # For symbols, convert to string
                group[name] = string(value)
                attrs(group[name])["symbol"] = true
            else
                # Try to convert to array for other types
                try
                    group[name] = [value]
                    attrs(group[name])["custom_type"] = string(typeof(value))
                catch e
                    @warn "Could not save variable $name of type $(typeof(value))"
                end
            end
        end
    end
    
    println("Variables saved to $filename")
end

"""
    save_current_workspace(filename::String; exclude_modules=true, exclude_functions=true)

Save all variables in the current workspace to an HDF5 file.
"""
function save_current_workspace(filename::String; exclude_modules=true, exclude_functions=true)
    # Get all variables in the current workspace
    vars = Dict()
    
    for name in names(Main; all=false)
        # Skip excluded types
        value = getfield(Main, name)
        if (exclude_modules && value isa Module) || 
           (exclude_functions && value isa Function) ||
           name == :ans || name == :exclude_modules || name == :exclude_functions
            continue
        end
        
        # Only save variables that can be serialized
        try
            vars[string(name)] = value
        catch e
            @warn "Skipping variable $name: cannot be serialized"
        end
    end
    
    save_variables_to_hdf5(filename, vars)
end

"""
    read_variables_from_hdf5(filename::String; group_path="/")

Read variables from an HDF5 file, reconstructing their types.

# Returns
- Dictionary of variable names to values
"""
function read_variables_from_hdf5(filename::String; group_path="/")
    vars = Dict()
    
    h5open(filename, "r") do file
        # Access the group
        group = group_path == "/" ? file : file[group_path]
        
        # Read each variable with type handling
        for name in keys(group)
            # Skip special complex array components
            if endswith(name, "_real") || endswith(name, "_imag") || endswith(name, "_complex")
                continue
            end
            
            dataset = group[name]
            
            # Check for complex arrays
            if haskey(group, "$(name)_complex") && group["$(name)_complex"][] == true
                real_part = read(group["$(name)_real"])
                imag_part = read(group["$(name)_imag"])
                vars[name] = real_part + im * imag_part
                continue
            end
            
            # Read the data
            data = read(dataset)
            
            # Handle scalar values
            if haskey(attrs(dataset), "scalar") && attrs(dataset)["scalar"]
                vars[name] = data[1]
            # Handle booleans
            elseif haskey(attrs(dataset), "boolean") && attrs(dataset)["boolean"]
                vars[name] = Bool(data[1])
            # Handle symbols
            elseif haskey(attrs(dataset), "symbol") && attrs(dataset)["symbol"]
                vars[name] = Symbol(data)
            else
                vars[name] = data
            end
        end
    end
    
    println("Variables loaded from $filename")
    return vars
end

"""
    load_to_workspace(filename::String; overwrite=false)

Load variables from an HDF5 file into the current workspace.
"""
function load_to_workspace(filename::String; overwrite=false)
    vars = read_variables_from_hdf5(filename)
    
    for (name, value) in vars
        var_sym = Symbol(name)
        
        # Skip if variable exists and overwrite=false
        if !overwrite && isdefined(Main, var_sym)
            @warn "Skipping $name: already exists in workspace"
            continue
        end
        
        # Assign to workspace
        Core.eval(Main, :($(var_sym) = $(value)))
    end
    
    println("Loaded $(length(vars)) variables into workspace")
end