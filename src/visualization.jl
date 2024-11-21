
# Function that generates a 2D vector field plot
function vectorfield2d(field, points, arrowlength=0.1; xlabel="x", ylabel="y", title="Vector Field")
    errormessage = "Incorrect formatting of points. Please format them as [x1 y1; x2, y2;...]"
    if typeof(points) <: Array{<:Number, 2} && size(points)[1] === 2
        vectors = similar(points)
        for i in 1:size(points)[2]
            vectors[:, i] .= collect(field(points[:, i]...))
        end
    else
        error(errormessage)
    end
    vectors .*= arrowlength
    plt = quiver(points[1, :],points[2, :],quiver=(vectors[1, :], vectors[2, :]), xlabel=xlabel, ylabel=ylabel, title=title)
    return plt
end

# Function that generates a meshgrid
function meshgrid(n)
    xs = ones(n) .* (1:n)'
    ys = xs'
    xys = permutedims(cat(xs, ys; dims = 3), [3, 1, 2])
    return reshape(xys, 2, n^2)
end