import Base.size

export TanhModule
export fprop, bprop, descend

# a tanh module
mutable struct TanhModule
    # number of nodes
    nodes::Int64

    # cached output of fprop for efficiency
    cachedOutput::Matrix

    # constructor of a tanh module
    function TanhModule(nodes::Int64)
        new(nodes, zeros(1, nodes))
    end
end

# The size of this module as (inputNodes, outputNodes)
function size(f::TanhModule)
    (f.nodes, f.nodes)
end


# Applies tanh to a vector z
# z size: [batchSize, dimension] 
function fprop(f::TanhModule, z::Matrix)
    result = tanh.(z)
    f.cachedOutput = result
    return result
end


# Backward propagation of a given vector v
# returns the gradient with regards to the input
# Note: fprop must have been called
function bprop(f::TanhModule, v::Matrix)
    return v .* (1 - f.cachedOutput .^ 2)
end


# Adjusts the parameters by gradient descend
function descend(f::TanhModule, rate)
    # no parameters, so nothing to do here
end