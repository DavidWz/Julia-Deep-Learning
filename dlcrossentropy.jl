import Base.size

export CrossEntropyModule
export fprop, bprop

# a cross entropy criterion module
mutable struct CrossEntropyModule
    # number of nodes
    nodes::Int64

    # target labels
    target::Matrix

    # cached input of fprop for efficiency in bprop
    cachedInput::Matrix 

    # constructor of a cross entropy module
    function CrossEntropyModule(nodes::Int64, target::Matrix)
        new(nodes, target, zeros(1, nodes))
    end
end

# The size of this module as (inputNodes, outputNodes)
function size(f::CrossEntropyModule)
    (f.nodes, 1)
end


# Applies cross entropy to a vector z with regards to the labels t
# z, labels size: [batchSize, dimension] 
function fprop(f::CrossEntropyModule, z::Matrix)
    f.cachedInput = z
    return -sum(f.target .* log.(z), 2)
end


# Backward propagation of a given vector v
# returns the gradient with regards to the input
# Note: fprop must have been called
function bprop(f::CrossEntropyModule, v::Matrix)
    return v .* (-f.target ./ f.cachedInput)
end