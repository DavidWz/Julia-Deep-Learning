import Base.size

export CrossEntropyLogModule
export fprop, bprop

# a cross entropy log criterion module
mutable struct CrossEntropyLogModule
    # number of nodes
    nodes::Int64

    # target labels
    target::Matrix

    # constructor of a cross entropy module
    function CrossEntropyLogModule(nodes::Int64, target::Matrix)
        new(nodes, target)
    end
end

# The size of this module as (inputNodes, outputNodes)
function size(f::CrossEntropyLogModule)
    (f.nodes, 1)
end


# Applies cross entropy to a vector z with regards to the labels t
# z, labels size: [batchSize, dimension] 
function fprop(f::CrossEntropyLogModule, z::Matrix)
    return -sum(f.target .* z, 2)
end


# Backward propagation of a given vector v
# returns the gradient with regards to the input
function bprop(f::CrossEntropyLogModule, v::Matrix)
    return v .* -f.target
end