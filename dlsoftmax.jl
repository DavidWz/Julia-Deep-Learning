import Base.size

export SoftMaxModule
export fprop, bprop, descend

# a softmax module
mutable struct SoftMaxModule
    # number of nodes
    nodes::Int64

    # cached output of fprop for efficiency
    cachedOutput::Matrix

    # constructor of a softmax module
    function SoftMaxModule(nodes::Int64)
        new(nodes, zeros(1, nodes))
    end
end

# The size of this module as (inputNodes, outputNodes)
function size(f::SoftMaxModule)
    (f.nodes, f.nodes)
end


# computes the softmax of a vector x
# x size: [batchSize, dimension] 
softmax(x::Matrix) = exp.(x) ./ sum(exp.(x), 2)


# Applies softmax to a vector z
# z size: [batchSize, dimension] 
function fprop(f::SoftMaxModule, z::Matrix)
    result = softmax(z)
    f.cachedOutput = result
    return result
end


# Backward propagation of a given vector v
# returns the gradient with regards to the input
# Note: fprop must have been called
function bprop(f::SoftMaxModule, v::Matrix)
    batchSize = size(v)[1]
    result = zeros(batchSize, f.nodes)
    for b=1:batchSize
        tmp = dot(v[b, :], f.cachedOutput[b, :])
        for i=1:f.nodes
            result[b, i] = f.cachedOutput[b, i] * (v[b, i] - tmp)
        end
    end
    return result
end


# Adjusts the parameters by gradient descend
function descend(f::SoftMaxModule, rate)
    # no parameters, so nothing to do here
end