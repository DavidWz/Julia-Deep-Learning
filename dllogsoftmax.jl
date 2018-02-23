import Base.size

export LogSoftMaxModule
export fprop, bprop, descend

# a log softmax module
mutable struct LogSoftMaxModule
    # number of nodes
    nodes::Int64

    # cached output of fprop for efficiency
    cachedOutput::Matrix

    # constructor of a log softmax module
    function LogSoftMaxModule(nodes::Int64)
        new(nodes, zeros(1, nodes))
    end
end

# The size of this module as (inputNodes, outputNodes)
function size(f::LogSoftMaxModule)
    (f.nodes, f.nodes)
end


# computes the logsoftmax of a vector x
# x size: [batchSize, dimension] 
function logsoftmax(x::Matrix)
    y = x .- maximum(x, 2)
    return y .- log.(sum(exp.(y), 2))
end


# Applies log softmax to a vector z
# z size: [batchSize, dimension] 
function fprop(f::LogSoftMaxModule, z::Matrix)
    result = logsoftmax(z)
    f.cachedOutput = result
    return result
end


# Backward propagation of a given vector v
# returns the gradient with regards to the input
# Note: fprop must have been called
function bprop(f::LogSoftMaxModule, v::Matrix)
    batchSize = size(v)[1]
    result = zeros(batchSize, f.nodes)
    for b=1:batchSize
        tmp = sum(v[b, :])
        for i=1:f.nodes
            result[b, i] = v[b, i] - exp(f.cachedOutput[b, i]) * tmp
        end
    end
    return result
end


# Adjusts the parameters by gradient descend
function descend(f::LogSoftMaxModule, rate)
    # no parameters, so nothing to do here
end