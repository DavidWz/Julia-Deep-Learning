using Distributions

import Base.size

export LinearModule
export fprop, bprop, descend

# a fully connected linear module
mutable struct LinearModule
    # the weight matrix
    W::Matrix
    # the bias vector
    b::Matrix

    # the latest input vector (cached for efficiency in bprop)
    cachedInput::Matrix

    # the gradients
    grad_W::Matrix
    grad_b::Matrix

    # constructor of a linear module
    function LinearModule(inputNodes::Int64, outputNodes::Int64)
        # create a gaussian distribution for initial weights
        dist = Normal(0, sqrt(2/(inputNodes+outputNodes)))
        W = rand(dist, inputNodes, outputNodes)
        b = zeros(1, outputNodes)
        cachedInput = zeros(1, inputNodes)
        grad_W = zeros(inputNodes, outputNodes)
        grad_b = zeros(1, outputNodes)
        new(W, b, cachedInput, grad_W, grad_b)
    end
end

# The size of this module as (inputNodes, outputNodes)
function size(f::LinearModule)
    size(f.W)
end

# Forward propagation of an input vector z in the linear module
# z size: [batchSize, dimension] 
function fprop(f::LinearModule, z::Matrix)
    # cache the input for later reuse in bprop
    f.cachedInput = z

    # propagate forwards
    batchSize = size(z)[1]
    broadB = repmat(f.b, batchSize, 1)
    return z * f.W + broadB
end


# Backward propagation of a given gradient output
# returns the gradient with regards to the input
# Note: fprop must have been called
function bprop(f::LinearModule, grad_output::Matrix)
    # gradient of the parameters
    f.grad_W = f.cachedInput' * grad_output
    f.grad_b = sum(grad_output, 1)

    # gradient of the input
    return grad_output * f.W'
end


# Adjusts the parameters by gradient descend
# Note: bprop must have been called
function descend(f::LinearModule, rate)
    f.W = f.W - rate * f.grad_W
    f.b = f.b - rate * f.grad_b
end