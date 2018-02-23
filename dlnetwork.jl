export NeuralNetwork
export evaluate, train

mutable struct NeuralNetwork
    # an array of modules
    modules

    # a criterion
    criterion
end

# evaluate the neural network on some data (without loss function in the end)
# and returns the costs
function evaluate(net::NeuralNetwork, data::Matrix)
    z = data
    for f=net.modules
        z = fprop(f, z)
    end
    return z
end

# trains the neural network on some batch of data with some learning rate
function train(net::NeuralNetwork, data::Matrix, rate)
    batchSize = size(data)[1]

    # evaluate network and determine costs/loss
    result = evaluate(net, data)
    costs = fprop(net.criterion, result)

    # back propagation
    initDelta = ones(batchSize, 1) ./ batchSize
    deltaZ = bprop(net.criterion, initDelta)
    for f=reverse(net.modules)
        deltaZ = bprop(f, deltaZ)
    end

    # adjust parameters
    for f=net.modules
        descend(f, rate)
    end

    return costs
end