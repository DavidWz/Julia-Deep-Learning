include("deeplearning.jl")

using DeepLearning

input = 10
output = 2
batchSize = 500
epochs = 20
tests = 10

# function to learn: if sum of elements at even positions > sum of elements at odd positions
function label(data)
    even = 2:2:input
    odd = 1:2:input

    evenSum = sum(data[:, even], 2)
    oddSum = sum(data[:, odd], 2)
    label1 = float(evenSum .> oddSum)
    label2 = 1 .- label1
    return [label1 label2]
end

# set up the neural network
println("Creating network...")
f1 = LinearModule(input, output)
f2 = SoftMaxModule(output)
l = CrossEntropyModule(output, zeros(batchSize, output))
net = NeuralNetwork([f1, f2], l)

# train the neural network
println("Training network...")
for i=1:epochs
    println("Epoch #", i)
    for j=1:batchSize
        data = rand(batchSize, input)
        l.target = label(data)

        train(net, data, 0.05)
    end
end

# test the neural network
println("Testing network...")
for i=1:tests
    data = rand(1, input)
    target = label(data)
    result = evaluate(net, data)

    println("Test input: ", data)
    println("Expected result: ", target)
    println("Network predicts: ", result)
end