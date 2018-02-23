include("../deeplearning.jl")
using DeepLearning

# load training data
println("Loading data.")
trainData = readdlm("mnist/mnist-train-data.csv")
trainLabelRaw = readdlm("mnist/mnist-train-labels.csv")
testData = readdlm("mnist/mnist-test-data.csv")
testLabelRaw = readdlm("mnist/mnist-test-labels.csv")
numData = size(trainData)[1]
numTest = size(testData)[1]

# prepare training data
trainData = trainData ./ 255
testData = testData ./ 255
# split the label into 10 different outputs (one for each digit)
trainLabel = zeros(numData, 10)
for i=1:numData
    trainLabel[i, Int64(trainLabelRaw[i]+1)] = 1
end
testLabel = zeros(numTest, 10)
for i=1:numTest
    testLabel[i, Int64(testLabelRaw[i]+1)] = 1
end

# config
epochs = 100
learningRate = 0.1
batchSize = 600
numBatches = numData / batchSize

# set up the network
println("Setting up network.")

# Deep net
f1 = LinearModule(28*28, 200)
f2 = TanhModule(200)
f3 = LinearModule(200, 10)
f4 = LogSoftMaxModule(10)
criterion = CrossEntropyLogModule(10, zeros(batchSize, 10))
net = NeuralNetwork([f1, f2, f3, f4], criterion)


# train the network
println("Started training.")
for i=1:epochs
    print("Epoch #", i, ", ")
    avgCosts = 0

    # shuffle data
    order = shuffle(1:numData)
    shuffledData = trainData[order, :]
    shuffledLabel = trainLabel[order, :]

    # train the batches
    for j=1:numBatches
        batchStart = Int64((j-1)*batchSize+1)
        batchEnd = Int64(min(j*batchSize, numData))

        dataBatch = shuffledData[batchStart:batchEnd, :]
        criterion.target = shuffledLabel[batchStart:batchEnd, :]

        # train the network on this batch
        costs = train(net, dataBatch, learningRate)
        avgCosts += sum(costs, 1) / size(costs)[1]
    end
    avgCosts /= numBatches
    println("avg costs: ", avgCosts)
end

# test the network
println("Started testing.")
# for logsoftmax, the result must be exponentiated
testResult = round.(exp.(evaluate(net, testData)))

# compare rows
numIncorrect = 0
for i=1:numTest
    if testResult[i, :] != testLabel[i, :]
        numIncorrect += 1
    end
end

println(numIncorrect, " out of ", numTest, " misclassified test points")
correct = (numTest - numIncorrect) / numTest * 100
println(correct, "% accuracy.")

# storing the resulting network
println("Store the network.")
writedlm("f1W.txt", f1.W)
writedlm("f1b.txt", f1.b)
writedlm("f3W.txt", f3.W)
writedlm("f3b.txt", f3.b)