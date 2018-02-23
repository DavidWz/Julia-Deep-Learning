include("../deeplearning.jl")
using Images, DeepLearning, FileIO

# load the network
println("Setting up network.")

f1 = LinearModule(28*28, 200)
f2 = TanhModule(200)
f3 = LinearModule(200, 10)
f4 = SoftMaxModule(10)
net = NeuralNetwork([f1, f2, f3, f4], nothing)

f1.W = readdlm("f1W.txt")
f1.b = readdlm("f1b.txt")
f3.W = readdlm("f3W.txt")
f3.b = readdlm("f3b.txt")

# load the input image
if length(ARGS) < 1
  println("arguments needed: pass file name of an image")
  quit()
end

imagePath = ARGS[1]
rawInputImage = load(imagePath)
inputImage = 1 .- imresize(rawInputImage, 28, 28)
inputImage = reshape(inputImage', 1, 28*28)

# run the image through the network
result = evaluate(net, inputImage)

# display result
for i=1:10
    println(i-1, ": ", round(result[i]*100), "%")
end