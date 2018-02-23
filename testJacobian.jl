include("deeplearning.jl")
using DeepLearning

# computes the jacobian matrix from fprop and bprop at a given input x
# returns a tuple of both
function getJacobians(f, x::Matrix)
    (input, output) = size(f)

    # compute jacobian with fprop
    J_f = zeros(output, input)
    for i=1:input
        h = sqrt(eps(Float64)) * max(x[1, i], 1)
        x1 = copy(x)
        x2 = copy(x)
        x1[1, i] += h
        x2[1, i] -= h

        J_f[:, i] = (fprop(f, x1) - fprop(f, x2)) / (2*h)
    end

    # compute jacobian with bprop
    J_b = zeros(output, input)
    fprop(f, x)
    for i=1:output
        z = zeros(1, output)
        z[i] = 1
        J_b[i, :] = bprop(f, z)
    end

    return (J_f, J_b)
end

# config
runs = 10
maxSize = 1000

# linear module
println("Testing Linear Module...")
for i=1:runs
    nodes1 = rand(1:maxSize)
    nodes2 = rand(1:maxSize)

    f = LinearModule(nodes1, nodes2)
    x = rand(1, nodes1)

    (J_f, J_b) = getJacobians(f, x)
    relError = norm(J_f - J_b)/norm(J_f)
    println(nodes1, " x ", nodes2, " nodes, error: ", relError)
end

# softmax module
println("Testing SoftMax Module...")
for i=1:runs
    nodes = rand(1:maxSize)

    f = SoftMaxModule(nodes)
    x = rand(1, nodes)

    (J_f, J_b) = getJacobians(f, x)
    relError = norm(J_f - J_b)/norm(J_f)
    println(nodes, " nodes, error: ", relError)
end

# cross entropy module
println("Testing CrossEntropy Module...")
for i=1:runs
    nodes = rand(1:maxSize)

    x = rand(1, nodes)
    t = rand(1, nodes)
    t /= sum(t)
    f = CrossEntropyModule(nodes, t)

    (J_f, J_b) = getJacobians(f, x)
    relError = norm(J_f - J_b)/norm(J_f)
    println(nodes, " nodes, error: ", relError)
end

# tanh module
println("Testing Tanh Module...")
for i=1:runs
    nodes = rand(1:maxSize)

    f = TanhModule(nodes)
    x = rand(1, nodes)

    (J_f, J_b) = getJacobians(f, x)
    relError = norm(J_f - J_b)/norm(J_f)
    println(nodes, " nodes, error: ", relError)
end

# logsoftmax module
println("Testing LogSoftMax Module...")
for i=1:runs
    nodes = rand(1:maxSize)

    f = LogSoftMaxModule(nodes)
    x = rand(1, nodes)

    (J_f, J_b) = getJacobians(f, x)
    relError = norm(J_f - J_b)/norm(J_f)
    println(nodes, " nodes, error: ", relError)
end

# cross entropy log module
println("Testing CrossEntropyLog Module...")
for i=1:runs
    nodes = rand(1:maxSize)

    x = rand(1, nodes)
    t = rand(1, nodes)
    t /= sum(t)
    f = CrossEntropyLogModule(nodes, t)

    (J_f, J_b) = getJacobians(f, x)
    relError = norm(J_f - J_b)/norm(J_f)
    println(nodes, " nodes, error: ", relError)
end