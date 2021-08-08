import math
import random
random.seed(0)

class ActivationFunctions:
    def sigmoid(array, data):
        # Data is an array of all node values in a layer yet to be "activated"
        # Array is layer.activatedValues so that layers keep track of both "unactivated" and "activated" for
        # backpropogation
        for i in range(len(data)):
            data[i] = 1/(1 + math.exp(-1*data[i]))
            array[i] = data[i]

    def sigmoidPrime(data):
        # Data is scalar
        return math.exp(-1 * data)/((1 + math.exp(-1*data))**2)

    def ReLU(array, data):
        for i in range(len(data)):
            if data[i] < 0:
                data[i] = 0
            array[i] = data[i]

    def ReLUPrime(data):
        # Data is scalar
        if data < 0:
            return 0
        else:
            return 1

class Network:
    def __init__(self):
        self.layer1 = Layer(784, 120)
        self.layer2 = Layer(120, 84)
        self.layer3 = Layer(84, 10)

        self.layers = [self.layer1, self.layer2, self.layer3]

    def __call__(self, data):
        return self.forward(data)
        
    def forward(self, data):
        x = data                # Must be column vector
        x = self.layer1(x)      # Feed through first layer
        self.g(self.layer1, x)  # Activation fn
        x = self.layer2(x)      # Feed through second layer
        self.g(self.layer2, x)  # Activation fn
        x = self.layer3(x)      # feed through final (classifier layer) output is array of 10
        return x

    def g(self, layer, data):
        ActivationFunctions.ReLU(layer.activatedValues, data)

    def gPrime(self, data):
        return ActivationFunctions.ReLUPrime(data)

    def zero_grad(self):
        # Make gradient matrix 0 for all layers for weights and biases
        for layer in self.layers:
            for i in range(len(layer.gradWeights)):
                for j in range(len(layer.gradWeights[i])):
                    layer.gradWeights[i][j] = 0
                layer.gradBias[i] = 0
                layer.gradValues[i] = 0

    def backward(self, original, labelVector):
        for i in range(len(self.layers)):
            # Creating necessary layer references
            currentLayer = self.layers[len(self.layers) - 1 - i]
            if i == len(self.layers) - 1:
                # Last layer
                nextLayer = original # Original data input
            else:
                nextLayer = self.layers[len(self.layers) - 2 - i]

            if i == 0:
                previousLayer = None
            else:
                previousLayer = self.layers[len(self.layers) - i] # Need for backprop

            # Loop over rows of weight matrix
            for j in range(len(currentLayer.weights)):

                # Calculating dcost/da_j
                if previousLayer == None:
                    #Formula with cost
                    dcost_da_j = 2*(currentLayer.activatedValues[j] -labelVector[j]) 
                else:
                    #Formula involving previous layer
                    #for loop over t of which we sum over
                    dcost_da_j = 0
                    for t in range(len(previousLayer.weights)):
                        dcost_da_j += previousLayer.weights[t][j]*self.gPrime(previousLayer.values[t])*previousLayer.gradValues[t]

                currentLayer.gradValues[j] = dcost_da_j # Saving for next layer use

                # Calculate g'(unactivated values of current layer)
                dg_dz_j = self.gPrime(currentLayer.values[j])

                # Store gradBias
                currentLayer.gradBias[j] = dg_dz_j*dcost_da_j

                # For loop over k (columns of weights)
                if i == len(self.layers) - 1:
                    # Last layer is just an array of data, not Layer object
                    for k in range(len(currentLayer.weights[j])):
                        # Store result in layer.gradWeights
                        a_k = nextLayer[k]
                        
                        currentLayer.gradWeights[j][k] = a_k * currentLayer.gradBias[j]
                else:
                    for k in range(len(currentLayer.weights[j])):
                        # Store result in layer.gradWeights
                        a_k = nextLayer.activatedValues[k]
                        
                        currentLayer.gradWeights[j][k] = a_k * currentLayer.gradBias[j]

    def step(self, learningRate = 0.001):
        # Update all weights and biases in each layer according to learning rate and gradient
        # values stored
        for layer in self.layers:
            # Iterate over rows (i)
            for i in range(len(layer.weights)):
                # Iterate over columns (j)
                for j in range(len(layer.weights[i])):
                    # Stochastic Gradient Descent
                    layer.weights[i][j] = layer.weights[i][j] - learningRate*layer.gradWeights[i][j]
                
                # Update biases
                layer.biases[i] = layer.biases[i] - learningRate*layer.gradBias[i]

class Layer:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.values = [0]*self.outputs # Array to hold values, ultimately a copy is returned
        self.activatedValues = [0]*self.outputs

        self.weights = [[random.random() for _ in range(self.inputs)] for _ in range(self.outputs)]     # Create row x col matrix
        self.biases = [random.random()]*self.outputs

        self.gradWeights = [[0 for _ in range(self.inputs)] for _ in range(self.outputs)]     # Create row x col matrix
        self.gradBias = [0]*self.outputs

        self.gradValues = [0]*self.outputs

    def __call__(self, data):
        self.values = self.updateValues(data) # Updates values in layer according to weightings and biases

        copy = [0]*len(self.values)
        for i in range(len(self.values)):
            copy[i] = self.values[i]

        # Return the copy
        return copy

    def updateValues(self, vector):
        row = len(self.weights)
        col = len(vector)
        res = [0]*row

        # Weights (out x in) * Vector (1 x in)
        for i in range(row):
            c = 0
            for j in range(col):
                c += self.weights[i][j]*vector[j]
            res[i] = c

        # Res is (1 x out)
        # + Bias
        for i in range(len(res)):
            res[i] = res[i] + self.biases[i]

        return res


model = Network()

# Example
# x = [random.normalvariate(0, 1) for _ in range(784)]
# print(x)
# output = model(x)
# print(output)
# label = [4, 3, 0, 4, 0, 81, 0, 7, 0, 1]

# dataset = []

def train(model, num_epochs, dataset, learning_rate):
    for epoch in range(num_epochs):  # loop over the dataset
        for i in range(len(dataset)):
            avgCost = 0
            # Get test data
            input, label = dataset[i]

            # Zero all gradients
            model.zero_grad()

            # Forward + Backward + Step
            output = model(input)
            model.backward(input, label)
            model.step(learning_rate)

            # Stats
            cost = [(output[j] - label[j])**2 for j in range(len(output))]
            total = 0
            for item in cost:
                total += item
            avgCost = total/len(cost)

            print(f"[{epoch + 1}, {i + 1}] loss: {avgCost}")

def validate(model, dataset):
    for i in range(len(dataset)):
        numCorrect = 0

        # Get validation data
        input, label = dataset[i] # label is an index

        # Forward
        output = model(input)

        # Compare
        max = output[0]
        max_idx = 0
        for i in range(1, len(output)):
            if output[i] > max:
                max = output
                max_idx = i
        
        if max_idx == label:
            numCorrect += 1

    print(f"Total accuracy: {numCorrect}/{len(dataset)} = {round(numCorrect/len(dataset) *100, 2)}")