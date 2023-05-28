## USER
Write a PYTHON program to scheme a few activation functions that are used in neural networks.

## ASSISTANT
Here's a Python program that implements several common activation functions used in neural networks:

```python
import numpy as np

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def softmax(x):
    """Softmax activation function"""
    exp_vals = np.exp(x)
    return exp_vals / np.sum(exp_vals)

def tanh(x):
    """Hyperbolic tangent activation function"""
    return np.tanh(x)

# Test the activation functions
x = np.array([1, 2, 3, 4, 5])

# Sigmoid
print("Sigmoid:", sigmoid(x))

# ReLU
print("ReLU:", relu(x))

# Softmax
print("Softmax:", softmax(x))

# Hyperbolic tangent
print("Hyperbolic tangent:", tanh(x))
```

In the code above, we define four commonly used activation functions: sigmoid, ReLU (Rectified Linear Unit), softmax, and hyperbolic tangent (tanh). Each function takes an input array `x` and applies the corresponding activation function element-wise, returning the transformed array.

To test the activation functions, we create a sample input array `x` with values [1, 2, 3, 4, 5]. We then apply each activation function to this array and print the results.

Note: The code assumes you have the NumPy library installed, as it is used for mathematical operations on arrays.

## USER
Write a PYTHON program to show back propagation network for XOR function with binary input and output

## ASSISTANT
Certainly! Here's a Python program that demonstrates a backpropagation neural network for the XOR function with binary input and output:

```python
import numpy as np

# Define the activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the activation function (sigmoid)
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the XOR function inputs and outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Set the random seed for reproducibility
np.random.seed(42)

# Initialize the weights randomly
weights_1 = np.random.random((2, 2))
weights_2 = np.random.random((2, 1))

# Set the learning rate
learning_rate = 0.1

# Train the neural network
for epoch in range(10000):
    # Forward propagation
    layer_1 = sigmoid(np.dot(X, weights_1))
    output = sigmoid(np.dot(layer_1, weights_2))

    # Backpropagation
    output_error = y - output
    output_delta = output_error * sigmoid_derivative(output)

    layer_1_error = output_delta.dot(weights_2.T)
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    # Update weights
    weights_2 += layer_1.T.dot(output_delta) * learning_rate
    weights_1 += X.T.dot(layer_1_delta) * learning_rate

# Test the trained neural network
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output = sigmoid(np.dot(sigmoid(np.dot(test_input, weights_1)), weights_2))

print("Predicted Output:")
print(predicted_output)
```

In the above program, we define the XOR function inputs `X` and outputs `y` as binary arrays. We initialize the weights randomly and set the learning rate. The neural network is trained using backpropagation through a loop of forward propagation, error calculation, and weight updates.

The activation function used is the sigmoid function, defined by the `sigmoid` function. The derivative of the sigmoid function is also defined by the `sigmoid_derivative` function, which is required for backpropagation.

After training, we test the neural network on the same input array `X` and print the predicted output.

Please note that the program uses the NumPy library for efficient array operations and the sigmoid activation function.

## USER
Write a PYTHON program for producing back propagation feed forward network

## ASSISTANT
Certainly! Here's a Python program that implements a backpropagation feedforward neural network:

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        for i in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[i], layers[i-1]))
            self.biases.append(np.random.randn(layers[i], 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, input):
        a = input
        activations = [a]
        zs = []
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], a) + self.biases[i]
            zs.append(z)
            a = self.sigmoid(z)
            activations.append(a)
        return activations, zs

    def backpropagation(self, input, target):
        activations, zs = self.feedforward(input)
        output = activations[-1]

        delta = (output - target) * self.sigmoid_derivative(output)
        nabla_b = [delta]
        nabla_w = [np.dot(delta, activations[-2].T)]

        for i in range(2, len(self.layers)):
            z = zs[-i]
            sp = self.sigmoid_derivative(activations[-i])
            delta = np.dot(self.weights[-i+1].T, delta) * sp
            nabla_b.insert(0, delta)
            nabla_w.insert(0, np.dot(delta, activations[-i-1].T))

        return nabla_b, nabla_w

    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            for input, target in zip(inputs, targets):
                delta_nabla_b, delta_nabla_w = self.backpropagation(input, target)
                nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.weights = [w - (learning_rate/len(inputs)) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (learning_rate/len(inputs)) * nb for b, nb in zip(self.biases, nabla_b)]

    def predict(self, input):
        activations, _ = self.feedforward(input)
        return activations[-1]

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create a neural network with 2 input neurons, 2 hidden neurons, and 1 output neuron
network = NeuralNetwork([2, 2, 1])

# Train the network
network.train(X, y, epochs=10000, learning_rate=0.1)

# Predict the outputs for the inputs
predictions = [network.predict(x) for x in X]
print("Predictions:")
print(predictions)
```

In the above program, we define a `NeuralNetwork` class that represents a backpropagation feedforward neural network. The network is initialized with a list of layer sizes, and the weights and biases are

## USER
Write a PYTHON program to demonstrate ART

## ASSISTANT
Certainly! Here's a Python program that demonstrates an Adaptive Resonance Theory (ART) neural network:

```python
import numpy as np

class ARTNetwork:
    def __init__(self, input_size, rho, vigilance):
        self.input_size = input_size
        self.rho = rho
        self.vigilance = vigilance
        self.categories = []
        self.weights = []

    def initialize_categories(self, input):
        self.categories.append(input)
        self.weights.append(input)

    def compute_similarity(self, input, category):
        return np.sum(np.minimum(input, category)) / np.sum(input)

    def find_matching_category(self, input):
        for i in range(len(self.categories)):
            similarity = self.compute_similarity(input, self.categories[i])
            if similarity >= self.vigilance:
                return i
        return None

    def create_new_category(self, input):
        self.categories.append(input)
        self.weights.append(np.maximum(input, self.rho))

    def update_category(self, input, category_index):
        self.weights[category_index] = np.maximum(self.weights[category_index], np.minimum(input, self.rho))

    def train(self, inputs):
        for input in inputs:
            matching_category = self.find_matching_category(input)
            if matching_category is None:
                self.create_new_category(input)
            else:
                self.update_category(input, matching_category)

    def predict(self, input):
        matching_category = self.find_matching_category(input)
        if matching_category is not None:
            return self.categories[matching_category]
        else:
            return None

# Example usage
X = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]])
y = np.array([[1], [2], [3]])

# Create an ART network with input size of 4, rho value of 0.3, and vigilance value of 0.9
network = ARTNetwork(input_size=4, rho=0.3, vigilance=0.9)

# Initialize the network with the first input
network.initialize_categories(X[0])

# Train the network
network.train(X)

# Predict the category for the second input
prediction = network.predict(X[1])
print("Prediction:")
print(prediction)
```

In the above program, we define an `ARTNetwork` class that represents an Adaptive Resonance Theory (ART) neural network. The network is initialized with the input size, rho value, and vigilance value.

The `initialize_categories` method initializes the network with the first input, creating a category and weight vector.

The `compute_similarity` method calculates the similarity between an input and a category using the minimum operator.

The `find_matching_category` method iterates over the existing categories and checks if any category matches the input based on the vigilance parameter.

The `create_new_category` method creates a new category if no matching category is found.

The `update_category` method updates the weight vector of the matching category.

The `train` method trains the network by iterating over the inputs and performing the matching and updating steps.

The `predict` method predicts the category for a given input by finding the matching category.

In the example usage, we create an ART network with an input size of 4, rho value of 0.3, and vigilance value of 0.9. We initialize the network with the first input and train it using the inputs. Finally, we predict the category for the second input.

Please note that the program assumes the inputs (`X`) are binary arrays, and the categories (`y`) are numeric labels.

## USER
Write a PYTHON program to demonstrate the perceptron learning law with its decision region using python. Give the output in graphical form

## ASSISTANT
Certainly! Here's a Python program that demonstrates the perceptron learning law and visualizes its decision region using Matplotlib:

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.1, max_epochs=100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def train(self, X, y):
        self.weights = np.zeros(X.shape[1] + 1)
        self.errors = []

        for _ in range(self.max_epochs):
            error_count = 0
            for xi, target in zip(X, y):
                xi = np.insert(xi, 0, 1)  # Inserting bias term
                update = self.learning_rate * (target - self.predict(xi))
                self.weights += update * xi
                error_count += int(update != 0.0)
            self.errors.append(error_count)

    def predict(self, xi):
        activation = np.dot(xi, self.weights)
        return np.where(activation >= 0.0, 1, -1)

# Generate random training data
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.where(X[:, 0] + X[:, 1] >= 0, 1, -1)

# Train the perceptron
perceptron = Perceptron()
perceptron.train(X, y)

# Plot the decision region
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Perceptron Decision Region')
plt.show()
```

In the above program, we define a `Perceptron` class that implements the perceptron learning algorithm. The `train` method trains the perceptron on the given training data by updating the weights iteratively. The `predict` method predicts the class label for a given input based on the learned weights.

We generate random training data, `X`, with two features, and corresponding class labels, `y`. We then create an instance of the `Perceptron` class and train it on the data. After training, we use the learned weights to predict the class labels for a grid of points spanning the feature space. Finally, we plot the decision region and scatter plot of the training data using Matplotlib.

Please note that the program assumes a binary classification problem, where class labels are either 1 or -1. The program uses the NumPy and Matplotlib libraries for numerical operations and visualization, respectively.
