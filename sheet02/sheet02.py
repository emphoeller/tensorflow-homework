'''
Implementing ANNs with TensorFlow 2021/2022
Homework Sheet 02

Implements a net of perceptrons and uses the backpropagation alorithm to train
them on logical gates.
'''

import numpy as np
from math import exp
import matplotlib.pyplot as plt

def sigmoid(x):
  '''
  Returns the value of the logistic function at the given point.
  
  Args:
    x (float): The point for which to evalue the logistic function.
  
  Returns:
    float: The value of the logistic function at point x.
  '''
  return 1 / (1 + exp(-x))

def sigmoid_to_sigmoidprime(s):
  '''
  Computes the value of the derivative of the logistic function from the value
  of the logistic function at that point.
  
  Args:
    s (float): The value of the logistic function at point x.
  
  Returns:
    float: The value of the derivative of the logistic function at point x.
  '''
  return s * (1 - s)

def sigmoidprime(x):
  '''
  Returns the value of the derivative of the logistic function at the given
  point.
  
  Args:
    x (float): The point for which to evalue the derivative of the logistic 
      function.
  
  Returns:
    float: The value of the derivative of the logistic function at point x.
  '''
  return sigmoid_to_sigmoidprime(sigmoid(x))

# Using a Generator instance is the recommended way to generate random numbers
# in numy these days. This way, we can easily modify the seed in one central
# location.
rng = np.random.default_rng(1234)

class Perceptron:
  '''
  Implements a perceptron, with functionality for evaluation and
  backpropagation.
  '''
  
  def __init__(self, input_units):
    '''
    Creates a Perceptron instance, given the number of its inputs.
    
    Args:
      input_units (int): The number of inputs this perceptron accepts. (I.e.,
        the number of perceptrons on the previous layer or the number of inputs
        for a layer 1 perceptron.)
    '''
    # We need one additional weight as the bias
    self.weights = rng.standard_normal(size = input_units + 1)
    # Now we want to turn the column vector into a row vector so we can just
    # multiply it with the inputs later. A 1-D array will always be interpreted
    # as a column vector, so we need a second dimension:
    self.weights = self.weights[:, np.newaxis].transpose()
    # We always use a learning rate of 1 in this example
    self.alpha = 1
  
  def forward_step(self, inputs):
    '''
    Evaluates the activation of this perceptron given input values. Also
    remembers inputs and activation for a subsequent call of the update method
    (i.e. backpropagation).
    
    Args:
      inputs (array_like of float): An array of the perceptron’s inputs. Must
        have a length equal to the number input_units passed to __init__().
    
    Returns:
      float: This perceptron’s activation.
    '''
    # Append 1 to multiply with the bias and save for backpropagation
    self.inputs = np.append(arr = inputs, values = (1,))
    # In this case, the result of matmul is an array with shape (1,), so we
    # just need to take out that one value
    drive = np.matmul(self.weights, self.inputs)[0]
    # Save this for backpropagation as well
    self.activation = sigmoid(drive)
    return self.activation
  
  def update(self, delta):
    '''
    Recomputes this perceptron’s weights and bias given the error signal.
    
    Args:
      delta (float): The error signal.
    '''
    gradients = delta * self.inputs
    self.weights -= self.alpha * gradients

class MLP:
  '''
  Implements a network of Perceptrons (multi-layer perceptron), with
  functionality for training via backpropagation.
  
  This implementation always
    – takes two inputs;
    – has one hidden layer with four hidden perceptrons and an output layer
      with one perceptron;
    – uses the logistic function as the activation function;
    – uses a learning rate of 1.
  '''
  
  def __init__(self):
    '''
    Initializes a new MLP instance.
    '''
    # Initialize hidden layer with four perceptrons that each take two inputs
    # (the inputs of the respective logical function)
    self.hidden_layer = tuple(Perceptron(2) for i in range(4))
    # The output perceptron will receive inputs from the four hidden
    # perceptrons
    self.output_perceptron = Perceptron(4)
  
  def forward_step(self, inputs):
    '''
    Runs the given inputs through the network and returns the result.
    
    Args:
      inputs (array_like of float): The two inputs to be passed into the
        network.
    
    Retruns:
      float: The activation of the single output layer perceptron.
    '''
    # Pass the inputs through the hidden layer
    hidden_layer_result = tuple(
      perceptron.forward_step(inputs) for perceptron in self.hidden_layer
    )
    # Pass the result of the hidden layer into the output perceptron
    output = self.output_perceptron.forward_step(hidden_layer_result)
    # Its output is the final output of the network
    return output
  
  def backprop_step(self, target_output):
    '''
    Adjusts weights of perceptrons via backpropagation, given the target value
    for the last forward_step.
    
    Args:
      target_output (float): The target value for the last inputs passed to
        forward_step.
    '''
    # 1. Compute the error signal for the output neuron
    # We can reuse the activation that we computed earlier to compute
    # sigmoidpirime of the drive
    # (We’re not sure about the “2 *” here – Courseware has it, the footnote on
    # the sheet does not.)
    output_perceptron_delta = (
      2 * (-target_output + self.output_perceptron.activation) *
      sigmoid_to_sigmoidprime(self.output_perceptron.activation)
    )
    # 2. Compute the error signal for the hidden layer perceptrons
    # (We need to do this before updating the output perceptron because we need
    # its old weights)
    hidden_layer_deltas = tuple(
      output_perceptron_delta * self.output_perceptron.weights[0][i] *
      sigmoid_to_sigmoidprime(perceptron.activation)
      for i, perceptron in enumerate(self.hidden_layer)
    )
    # 3. Update the output perceptron now with its error signal
    self.output_perceptron.update(output_perceptron_delta)
    # 4. Update the hidden perceptrons with their respective error signals
    for perceptron, delta in zip(self.hidden_layer, hidden_layer_deltas):
      perceptron.update(delta)

# Store the training data in an iteration-friendly format:
# n-tuple of our logical gates:
#   2-tuple of logical gate data:
#     1. logical gate name
#     2. 2-tuple for input-output-mapping:
#       1. 2-tuple of inputs:
#         1. first input to gate
#         2. second input to gate
#       2. that gate’s output for these two inputs
logical_gates = (
  ('AND', (
    ((0, 0), 0),
    ((0, 1), 0),
    ((1, 0), 0),
    ((1, 1), 1),
  )),
  ('OR', (
    ((0, 0), 0),
    ((0, 1), 1),
    ((1, 0), 1),
    ((1, 1), 1),
  )),
  ('NAND', (
    ((0, 0), 1),
    ((0, 1), 1),
    ((1, 0), 1),
    ((1, 1), 0),
  )),
  ('NOR', (
    ((0, 0), 1),
    ((0, 1), 0),
    ((1, 0), 0),
    ((1, 1), 0),
  )),
  ('XOR', (
    ((0, 0), 0),
    ((0, 1), 1),
    ((1, 0), 1),
    ((1, 1), 0),
  )),
  # bonus
  ('XNOR', (
    ((0, 0), 1),
    ((0, 1), 0),
    ((1, 0), 0),
    ((1, 1), 1),
  )),
)

n_gates = len(logical_gates)
n_epochs = 1000
training_results = []
for label, data in logical_gates:
  # Train a different MLP for each gate
  mlp = MLP()
  n_datapoints = len(data)
  # ndarray for storing performance metrics: loss in column 0, accuracy in
  # column 1, with a row for each epoch
  current_gate_results = np.empty(shape = (n_epochs, 2))
  for i in range(n_epochs):
    loss = 0
    accuracy = 0
    for input, target_output in data:
      # Take note of the output
      actual_output = mlp.forward_step(input)
      # Perform backpropagation using the target value
      mlp.backprop_step(target_output)
      # Compute performance metrics
      loss += (target_output - actual_output) ** 2
      accuracy += round(actual_output) == target_output
    # Average performance metrics (add up, then divide by n)
    loss /= n_datapoints
    accuracy /= n_datapoints
    # Store metrics in appropriate row
    current_gate_results[i] = (loss, accuracy)
  # Store metrics table of current gate in list of all gates
  training_results.append((label, current_gate_results))

# Plot our results
fig, axs = plt.subplots(
  nrows = 2, ncols = len(training_results),
  sharex = 'col', squeeze = False, figsize = (12, 4.5)
)
axs[0, 0].set_ylabel('Loss'    , rotation = 0, size = 'large')
axs[1, 0].set_ylabel('Accuracy', rotation = 0, size = 'large')
# Iterate over gates
for i, (label, results) in enumerate(training_results):
  axs[0, i].set_title(label)
  # Plot loss
  axs[0, i].plot(results[:, 0])
  # Plot accuracy
  axs[1, i].plot(results[:, 1])

# Finalize
fig.tight_layout()
plt.show()
