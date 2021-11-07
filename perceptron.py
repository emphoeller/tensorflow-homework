import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
        
    return 1 / (1 + np.exp(-x))
    
def sigmoid_prime(x):

    return sigmoid(x)*(1-sigmoid(x))


class Perceptron(object):

    def __init__(self, num_inputs, alpha=1):

        self.num_inputs = num_inputs
        self.alpha = alpha 

        self.weights = np.random.randn(num_inputs + 1)
    
    def forward_step(self, inputs):

        self.input = np.insert(inputs, 0, 1)
     
        self.activation = np.dot(self.weights, self.input)
        #print('activation:', self.activation)
        self.output = sigmoid(self.activation)
        #print('output:', self.output)

        return self.output

    def update_step(self, delta): 

        gradients = delta * self.input
        #print('gradients:', gradients)
        self.weights += self.alpha * gradients
        #print('weights:', self.weights)
        

class MLP(object):


    def __init__(self, hidden_layer=[1,4], num_ouput=1):

        self.num_output = num_ouput
        self.hidden_layer = hidden_layer

        self.hidden_layer = [Perceptron(2) for i in range(4)]
        self.num_output = Perceptron(4)
    
    def mse(self, output, target):

        return (output - target) ** 2

    def forwardpropagate(self, inputs):
        #print('inputs:', inputs)
        self.hidden_layer_output = np.array([p.forward_step(inputs) for p in self.hidden_layer])
        #print('hidden_layer_output:', self.hidden_layer_output)

        return self.num_output.forward_step(self.hidden_layer_output)

    def backpropagate(self, target):

        delta_output = (target - self.num_output.output) * sigmoid_prime(self.num_output.activation)
        
        for i, p in enumerate(self.hidden_layer):   
            #print('delta_output:', delta_output)
            delta = (delta_output * self.num_output.weights[i]) * sigmoid_prime(p.activation)
            #print('delta:', delta)
            p.update_step(delta)

        self.num_output.update_step(delta_output)

    def training(self, inputs, label, epochs, title):
        
        all_errors = []
        all_accs = []

        label = label.reshape((4,1))
        sum_error = 0

        for i in range(epochs):
            correct = 0
            error = 0

            for x1, x2, target in np.concatenate((inputs, label), axis=1):
                pred = self.forwardpropagate([x1, x2])
                #print('pred:', pred)
                self.backpropagate(target)
 
                error = self.mse(pred, target)
                sum_error += error

                if round(pred) == target:
                   correct += 1

            
            accuracy = correct / len(inputs)

            sum_error = sum_error / len(inputs)
            all_errors.append(sum_error)
            all_accs.append(accuracy)

            print("Error: {} at epoch {}".format(sum_error, i+1))
            print("Accuracy: {} at epoch {}".format(accuracy, i+1))
        x = range(epochs)
        plt.plot(x, all_accs, label="Accuracy")
        plt.plot(x, all_errors, label="Loss")
        plt.title(title)
        plt.legend()
        plt.show()
        print("Training complete!")
        print("=====")

if __name__ == "__main__":

    input = np.array([[True, True], [False, True], [True, False], [False, False]])

    label_and = np.array([1,0,0,0])
    label_or = np.array([1,1,1,0])
    label_not_or = np.array([0,0,0,1])
    label_not_and = np.array([0,1,1,1])
    label_xor = np.array([0,1,1,0])

    mlp = MLP() 

    mlp.training(input, label_and, 1000, 'AND')
    mlp.training(input, label_or, 1000, 'OR')
    mlp.training(input, label_not_or, 1000, 'NOT OR')
    mlp.training(input, label_not_and, 1000, 'NOT AND')
    mlp.training(input, label_xor, 1000, 'XOR')
    

   

    



        






