import numpy as np
from Load_MNIST import load_mnist, show_images
import random
import matplotlib.pyplot as plt

# ReLU function
def relu(x):
    return np.maximum(0, x)

# the derivative of ReLU function
def d_relu(x):
    return (x > 0).astype(float)

# the soft-max function
def soft_max(x):
    x = x - np.max(x, axis = 0, keepdims = True)
    exp_activations = np.exp(x)
    return exp_activations / exp_activations.sum(axis = 0)

# given the last activation and difference between soft max and output, return the gradient
def d_cost_d_output(last_activations, difference):
    Jacobian = np.diag(last_activations) - np.outer(last_activations, last_activations)
    return 2 * np.dot(Jacobian.transpose(), difference)

class parameters(object):
    '''
    Contains all the weights and biases for a shallow network.
    '''

    # the inputs are layer 0, neurons are layers 1 to num_neurons, and num_neurons + 1 is the output layer. 
    # the nth weight refers to the weight between nth and n+1th layer, and biases are for that layer
    # If rand is on, uses the He initialization
    def __init__(self, n_input, n_output, num_neurons_list, rand = False):
        self.n_input = n_input
        self.n_output = n_output
        self.n_layers = len(num_neurons_list)
        self.num_neurons_list = num_neurons_list
        self.weights_list = []
        self.biases_list = [None]

        lst = [n_input] + num_neurons_list + [n_output]
        for i in range(self.n_layers + 1):
            neurons = lst[i + 1]
            inputs = lst[i]
            if rand:
                std_dev = np.sqrt(4 / (neurons + inputs))
                weights = np.random.normal(loc = 0, scale = std_dev, size = (neurons, inputs))
            else:
                weights = np.zeros((neurons, inputs))
            biases = np.zeros((neurons, 1))
            self.weights_list.append(weights)
            self.biases_list.append(biases)
        self.weights_list.append(None)
    
    def __repr__(self):
        return "The parameters of a deep neural network with {} inputs, {} outputs, and layers of {} neurons".format(
            self.n_input, self.n_output, str(self.num_neurons_list)[1: -1])

    # add another set of parameters to this set
    def add(self, other):
        for i in range(self.n_layers + 2):
            if i <= self.n_layers:
                self.weights_list[i] = self.weights_list[i] + other.weights_list[i]
            if i > 0:
                self.biases_list[i] = self.biases_list[i] + other.biases_list[i]
    
    # in-place multiplication
    def multiply(self, value: float):
        for i in range(self.n_layers + 2):
            if i <= self.n_layers:
                self.weights_list[i] = self.weights_list[i] * value
            if i > 0:
                self.biases_list[i] = self.biases_list[i] * value
    
    # in-place momentum operation. k is the weight of previous momentum
    def momentum(self, last, k: float = 0.7):
        self.multiply(1 - k)
        last.multiply(k)
        self.add(last)

class picture(object):
    '''
    A picture input with its tag for training. Contains an array of 284 float values and a tag.
    '''

    def __init__(self, pic, tag, prediction = None):
        self.array = pic
        self.tag = tag
        self.prediction = prediction
    
    def __repr__(self):
        return "a picture with the digit {}". format(self.tag)

    # visualize a single graph. If an array of network output is provided, a bar chart of digit probabilities will be drawn.
    def visualize(self, output = None):
        image = np.multiply(self.array.reshape((28, 28)), 256)
        if output is None:
            plt.figure(figsize = (4, 3))
            plt.imshow(image, cmap=plt.cm.gray)
            if not self.prediction:
                plt.title("({})".format(self.tag), fontsize = 8)
            else:
                plt.title("{}: ({})".format(self.prediction, self.tag), fontsize = 8)
        else:
            plt.figure(figsize = (8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap=plt.cm.gray)
            if not self.prediction:
                plt.title("({})".format(self.tag), fontsize = 8)
            else:
                plt.title("{}: ({})".format(self.prediction, self.tag), fontsize = 8)
            plt.subplot(1, 2, 2)
            plt.bar(list(range(10)), list(output.flatten()))
            plt.xlabel('Digit')
            plt.ylabel('Probability')
            plt.title('Probabilities of digits')
        plt.show()

class batch(object):
    '''
    A batch of inputs (pictures and tags). This will be used for training and testing.
    The images input is an array of 284 pixels, the tags input is an array of all digits.
    Note that this is not a list of input objects, but seperate arrays
    '''

    def __init__(self, images, tags, predictions = None):
        self.n_data = images.shape[1]
        self.images = images
        self.tags = tags
        self.predictions = predictions

    def __repr__(self):
        return "A batch of {} images and tags".format(self.n_data)

    # Given a batch index, convert to picture object
    def to_picture(self, index):
        return picture(self.images[:, index], self.tags[index])
    
    # display all the pictures and tags
    def display_all(self):
        images_display = []
        tags_display = []
        for i in range(self.n_data):
            images_display.append(self.images[i])
            if not predictions:
                tags_display.append("[{}]: ({})".format(i, self.tags[i]))
            else: 
                tags_display.append("[{}]: {} ({})".format(i, self.predictions[i], self.tags[i]))
        show_images(images_display, tags_display)
    
    # display a specifed number of pictures and tags
    def display(self, number = 10):
        images_display = []
        tags_display = []
        index = sorted(random.sample(range(0, self.n_data), number))
        for i in index:
            images_display.append(self.images[i])
            if not self.predictions:
                tags_display.append("[{}]: ({})".format(i, self.tags[i]))
            else: 
                tags_display.append("[{}]: {} ({})".format(i, self.predictions[i], self.tags[i]))
        show_images(images_display, tags_display)

class deep_network(object):
    '''
    A deep neural network with arbitrary layers for digit recognition.
    Uses the ReLU function and the square difference cost function.
    The input and output must be between 0 and 1.
    '''

    # set up the neural network, give random numbers for weights and biases
    def __init__(self, num_neurons_list, n_input = 784, n_output = 10):
        self.n_input = n_input
        self.n_output = n_output
        self.n_layers = len(num_neurons_list)
        self.num_neurons_list = num_neurons_list
        self.para: parameters = parameters(n_input, n_output, num_neurons_list, True)

    def __repr__(self):
        return "A deep neural network with {} inputs, {} outputs, and layers of {} neurons".format(
            self.n_input, self.n_output, str(self.num_neurons_list)[1:-1])

    # given a picture / batch, return an array / matrix of outputs
    def compute_output(self, input, is_batch = False):
        if is_batch:
            last_activations = input.images
        else:
            last_activations = input.array
        
        for i in range(1, self.n_layers + 2):
            pre_activation = np.dot(self.para.weights_list[i - 1], last_activations) + self.para.biases_list[i]
            if i == self.n_layers + 1:
                break
            activations = relu(pre_activation)
            last_activations = activations
        return soft_max(pre_activation)

    # given a picture, return an array for the difference between predicted and actual activation (predicted - actual)
    # can pass in output to avoid calculating it again
    def prediction_difference_picture(self, pic, output = None):
        if output is None:
            output = self.compute_output(pic)
        target = np.zeros(self.n_output).reshape((self.n_output, 1))
        target[pic.tag, 0] = 1.0
        return output - target
    
    # given a batch of pictures, return a matrix for the difference between predicted and actual activation (predicted - actual)
    # can pass in output to avoid calculating it again
    def prediction_difference_batch(self, pics, output = None):
        if output is None:
            output = self.compute_output(pics, is_batch = True)
        target = np.zeros((self.n_output, pics.n_data))
        target[pics.tags, np.arange(pics.n_data)] = 1 
        return output - target

    # given a picture, return the cost of this output (difference squared)
    def compute_cost(self, input, is_batch = False):
        if is_batch:
            difference = self.prediction_difference_batch(input)
        else:
            difference = self.prediction_difference_picture(input)
        return (difference ** 2).sum()
    
    # given a picture, return the predicticted digit. Can pass in the output to avoid computing again
    def prediction(self, pic, output = None):
        if not output:
            return np.argmax(self.compute_output(pic))
        else:
            return np.argmax(output)

    # given a batch of pictures, return an array digit. Can pass in the output to avoid computing again
    def prediction_batch(self, pics, output = None):
        if not output:
            output = self.compute_output(pics, is_batch = True)
        return output.argmax(axis = 0)

    # given a picture, return a tuple which contains a list of pre activations and a list of strectched activations
    def compute_activations(self, input, is_batch = True):
        if is_batch:
            activations_list = [input.images]
        else:
            activations_list = [input.array]
        pre_activations_list = [None]

        for i in range(1, self.n_layers + 2):
            pre_activations_list.append(np.dot(self.para.weights_list[i - 1], activations_list[i - 1]) + self.para.biases_list[i])
            if i == self.n_layers + 1:
                activations_list.append(soft_max(pre_activations_list[i]))
            else:
                activations_list.append(relu(pre_activations_list[i]))

        return (pre_activations_list, activations_list)

    # given a single training data point, return the gradient for gradient decsent
    def train_picture(self, pic):
        pre_activations_list, activations_list = self.compute_activations(pic, is_batch = False)
        d_cost_d_activation_list = [None] * (self.n_layers + 2)
        differences = self.prediction_difference_batch(pics, activations_list[-1])
        d_cost_d_activation_list[-1] = d_cost_d_output(activations_list[-1], differences)
        gradients = parameters(self.n_input, self.n_output, self.num_neurons_list)

        for i in range(self.n_layers + 1):
            layer = self.n_layers + 1 - i
            d_relu_pre_activation = d_relu(pre_activations_list[layer])
            gradients.biases_list[layer] = d_relu_pre_activation * d_cost_d_activation_list[layer]
            gradients.weights_list[layer - 1] = np.dot(d_cost_d_activation_list[layer] * d_relu_pre_activation, activations_list[layer - 1].transpose())
            d_cost_d_activation_list[layer - 1] = np.dot(self.para.weights_list[layer - 1].transpose(), d_cost_d_activation_list[layer])
        return gradients

    # given the last_activations difference, compute the gradient of cost over the pre activation of the final layer
    def compute_d_cost_d_activations_batch(self, last_activations, differences, batch_size):
        return 2 * differences
        gradients = np.zeros_like(differences)
        for i in range(batch_size):
            gradients[:, i] = d_cost_d_output(last_activations[:, i], differences[:, i])
        return gradients

    # given a batch of pictures and their tags, use gradient descent to update the neural network, and return the costs and gradients
    def train_batch(self, pics, learning_rate = 0.01, momentum = None, debug = False):
        pre_activations_list, activations_list = self.compute_activations(pics, is_batch = True)
        d_cost_d_activation_list = [None] * (self.n_layers + 2)
        differences = self.prediction_difference_batch(pics, activations_list[-1])
        d_cost_d_activation_list[-1] = self.compute_d_cost_d_activations_batch(activations_list[-1], differences, pics.n_data)
        gradients = parameters(self.n_input, self.n_output, self.num_neurons_list)

        for i in range(self.n_layers + 1):
            layer = self.n_layers + 1 - i
            if layer == self.n_layers + 1:
                d_relu_pre_activation = pre_activations_list[layer]
            else:
                d_relu_pre_activation = d_relu(pre_activations_list[layer])
            gradients.biases_list[layer] = np.sum(d_relu_pre_activation * d_cost_d_activation_list[layer], axis = 1, keepdims = True)
            gradients.weights_list[layer - 1] = np.dot(d_cost_d_activation_list[layer] * d_relu_pre_activation, activations_list[layer - 1].transpose())
            d_cost_d_activation_list[layer - 1] = np.dot(self.para.weights_list[layer - 1].transpose(), d_cost_d_activation_list[layer])
        
        if debug:
            gradients.multiply(1 / pics.n_data)
            return gradients
        gradients.multiply(-learning_rate / pics.n_data)
        if momentum is not None:
            gradients.momentum(momentum)
        self.para.add(gradients)
        return (self.compute_cost(pics, is_batch = True), gradients)

    # given a training set, use SGD to train it
    def train(self, mnist, learning_rate = 0.01, iterations = 1):
        for i in range(iterations):
            trainings = mnist.segment_training(100)
            momentum = None
            sum_cost = 0.0
            for index, batch in enumerate(trainings):
                cost, momentum = self.train_batch(batch, learning_rate, momentum)
                sum_cost += cost
            print("Finished training iteration {} with epoch total cost {:.2f}".format(i + 1, sum_cost))

    # given a list of batches, print out the intial costs and total costs, returning the total costs
    def initial_costs(self, batches):
        sum_cost = 0.0
        for index, batch in enumerate(batches):
            cost = self.compute_cost_batch(batches[index])
            print("Initial cost for batch {} is {:.2f}".format(index, cost))
            sum_cost += cost
        print("Inital total cost is {:.2f}".format(sum_cost))
        return sum_cost

    # given a batch of pictures and their tags, predict and output success rate
    def test(self, pics):
        predictions = self.prediction_batch(pics)
        success = (predictions == pics.tags).sum()
        return success / pics.n_data

    # visualize a neuron by plotting its responsiveness to the inputs. Only the first layer can be visualized
    def visualize_neuron(self, index):
        weights = self.para.weights_list[0][index]
        weights = weights - weights.min()
        weights = weights / weights.max()
        neuron = picture(weights, "Neuron {}".format(index))
        neuron.visualize()

class MNIST_set(object):
    '''
    The MNIST data set, which contains 60,000 training data and 10,000 testing data
    '''

    # load the data set 
    def __init__(self):
        (train_images, train_tags, test_images, test_tags) = load_mnist()
        self.train_images = train_images.reshape((60000,784)).transpose()
        self.train_tags = train_tags
        self.test_images = test_images.reshape((10000,784)).transpose()
        self.test_tags = test_tags
    
    # segment the training data into a list of random batches, 
    def segment_training(self, batch_size = 100):
        indices = np.random.permutation(60000)
        shuffled_data = self.train_images[:,indices]
        shuffled_tags = self.train_tags[indices]
        trainings = [batch(shuffled_data[:, i: i + batch_size],
                    shuffled_tags[i: i + batch_size]) for i in range(0, 60000, batch_size)]
        return trainings
    
    # return a single batch of all training data
    def return_training(self):
        return batch(self.train_images, self.train_tags)

    # return a single batch of all training data
    def return_test(self):
        return batch(self.test_images, self.test_tags)

if __name__ == "__main__":
    mnist = MNIST_set()
    train = mnist.segment_training()
    test = mnist.return_test()
    sample_pic = test.to_picture(1258)
    network = deep_network([200,100,50])

    # run
    print("Initial suceess rate is {:.2f}%".format(network.test(test) * 100))
    network.train(mnist, 0.01, 50)
    network.train(mnist, 0.001, 20)
    print("Final suceess rate for test set is {:.2f}%".format(network.test(test) * 100))