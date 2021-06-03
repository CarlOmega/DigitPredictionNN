import numpy as np
import random
import sys

# /////////////////////////////////////////////////////////////////////////////////////////////////
# Small Functions Start
# /////////////////////////////////////////////////////////////////////////////////////////////////
def sigmoid(net):
    return 1/(1+np.exp(-net))

def softmax(net):
    exps = np.exp(net)
    return exps / np.sum(exps, axis=1, keepdims=True)

def sigmoid_prime(out):
    return out*(1-out)

def cost(target, Y):
    return 0.5*(target-Y)**2

def read_in_data(train_data_file, train_label_file, test_data_file):
    print("Reading in:", train_data_file, train_label_file, test_data_file)
    train_data = np.loadtxt(train_data_file, delimiter=',')
    train_label = np.array(list(map(lambda x: [1.0 if x==float(i) else 0.0 for i in range(10)],
                np.loadtxt(train_label_file, delimiter=','))))
    test_data = np.loadtxt(test_data_file, delimiter=',')
    print("Done")
    return train_data, train_label, test_data
# /////////////////////////////////////////////////////////////////////////////////////////////////
# Small Functions End
# /////////////////////////////////////////////////////////////////////////////////////////////////

# /////////////////////////////////////////////////////////////////////////////////////////////////
# display_resultss Start
# /////////////////////////////////////////////////////////////////////////////////////////////////
def display_results(network, data):
    """To display visually what it is outputting.

        Loops through the outputs and displays the data,
        and what it thinks the data was.

        Args:
            network (Neural_Network): Trained network.
            data (float)[][]: data to feed through and check results.
    """
    output = network.forward(data)
    data = list(data)
    for x in range(len(data)):
        for i in range(28):
            for j in range(28):
                if data[x][28*i+j] >= 0.90:
                    print("X", end="")
                elif data[x][28*i+j]>= 0.75:
                    print("x", end="")
                elif data[x][28*i+j]>= 0.5:
                    print("*", end="")
                elif data[x][28*i+j]>= 0.25:
                    print(".", end="")
                else:
                    print(" ", end="")
            print("")
        print(list(output[x]).index(max(list(output[x]))))
        print(list(output[x]))
# /////////////////////////////////////////////////////////////////////////////////////////////////
# display_resultss End
# /////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////
# Neural_Network Class Start
# /////////////////////////////////////////////////////////////////////////////////////////////////
class Neural_Network(object):
    """Neural Network Class that uses either sum of square error or cross entropy.

    Initialises a new network with given hyperparameters, radnom initial
    weights are set. Use self.learn() to train the network. Once trained
    use self.forward() to make predictions.

    Attributes:
        inputs (int): Neurons on the input layer.
        hidden (int): Neurons on the hideen layer.
        output (int): Neurons on the output layer.
        error_type (string): Either "Cross" or "Square".
        weights_l1  (float)[][]: Weights on layer 1.
        weights_l2  (float)[][]: Weights on layer 2.
        bias_l1  (float)[][]: Biases on layer 1.
        bias_l2  (float)[][]: Biases on layer 2.

    """
    def __init__(self, inputs, hidden, outputs, error_type):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.error_type = error_type

        self.weights_l1 = np.random.randn(self.inputs, self.hidden)
        self.weights_l2 = np.random.randn(self.hidden, self.outputs)
        self.bias_l1 = np.random.randn(self.hidden)
        self.bias_l2 = np.random.randn(self.outputs)

    def forward(self, X):
        """Passes input data through the network and returns predictions.

        Passes inputs through the network using sigmoid
        activations, unless error_type = "Cross" then output layer
        activation uses softmax.

        Args:
            X (float)[][]: Input data to feed through the network.

        Returns:
            (float)[][]: Predictions of the network after feeding.

        """
        #neth = XW(1)
        #outh = sigmoid(neth)
        #neto = outhW(2)
        #outo = sigmoid(neto) or softmax(neto)
        self.net_h = np.dot(X, self.weights_l1)+self.bias_l1
        self.out_h = sigmoid(self.net_h)
        self.net_o = np.dot(self.out_h, self.weights_l2)+self.bias_l2
        # Change in Error type
        if self.error_type == "Cross":
            self.out_o = softmax(self.net_o)
        elif self.error_type == "Square":
            self.out_o = sigmoid(self.net_o)
        return self.out_o



    def learn(self, X, Y, batch_size, rate, epoch):
        """Trains the network with given train data, tracks accuracy on test data.

        Given different parameters it cycles epochs updating the weights after
        each iteration of a minibatch. Randomly shuffles the data before
        separating into mini_batches.

        Args:
            X (float)[][]: Input data to train the network.
            Y (float)[][]: Input label to train the network.
            batch_size (int): Size of mini batches.
            rate (float): Learning rate usually represented as eta.
            epoch (int): Amount of interations the network should be trained on the data.
            ?test_data (float)[][]: Input data to get accuracy of the network. (to gauge accuracy)
            ?test_label (float)[][]: Input data to get accuracy of the network. (to gauge accuracy)

        """
        for i in range(epoch):
            # Shuffle training data to train more generally
            c = list(zip(X, Y))
            random.shuffle(c)
            X, Y = zip(*c)
            X = np.array(X)
            # Split mini batches up
            mini_batches = [(X[i*batch_size:(i+1)*batch_size], Y[i*batch_size:(i+1)*batch_size]) for i in range(len(X)//batch_size)]
            for mini_batch in mini_batches:
                # Back propigate to get gradients
                dE_totaldB2, dE_totaldW2, dE_totaldB1, dE_totaldW1 = self.back_prop(mini_batch[0], mini_batch[1])
                # Apply changes to the weights
                self.weights_l1 = self.weights_l1 - (rate/float(batch_size))*dE_totaldW1
                self.weights_l2 = self.weights_l2 - (rate/float(batch_size))*dE_totaldW2
                self.bias_l1 = self.bias_l1 - (rate/float(batch_size))*dE_totaldB1
                self.bias_l2 = self.bias_l2 - (rate/float(batch_size))*dE_totaldB2
            print("Epoch:", i)
            # Test lines/debugging
            # self.show_weights()
            # print(self.evaluate(test_data, test_label))


    def back_prop(self, X, Y):
        """After feeding forward gradients are calucalted.

        Using Backpropagation grident values are calucalted over a give
        mini batch. The sums are added using dot product and sum.
        These values are used to update the weights and Biases

        Args:
            X (float)[][]: Input data to train the network.
            Y (float)[][]: Input label to train the network.

        Returns:
            (float)[][]: Gradients for new Biases layer 2.
            (float)[][]: Gradients for new weights layer 2.
            (float)[][]: Gradients for new Biases layer 1.
            (float)[][]: Gradients for new weights layer 2.

        """
        self.forward(X)
        if self.error_type == "Cross":
            dE_totaldnet_o = (self.out_o-Y)
        else:
            dE_totaldnet_o = np.multiply((self.out_o-Y), sigmoid_prime(self.out_o))
        dE_totaldB2 = np.sum(dE_totaldnet_o, axis=0)
        dE_totaldW2 = np.dot(self.out_h.T, dE_totaldnet_o)

        dE_totaldnet_h = np.dot(dE_totaldnet_o, self.weights_l2.T)*sigmoid_prime(self.out_h)
        dE_totaldB1 = np.sum(dE_totaldnet_h, axis=0)
        dE_totaldW1 = np.dot(X.T, dE_totaldnet_h)

        return dE_totaldB2, dE_totaldW2, dE_totaldB1, dE_totaldW1


    def test_part_1(self):
        # Only used to compare manual results. Forces parameters
        self.inputs = 2
        self.hidden = 2
        self.outputs = 2

        self.weights_l1 = np.array([[0.1, 0.2], [0.1, 0.1]])
        self.weights_l2 = np.array([[0.1, 0.1], [0.1, 0.2]])
        self.bias_l1 = np.array([[0.1, 0.1]])
        self.bias_l2 = np.array([[0.1, 0.1]])
        rate = 0.1
        batch_size = 2
        X = np.array([[0.1, 0.1], [0.1, 0.2]])
        Y = [[1.0, 0.0], [0.0, 1.0]]
        self.learn(X, Y, 2, 0.1, 3, X, Y)

    def evaluate(self, test_data, test_label):
        # Used during testing to compare results to labels to gain accuracy.
        output = self.forward(test_data)
        count = 0
        for i in range(len(output)):
            if list(test_label[i]).index(max(list(test_label[i]))) == list(output[i]).index(max(list(output[i]))):
                count += 1
        return count/float(len(output))

    def show_weights(self):
        # using in test part 1 to view weights.
        for i in self.weights_l1:
            for j in i:
                print(j)
        for i in self.weights_l2:
            for j in i:
                print(j)
        for i in self.bias_l1:
            for j in i:
                print(j)
        for i in self.bias_l2:
            for j in i:
                print(j)


# /////////////////////////////////////////////////////////////////////////////////////////////////
# Neural_Network Class End
# /////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////
# Main Start
# /////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":
    if len(sys.argv) == 8:
        print(len(sys.argv))
        try:
            NInput = int(sys.argv[1])
            NHidden = int(sys.argv[2])
            NOutput = int(sys.argv[3])
            train_data_file = sys.argv[4]
            train_label_file = sys.argv[5]
            test_data_file = sys.argv[6]
            test_output_file = sys.argv[7]
            # "TrainDigitX.csv.gz", "TrainDigitY.csv.gz", "TestDigitX.csv.gz", "TestDigitY.csv.gz", "TestDigitX2.csv.gz"
            data, label, test_data = read_in_data("TrainDigitX.csv.gz", "TrainDigitY.csv.gz", "TestDigitX.csv.gz")
            NN = Neural_Network(NInput, NHidden, NOutput, "Square")
            NN.learn(data, label, 20, 3.0, 30)
            output = NN.forward(test_data)
            display_results(NN, test_data)
            # np.savetxt(test_output_file, output, delimiter=',')
        except Exception as e:
            print("Incorrect read in.", e)
# NOTE: Ran low on time to completely polish code. Had to change it to fit assignment specs.
# /////////////////////////////////////////////////////////////////////////////////////////////////
# Main End
# /////////////////////////////////////////////////////////////////////////////////////////////////
