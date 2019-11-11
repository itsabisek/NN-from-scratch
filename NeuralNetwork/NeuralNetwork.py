import numpy as np
from NeuralNetwork import NNUtils as util


class NeuralNet():

    weight_dict = {}
    bias_dict = {}
    activation_dict = {}
    dactivation_dict = {}
    error_dict = {}
    dW = {}
    dB = {}
    op = None

    def __init__(self,inputSize,outputSize,hiddenLayerSize,learning_rate=0.06):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenLayerSize = hiddenLayerSize
        self.noOfLayers = len(self.hiddenLayerSize) + 2
        self.learning_rate = learning_rate

    #Function that
    def buildNetwork(self):

        inputDims = self.inputSize[1]
        outputClasses = 1 #self.outputSize[1]

        self.weight_dict['12'] = np.random.randn(inputDims, self.hiddenLayerSize[0]) * 0.01
        self.bias_dict['12'] = np.zeros((1, self.hiddenLayerSize[0]))

        self.weight_dict[str(self.noOfLayers - 1) + str(self.noOfLayers)] = np.random.randn(self.hiddenLayerSize[-1], outputClasses) * 0.01
        self.bias_dict[str(self.noOfLayers - 1) + str(self.noOfLayers)] = np.zeros((1, outputClasses))

        for x in range(2, self.noOfLayers - 1):
            self.weight_dict[str(x) + str(x + 1)] = np.random.randn(self.hiddenLayerSize[x - 2], self.hiddenLayerSize[x - 1]) * 0.01
            self.bias_dict[str(x) + str(x + 1)] = np.zeros((1, self.hiddenLayerSize[x - 1]))


    def forwardPass(self,X):
        self.op = X
        self.activation_dict['1'] = X

        for layer in range(1, self.noOfLayers):
            #print("Enter op size: {}".format(self.op.shape))
            z = np.dot(self.op, self.weight_dict[str(layer) + str(layer + 1)]) + self.bias_dict[str(layer) + str(layer + 1)]
            if layer != self.noOfLayers - 1:
                self.op = util.relu(z)
                self.dactivation_dict[str(layer + 1)] = util.drelu(self.op)
                self.activation_dict[str(layer + 1)] = self.op
            else:
                self.op = util.sigmoid(z)
                self.dactivation_dict[str(layer + 1)] = util.dsigmoid(self.op)
                self.activation_dict[str(layer + 1)] = self.op

            #print("Exit op size: {}".format(self.op.shape))


    def calculate_layer_errors(self,y):

        for layer in reversed(range(1, self.noOfLayers)):
            if layer == self.noOfLayers - 1:
                self.error_dict[str(layer + 1)] = self.op - y.reshape(-1, 1)
                continue

            self.error_dict[str(layer + 1)] = np.dot(self.error_dict[str(layer + 2)],
                                            self.weight_dict[str(layer + 1) + str(layer + 2)].T) \
                                            * self.dactivation_dict[str(layer + 1)]


    def calculate_gradients(self,noOfTrainingSamples):

        for layer in range(1, self.noOfLayers):
            key = str(layer) + str(layer+1)
            self.dW[key] = 1 / noOfTrainingSamples * np.dot(self.activation_dict[str(layer)].T,self.error_dict[str(layer+1)])
            self.dB[key] = 1 / noOfTrainingSamples * np.sum(self.error_dict[str(layer + 1)], axis=0, keepdims=True)


    def update_weights(self):
        updated_weights = {}
        updated_bias = {}
        for key in self.weight_dict.keys():
            updated_weights[key] = self.weight_dict[key] - self.learning_rate * self.dW[key]
            updated_bias[key] = self.bias_dict[key] - self.learning_rate * self.dB[key]

        self.weight_dict = updated_weights
        self.bias_dict = updated_bias


    def predict(self,X_test):
        op = X_test
        for layer in range(1, self.noOfLayers):
            z = np.dot(op, self.weight_dict[str(layer) + str(layer + 1)]) + self.bias_dict[str(layer) + str(layer + 1)]
            if layer == self.noOfLayers-1:
                op = util.sigmoid(z)
                return op

            op = util.relu(z)

    def showNetworkArch(self):
        if self.inputSize[1] == 0 or self.inputSize == None and self.outputSize == None or self.outputSize[0] == 0:
            print("Invalid input/output sizes")
            return

        print("The network has {} input dims and {} output dims".format(self.inputSize[1],self.outputSize[0]))
        for key in self.weight_dict.keys():
            print(self.weight_dict[key].shape)

