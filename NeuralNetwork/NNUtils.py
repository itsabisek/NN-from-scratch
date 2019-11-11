import numpy as np

def relu(z):
    z[z<0] = 0
    return z


def drelu(a):
    a[a>0] = 1
    return a



def sigmoid(z):
    return np.divide(1,(1+np.power(np.e,-z)))




def dsigmoid(a):
    return a * (1-a)


def accuracy(y, a):
    return np.squeeze(np.dot(y.T, a) + np.dot((1 - y).T, (1 - a))) / y.shape[0] * 100


def calculateCrossentropyCost(y, a, noOfTrainingSamples, weights=0):
    cost = (-1 / noOfTrainingSamples) * (np.dot(y.T, np.log(a)) + np.dot((1 - y).T, np.log(1 - a)))
    return np.squeeze(cost)
1