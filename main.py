from NeuralNetwork.NeuralNetwork import NeuralNet
from NeuralNetwork.NNUtils import accuracy, calculateCrossentropyCost
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X,y = make_moons(800,noise=0.15,shuffle=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=True)
costs = []

nn = NeuralNet(X_test.shape,y_test.reshape(-1,1).shape,(14,),0.5)
nn.buildNetwork()
#nn.showNetworkArch()
for i in range(10000):

    nn.forwardPass(X_train)
    cost = calculateCrossentropyCost(y_train,nn.op,X_train.shape[0])
    costs.append(cost)
    nn.calculate_layer_errors(y_train)
    nn.calculate_gradients(X_train.shape[0])
    nn.update_weights()

a = nn.predict(X_test)
print("   Accuracy: {}".format(accuracy(y_test,a)))
print(costs)
