import sys
import numpy as np
import pdb
import math
import random
quadraticCost = False



class NeuralNetwork:
    def __init__(self, inputLayer = 784, hLayer = 30, oLayer = 10, eta = 3.0, m = 20, epoch = 30):
        self.inputSize = inputLayer
        self.hiddenSize = hLayer
        self.outSize = oLayer

        self.eta = eta  # learning rate
        self.m = m  # minibatchsize
        self.epoch = epoch  # epoch

        # initialize random weights and biases for NN

        w = [np.random.randn(hLayer, inputLayer), np.random.randn(oLayer, hLayer)]
        self.w = w

        b = [np.random.randn(hLayer, 1), np.random.randn(oLayer, 1)]
        self.b = b

    def forPropagate(self, data):
        data = np.array(data).reshape((len(data), 1))
        # a2 is the activated hidden layer
        a2 = sigmoid(np.dot(self.w[0], data) + self.b[0])

        yhat = sigmoid(np.dot(self.w[1], a2) + self.b[1])

        # return the activated output(yhat)
        return yhat

    def iterate(self, mB):
        djdb = [np.zeros(bL.shape) for bL in self.b]        #Variables to contain the errors for bias and weights
        djdw = [np.zeros(wL.shape) for wL in self.w]

        for x, y in mB:
            # Run back propagation to find errors
            newErrorbias, newErrorweight = self.backPropagation(x, y)
            for level, errors in enumerate(newErrorbias):
                djdb[level] = djdb[level] + errors
            for level, errors in enumerate(newErrorweight):
                djdw[level] = djdw[level] + errors

        # nm is the factor of the learning rate / mini batch size for later application to the stochastic gradient descent
        nm = (float(self.eta) / self.m)


        for index, array in enumerate(self.w):
            self.w[index] = array - (nm * djdw[index])

        # Finding Hidden Layer biases:
        self.b[0] = self.b[0] - (nm * djdb[0])
        # Finding Output Layer biases:
        self.b[1] = self.b[1] - (nm * djdb[1])

    def backPropagation(self, input, Lbl):
        input = np.array(input).reshape((len(input), 1))
        a = [input]
        # hidden layer activations
        mA = sigmoid(np.dot(self.w[0], input) + self.b[0])
        a.append(mA)

        oA = sigmoid(np.dot(self.w[1], mA) + self.b[1])
        a.append(oA)


        dCostA = costFunction(oA, Lbl)
        oGradient = oA * (1 - oA)
        outB = oGradient * dCostA

        outErrw = np.zeros(self.w[1].shape)
        # calculate weights from hidden-output
        for i in range(0, self.outSize):
            # Output layer errb
            outputLayerErrb = outB[i]
            oErrw = mA * outputLayerErrb
            outErrw[i] = oErrw.reshape((len(oErrw),))

        hErrb = []
        hiddenGradient = mA * (1 - mA)
        #find hidden layer bias errors
        for num in range(0, self.hiddenSize):
            weights = []

            for i in range(0, self.outSize):
                weights.append(self.w[1][i][num])

            err = np.dot(weights, outB) * hiddenGradient[num]
            hErrb.append(err)
        hErrb = np.array(hErrb).reshape((self.hiddenSize, 1))

        hiddenLayerErrws = np.zeros(self.w[0].shape)
        # calculate weights using hidden layer inputs
        for i in range(0, self.hiddenSize):
            hiddenErrb = hErrb[i]
            hErrw = input * hiddenErrb
            hiddenLayerErrws[i] = hErrw.reshape((len(hErrw),))

        return [hErrb, outB], [hiddenLayerErrws, outErrw]

    def train(self, trainingData, testingData=None):
        for index in range(1, self.epoch + 1):      #Train epoch many times and however many iterations exist for the size of data/mini batch size
            batches = []
            random.shuffle(trainingData)
            for Xi in range(0, len(trainingData), self.m):
                batches.append(trainingData[Xi: Xi + self.m])
            for batch in batches:
                self.iterate(batch)
            if testingData:
                self.getAccuracy(testingData)
            print("Epoch no. {num} finished".format(num=index))

    def getAccuracy(self, testData):        #Function to find the accuracy of the algorith,
        count = 0

        for testExample in testData:
            data = testExample[0]
            Lbl = int(testExample[1])
            oA = self.forPropagate(data)    #after training for each epoch, find the classification accuracy

            if np.argmax(oA) == Lbl:
                count += 1
        accuracy = count / len(testData)
        print("Accuracy: %s" % accuracy)

    def generatePredictions(self, testX, FileName):       #Create predictions file
        if FileName:
            predictions = [np.argmax(self.forPropagate(data)) for data in testX]
            np.savetxt(FileName, predictions, fmt='%i', delimiter=',')

def sigmoid(z):         #sigmoid activation function
    return 1.0 / (1.0 + np.exp(-z))

def costFunction(a, actual):        #cost function for either quadratic or cross entropy(else)
    if quadraticCost:
        # (a - y) at the index where it's classified
        temp = np.copy(a)
        temp[int(actual)] -= 1
        return temp
    else:
        temp = np.copy(a)
        vL = Vectorize(actual)
        return (-1.0) * ((vL / (1.0 * temp)) + (vL - 1.0) * (1.0 / 1.0 - temp))

def Vectorize(Y): #convert y output to 0 - 9
    vL = np.zeros((10, 1))
    vL[int(Y)] = 1.0
    return vL

#Retrieve input and process

args = input("Please enter the size of input layer, hidden layer and output layer: \n")
inLayer, hidLayer, outLayer = args.split(" ")
data = input("Please enter the trainingdata, traininglabeldata, testingdata file names respectively: \n")
train,trainlabel,test = data.split(" ")

inSize = int(inLayer)
hSize = int(hidLayer)
oSize = int(outLayer)
trainX = np.loadtxt(train, delimiter=',')
trainY = np.loadtxt(trainlabel, delimiter=',')
testingX = np.loadtxt(test, delimiter=',')
testingY = np.loadtxt('TestDigitY.csv.gz', delimiter=',')


    # Defaults learning rate, minibatchsize, epoch
eta = 3
m = 20
epoch = 30

trainset = list(zip(trainX, trainY))
testset = list(zip(testingX, testingY))

nn = NeuralNetwork(inSize, hSize, oSize, eta, m, epoch)     #Create NN object
nn.train(trainset, testset)                                 #Train NN object
nn.generatePredictions(testingX, "PredictDigitY2.csv.gz")

#TrainDigitX.csv.gz TrainDigitY.csv.gz TestDigitX.csv.gz