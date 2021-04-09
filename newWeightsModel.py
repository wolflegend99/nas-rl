import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import constants as C
from statistics import mean


class TestModel(nn.Module):
    def __init__(self, input_dims, output_dims, lr, num_layers, num_nodes, trainloader, testloader):
        super(TestModel, self).__init__()
        self.input_dims = input_dims
        self.lr = lr
        self.num_layers = num_layers
        self.output_dims = output_dims
        self.num_nodes = num_nodes
        self.fcs = None
        self.output = None
        self.initialise(num_layers, num_nodes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.trainloader = trainloader
        self.testloader = testloader
        # self.criterion = nn.BCELoss()
        

    def initialise(self, layers, neurons):
        nodes = [neurons]*layers
        hidden_layers = zip(nodes[:-1], nodes[1:])
        self.fcs = nn.ModuleList([nn.Linear(*self.input_dims, neurons)])
        self.fcs.extend([nn.Linear(h1, h2) for h1, h2 in hidden_layers])
        self.output = nn.Linear(neurons, self.output_dims)
        self.num_layers = layers
        self.num_nodes = neurons
        
        
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        #self.reset_optimizer()
    
    def reset_optimizer(self):
        pass
        #self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        

    def forward(self, x):
        for layer in self.fcs:
            x = layer(x)
            x = F.relu(x)
        x = self.output(x)
        return x

    def add_neurons(self, num):

        # Getting the older weights of all layers
        self.num_nodes += num
        self.initialise(self.num_layers, self.num_nodes)
        return [self.num_layers, self.num_nodes]

    def remove_neurons(self, num):
        
        # Getting the older weights of all layers
        fin_neurons = max(self.num_nodes - num,1)
        self.num_nodes = fin_neurons
        self.initialise(self.num_layers, self.num_nodes)
        return [self.num_layers, self.num_nodes]

    def add_layers(self, num):
        self.num_layers += num
        self.initialise(self.num_layers, self.num_nodes)
        return [self.num_layers, self.num_nodes]

    def remove_layers(self, num):
        self.num_layers = max(self.num_layers-num, 1)
        return [self.num_layers, self.num_nodes]

    def print_param(self):
        x = next(self.parameters()).data
        print(x)

    def train(self):

        loss_list, acc_list = [], []
        for epochs in range(C.EPOCHS):
            correct = 0
            total = 0
            train_loss = 0
            loader = self.trainloader
            for data, target in loader:   # print("Target = ",target[0].item())
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.forward(data.float())
                #target = target.type(T.FloatTensor)
                loss = self.criterion(output, target.long().squeeze())
                train_loss += loss.item()*data.size(0)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update running training loss
                
                total += target.size(0)

                # accuracy
                _, predicted = T.max(output.data, 1)
                correct += (predicted == target.squeeze()).sum().item()
            acc_list.append(100*correct/total)
            loss_list.append(train_loss/total)
            #print("Epoch {} / {}: Accuracy is {}, loss is {}".format(epochs,C.EPOCHS,100*correct/total,train_loss/total))
        return acc_list[-1], loss_list[-1]
        #return mean(acc_list[-4:]), mean(loss_list[-4:])
    
    
    def test(self):
        correct = 0
        total = 0
        val_loss = 0
        with T.no_grad():
            for data, target in self.testloader:

            # Predict Output
                output = self.forward(data.float())

            # Calculate Loss
                #target = target.view(-1)
                loss = self.criterion(output, target.squeeze())
                val_loss += loss.item()*data.size(0)
            # Get predictions from the maximum value
                _, predicted = T.max(output.data, 1)

            # Total number of labels
                total += target.size(0)

        # Total correct predictions
                correct += (predicted == target.squeeze()).sum().item()

    # calculate average training loss and accuracy over an epoch
        val_loss = val_loss/len(self.testloader.dataset)
        accuracy = 100 * correct/float(total)
        return accuracy, val_loss