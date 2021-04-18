import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import constants as C

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self, lr, input_dims, output_dims, hidden_layer_array, non_linear_functions, trainloader, testloader): 
        super(Model, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.output = None
        self.fcs = nn.ModuleList()
        self.hidden_layer_array = hidden_layer_array
        self.non_linear_functions = non_linear_functions
        self.num_layers = len(hidden_layer_array)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.trainloader = trainloader
        self.testloader = testloader

    def initialise(self, hidden_layer_array):

        self.hidden_layer_array = hidden_layer_array
        self.num_layers = len(hidden_layer_array)
        self.fcs = nn.ModuleList([nn.Linear(*self.input_dims, hidden_layer_array[0])])

        if(self.num_layers > 1):
            hidden_layers = zip(hidden_layer_array[:-1], hidden_layer_array[1:])
            self.fcs.extend([nn.Linear(h1, h2) for h1, h2 in hidden_layers])

        self.output = nn.Linear(hidden_layer_array[-1], self.output_dims)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def add_neurons(self, num,agent_no):

        # Getting the older weights of all layers
        self.hidden_layer_array[agent_no] += num
        self.initialise(self.hidden_layer_array)
        return self.hidden_layer_array

    def remove_neurons(self, num,agent_no):
        
        # Getting the older weights of all layers
        self.hidden_layer_array[agent_no] = max(self.hidden_layer_array[agent_no] - num,1)
        self.initialise(self.hidden_layer_array)
        return self.hidden_layer_array

    def forward(self, x):
        
        for i in range(self.num_layers):
            x = self.fcs[i](x)
            x = getattr(F, self.non_linear_functions[i])(x)
        x = self.output(x)  
        return x

    def print_param(self):
        x = next(self.parameters()).data
        print(x)
    
    def train(self):

        loss_list, acc_list = [], []
        for epochs in range(C.EPOCHS):
            correct = 0
            total = 0
            train_loss = 0
            for data, target in self.trainloader:   # print("Target = ",target[0].item())

                # send to gpu device
                data, target = data.to(device), target.to(device)

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
 
                # send to gpu device
                data, target = data.to(device), target.to(device)
 
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