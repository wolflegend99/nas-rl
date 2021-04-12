#
# debugs and tests go here
#
from variableModel import Model
from dataset import Dataset
import helper as H
import constants as C

import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

use_cuda = T.cuda.is_available()
device = T.device("cuda:0" if use_cuda else "cpu")
'''
print(T.__version__)
print(T.cuda.is_available())
'''
dataset = Dataset()
X, y = dataset.preprocess()
X_train, X_test, y_train, y_test = dataset.split(X, y, fraction=0.3)
X_train, X_test = dataset.scale(X_train, X_test)
trainloader, testloader = H.load(X_train, X_test, y_train, y_test)
mVar = Model(0.01, [12], 2, [8,6,6], ['relu', 'relu', 'relu'], trainloader, testloader).to(device)

#mVar = mVar.to(device)
train_acc, train_loss = mVar.train()
test_acc, test_loss = mVar.test()
print(train_acc, train_loss)
print(test_acc, test_loss)

mVar.initialise([8,6])
mVar = mVar.to(device)

train_acc, train_loss = mVar.train()
test_acc, test_loss = mVar.test()
print(train_acc, train_loss)
print(test_acc, test_loss)


#print(next(iter(trainloader)))
# m = TestModel([4], 2, 0.01, 3, 3, trainloader=[], testloader=[])
# print(m(imp))
