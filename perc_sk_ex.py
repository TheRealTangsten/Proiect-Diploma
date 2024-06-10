import numpy as np
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim

import perceptron as perc
def iulianPercExample():
    #
    #
    #
    #sklearn IMPLEMENTATION OF THE PERCEPTRON

    # Set random seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # Load the digits dataset
    X, y = load_digits(return_X_y=True)

    data_train, labels_train = perc.getTrainData()
    data_test, labels_test = perc.getTestData()
    '''
    perm = np.random.permutation(data_train.shape[0])
    data_train = data_train[perm, :]
    labels_train = labels_train[perm]
    '''
    '''
    perm = np.random.permutation(data_test.shape[0])
    data_test = data_test[perm, :]
    labels_test = labels_train[perm]
    '''
    # Train the Perceptron model
    clf = Perceptron(tol=1e-3, random_state=0, shuffle=False)
    perceptron_accuracy = []

    # Pass the classes parameter on the first call to partial_fit
    #clf.partial_fit(X, y, classes=np.unique(y))
    #clf.partial_fit(X, y, classes=np.unique(y))
    clf.partial_fit(data_train, labels_train, classes=np.unique(labels_train))
    #clf.coef_ = np.zeros([3, 2048])
    print(type(clf.coef_))
    print(clf.coef_)
    for epoch in range(30):
        #clf.partial_fit(X, y)
        clf.partial_fit(data_train, labels_train)
        #clf.partial_fit(data_test, labels_test)
        if epoch == 0:
            clf.coef_ = clf.coef_ - clf.coef_
            print(type(clf.coef_))
            print(clf.coef_)
        #perceptron_accuracy.append(clf.score(X, y))
        perceptron_accuracy.append(clf.score(data_test, labels_test))
    print(clf.coef_.shape)
    print(type(clf.coef_))
    #print(perceptron_accuracy)
    #
    #
    #
    #
    #TORCH IMPLEMENTATION OF THE PERCEPTRON

    # Convert numpy arrays to torch tensors
    #X_tensor = torch.tensor(X, dtype=torch.float32)
    #y_tensor = torch.tensor(y, dtype=torch.long)
    X_tensor = torch.tensor(data_train, dtype=torch.float32)
    y_tensor = torch.tensor(labels_train, dtype=torch.long)
    # Create a DataLoader for batch processing
    dataset = TensorDataset(X_tensor, y_tensor)
    #dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

    test_X_tensor = torch.tensor(data_test, dtype=torch.float32)
    test_y_tensor = torch.tensor(labels_test, dtype=torch.long)
    #data_test = TensorDataset(test_X_tensor,test_y_tensor)
    #dataloader = DataLoader(data_test, batch_size=30, shuffle=True)

    # Define the model
    model = nn.Sequential(
        nn.Linear(2048, 3)
        #nn.Linear(64, 10)
    )

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Lists to store accuracy values
    neural_network_accuracy = []
    loss_list = []

    # Train the models
    for epoch in range(30):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss_list.append(np.float32(loss.copy_))
            loss.backward()
            optimizer.step()

        # Evaluate the models
        with torch.no_grad():
            #output = model(X_tensor)
            output = model(test_X_tensor)
            predicted = torch.argmax(output, dim=1)
            #accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
            accuracy = (predicted == test_y_tensor).sum().item() / len(test_y_tensor)
            neural_network_accuracy.append(accuracy)
    print(list(model.parameters())[0].grad)
    # Plot the accuracy curves
    plt.plot(range(1, 31), perceptron_accuracy, label='Perceptron sklearn')
    plt.plot(range(1, 31), neural_network_accuracy, label='Perceptron torch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    print(loss_list)
    plt.plot(loss_list, label='Perceptron torch')
    plt.show()